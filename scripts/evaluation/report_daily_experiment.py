"""Verify completed daily point/NLL experiments and write a local Chinese report.

Only reads existing predictions and checkpoints; never trains or selects a model.
"""
import argparse
import base64
from datetime import datetime, timezone, timedelta
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_paths import configure_imports, dataset_paths, project_path
configure_imports()
from wip_pred.daily.metrics import point_metrics, distribution_metrics, distribution_summary
from wip_pred.daily.runner import code_hashes, file_hash


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')


def read_frame(path):
    frame = pd.read_csv(path)
    for name in ('forecast_date', 'forecast_at', 'last_snapshot_at', 'first_snapshot_at',
                 'completion_at', 'label_available_at'):
        if name in frame:
            frame[name] = pd.to_datetime(frame[name])
    return frame


def assert_close(a, b, label):
    if not np.isclose(a, b, rtol=1e-5, atol=1e-5):
        raise ValueError(f'{label}: recomputed {a} differs from saved {b}')


def same_samples(expected, observed):
    columns = ['sample_id','lot_id','product_id','forecast_date','forecast_at','last_snapshot_at',
               'first_snapshot_at','completion_at','label_available_at','remaining_days',
               'sequence_length','no_snapshot_today','stale_fallback','known_quantity']
    pd.testing.assert_frame_equal(expected[columns].reset_index(drop=True),
                                  observed[columns].reset_index(drop=True), check_dtype=False)


def validate_mode(run, mode, data_manifest, index):
    folder = run/'training'/mode
    state = read_json(folder/'training_state.json')
    config = read_json(folder/'daily_config.json')
    if state['status'] != 'completed' or state['test_evaluated_during_training']:
        raise ValueError(f'{mode}: incomplete training or test used during training')
    if config['data_manifest'] != data_manifest:
        raise ValueError('Modes did not use identical training data/protocol/source records.')
    for name, field in [('best_model.pth','best_model_sha256'), ('daily_config.json','config_sha256'),
                        ('preprocessor.pkl','preprocessor_sha256')]:
        if file_hash(folder/name) != state[field]:
            raise ValueError(f'{mode}: {name} no longer matches its training record')
    history = pd.read_json(folder/'training.jsonl', lines=True)
    if len(history) != state['epochs_completed']:
        raise ValueError(f'{mode}: epoch count differs from training state')
    selected = history.loc[history.validation_score.idxmin()]
    if int(selected.epoch) != state['best_epoch']:
        raise ValueError(f'{mode}: selected epoch differs from validation-only minimum')
    assert_close(selected.validation_score, state['best_validation_score'], 'best log score')
    result, frames = {}, {}
    spec = config['target_spec']
    h = spec['horizon']
    for split in ('validation','test'):
        if split == 'validation':
            frame = read_frame(folder/'validation_predictions.csv')
            saved = read_json(folder/'validation_metrics.json')
        else:
            evaluation = run/'evaluation'/mode
            metadata = read_json(evaluation/'test_metrics.json')
            if metadata['used_for_model_selection'] or metadata['smoke_only'] != data_manifest['smoke_only']:
                raise ValueError('Test evaluation has invalid scope metadata')
            saved = metadata['metrics']
            frame = read_frame(evaluation/'test_predictions.csv')
        same_samples(index.loc[index.split == split], frame)
        if mode == 'point':
            predicted = frame.predicted_remaining_days.to_numpy()
            if not np.allclose(predicted, frame.raw_predicted_remaining_days.clip(lower=0)):
                raise ValueError('Point postprocessing differs from configured nonnegative clipping')
            recalculated = point_metrics(frame, predicted)
            for key in recalculated:
                assert_close(recalculated[key], saved[key], f'{mode}/{split}/{key}')
            recalculated['negative_raw_predictions'] = int((frame.raw_predicted_remaining_days < 0).sum())
            if split == 'validation':
                assert_close(recalculated['mae_days'], state['best_validation_score'], 'selected MAE')
        else:
            columns = [*[f'p_day_{day}' for day in range(1,h+1)], 'p_tail']
            probabilities = frame[columns].to_numpy()
            distribution_summary(probabilities, h)  # Validates support and normalization.
            classes = np.minimum(frame.remaining_days.to_numpy(dtype=int)-1, h)
            nll = -np.log(np.maximum(probabilities[np.arange(len(frame)),classes], np.finfo(float).tiny))
            recalculated = distribution_metrics(frame, probabilities, h, nll)
            recalculated.update(interval_width_metrics(probabilities,h))
            for key in ('coarsened_nll','lot_macro_coarsened_nll','central_90_class_interval_coverage',
                        'central_90_finite_upper_fraction','actual_tail_fraction','mean_predicted_tail_probability'):
                assert_close(recalculated[key], saved[key], f'{mode}/{split}/{key}')
            if split == 'validation':
                assert_close(recalculated['coarsened_nll'], state['best_validation_score'], 'selected NLL')
            frame['coarsened_nll'] = nll
        recalculated['by_remaining_days'] = saved.get('by_remaining_days',{})
        result[split], frames[split] = recalculated, frame
    result.update(state=state, target_spec=spec,
                  parameters=sum(v.numel() for v in __import__('torch').load(folder/'best_model.pth', map_location='cpu', weights_only=True).values()),
                  execution_minutes=(folder.joinpath('training_state.json').stat().st_mtime-folder.joinpath('daily_config.json').stat().st_mtime)/60,
                  duration_definition='Approximate wall time from config save to completed state save, including final checkpoint verification.')
    return result, frames, history


def fit_conditional_cycle(index, h):
    """Fit unique-Lot cycle histories; never inspect held-out labels.

    Fixed smoothing: global conditional class counts receive Laplace(1).
    Product counts receive one pseudo-Lot distributed as that global prior.
    These constants are not tuned on validation or test outcomes.
    """
    training = index.loc[index.split == 'train'].copy()
    if training.empty or h < 1:
        raise ValueError('Conditional baseline requires training Lots and positive horizon.')
    fields = ['product_id', 'first_snapshot_at', 'completion_at']
    if training.groupby('lot_id')[fields].nunique(dropna=False).gt(1).any().any():
        raise ValueError('Inconsistent product or cycle boundaries within a training Lot.')
    lots = training.drop_duplicates('lot_id').copy()
    lots['cycle_days'] = (lots.completion_at.dt.normalize()-lots.first_snapshot_at.dt.normalize()).dt.days
    if lots.cycle_days.isna().any() or (lots.cycle_days < 1).any():
        raise ValueError('Training cycles must be positive integer calendar days.')
    classes = np.minimum(training.remaining_days.to_numpy(dtype=int)-1, h)
    if np.any(classes < 0):
        raise ValueError('Training remaining days must be positive.')
    counts = np.bincount(classes, minlength=h+1)
    return {'horizon': h, 'cycles': lots.cycle_days.to_numpy(dtype=int),
            'by_product': {product: group.cycle_days.to_numpy(dtype=int)
                           for product, group in lots.groupby('product_id')},
            'frequency_fallback': (counts+1)/(counts.sum()+h+1),
            'product_prior_strength': 1.0, 'global_laplace_alpha': 1.0}


def predict_conditional_cycle(fitted, features):
    """P(T-e in class | product, T>e), using only available forecast features."""
    h = fitted['horizon']
    elapsed = (features.forecast_date.dt.normalize()-features.first_snapshot_at.dt.normalize()).dt.days
    if elapsed.isna().any() or (elapsed < 0).any():
        raise ValueError('Forecast dates cannot precede first observations.')
    global_cache, product_cache, rows, metadata = {}, {}, [], []
    for product, age in zip(features.product_id, elapsed):
        age = int(age)
        if age not in global_cache:
            remaining = fitted['cycles']-age
            remaining = remaining[remaining > 0]  # Known unfinished at this date.
            n_global = len(remaining)
            counts = np.bincount(np.minimum(remaining-1, h), minlength=h+1)
            alpha = fitted['global_laplace_alpha']
            prior = ((counts+alpha)/(n_global+alpha*(h+1)) if n_global
                     else fitted['frequency_fallback'])
            global_cache[age] = (prior, n_global)
        prior, n_global = global_cache[age]
        key = (product, age)
        if key not in product_cache:
            remaining = fitted['by_product'].get(product, np.array([], dtype=int))-age
            remaining = remaining[remaining > 0]
            n_product = len(remaining)
            counts = np.bincount(np.minimum(remaining-1, h), minlength=h+1)
            strength = fitted['product_prior_strength']
            probabilities = (counts+strength*prior)/(n_product+strength)
            source = ('product_conditional_smoothed' if n_product else
                      'global_conditional_fallback' if n_global else 'global_frequency_fallback')
            product_cache[key] = (probabilities, n_product, source)
        probabilities, n_product, source = product_cache[key]
        rows.append(probabilities)
        metadata.append({'conditional_elapsed_days': age,
                         'conditional_product_support_lots': n_product,
                         'conditional_global_support_lots': n_global,
                         'conditional_source': source})
    return np.asarray(rows).reshape(-1, h+1), pd.DataFrame(metadata)


def interval_width_metrics(probabilities, h):
    summaries = distribution_summary(probabilities, h)
    finite = np.isfinite(summaries['q95_upper_days'])
    widths = summaries['q95_lower_days']-summaries['q05_lower_days']
    return {'central_90_mean_finite_width_days': float(widths[finite].mean()) if finite.any() else None}


def baselines(index, h):
    """Fixed training-only point, unconditional and conditional baselines."""
    training = index.loc[index.split == 'train'].copy()
    lots = training.drop_duplicates('lot_id').copy()
    lots['cycle_days'] = (lots.completion_at.dt.normalize()-lots.first_snapshot_at.dt.normalize()).dt.days
    means = lots.groupby('product_id').cycle_days.mean()
    fallback = float(lots.cycle_days.mean())
    classes = np.minimum(training.remaining_days.to_numpy(dtype=int)-1, h)
    counts = np.bincount(classes, minlength=h+1)
    empirical = (counts+1)/(len(classes)+h+1)  # Fixed Laplace(1) smoothing.
    conditional_model = fit_conditional_cycle(index, h)
    results, frames = {}, {}
    for split in ('validation','test'):
        group = index.loc[index.split == split].reset_index(drop=True).copy()
        elapsed = (group.forecast_date-group.first_snapshot_at.dt.normalize()).dt.days
        predicted = np.maximum(group.product_id.map(means).fillna(fallback)-elapsed, 0)
        probabilities = np.tile(empirical, (len(group),1))
        category = np.minimum(group.remaining_days.to_numpy(dtype=int)-1,h)
        nll = -np.log(empirical[category])
        conditional, support = predict_conditional_cycle(
            conditional_model, group[['product_id','forecast_date','first_snapshot_at']])
        conditional_nll = -np.log(conditional[np.arange(len(group)),category])
        conditional_metrics = distribution_metrics(group,conditional,h,conditional_nll)
        conditional_metrics.update(interval_width_metrics(conditional,h))
        conditional_metrics['support_source_windows'] = {str(k):int(v) for k,v in support.conditional_source.value_counts().items()}
        results[split] = {'point_product_cycle': point_metrics(group,predicted),
                          'distribution_training_frequency': distribution_metrics(group,probabilities,h,nll),
                          'distribution_product_elapsed': conditional_metrics}
        results[split]['distribution_training_frequency'].update(interval_width_metrics(probabilities,h))
        group['baseline_predicted_remaining_days'] = predicted
        group['baseline_coarsened_nll'] = nll
        group = pd.concat([group,support],axis=1)
        for i, name in enumerate([*[f'conditional_p_day_{day}' for day in range(1,h+1)],'conditional_p_tail']):
            group[name] = conditional[:,i]
        group['conditional_coarsened_nll'] = conditional_nll
        for name, values in distribution_summary(conditional,h).items():
            group[f'conditional_{name}'] = values
        frames[split] = group
    results['definition'] = {'point_training_unique_lots': len(lots),
                            'point_global_fallback_cycle_days': fallback,
                            'distribution_training_class_counts': counts.tolist(),
                            'distribution_class_probabilities': empirical.tolist(),
                            'distribution_laplace_alpha': 1,
                            'conditional_cycle': {
                                'version': 'product_elapsed_v1', 'training_unique_lots': len(lots),
                                'cycle_origin': 'first observed calendar date',
                                'conditioning': 'T > elapsed; remaining = T - elapsed',
                                'product_prior_strength': 1, 'global_conditional_laplace_alpha': 1,
                                'fallback': 'No product survivors: global conditional prior. No global survivors: training-window frequency prior.',
                                'uses_stage': False, 'uses_evaluation_labels_for_prediction': False,
                                'selection': 'Fixed rule fitted on training data only; no validation/test tuning.'}}
    return results, frames


def table(headers, rows):
    def cells(tag, values):
        return ''.join(f'<{tag}>{html.escape(str(value))}</{tag}>' for value in values)
    return '<table><thead><tr>'+cells('th',headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+cells('td',row)+'</tr>' for row in rows)+'</tbody></table>'


def point_table(model, baseline):
    rows = []
    for split, label in [('validation','验证'),('test','测试')]:
        for name, result in [('Transformer',model[split]),('训练商品平均周期',baseline[split]['point_product_cycle'])]:
            rows.append([label,name,result['windows'],f"{result['mae_days']:.4f}",f"{result['rmse_days']:.4f}",
                         f"{result['lot_macro_mae_days']:.4f}",f"{result['within_3_days']:.1%}"])
    return table(['集合','模型','窗口数','MAE / 天','RMSE / 天','Lot 等权 MAE / 天','误差 ≤ 3 天'],rows)


def probability_table(model, baseline):
    rows = []
    for split,label in [('validation','验证'),('test','测试')]:
        for name,result in [('Transformer NLL',model[split]),
                            ('同商品周期＋已过天数（主要基线）',baseline[split]['distribution_product_elapsed']),
                            ('训练标签频率（弱基线）',baseline[split]['distribution_training_frequency'])]:
            width=result['central_90_mean_finite_width_days']
            rows.append([label,name,f"{result['coarsened_nll']:.4f}",f"{result['lot_macro_coarsened_nll']:.4f}",
                         f"{result['central_90_class_interval_coverage']:.1%}",
                         f"{result['central_90_finite_upper_fraction']:.1%}",
                         f"{width:.2f}" if width is not None else '—',
                         f"{result['mean_predicted_tail_probability']:.2%}"])
    return table(['集合','模型','分箱 NLL','Lot 等权 NLL','中央 90% 类别区间覆盖率','区间有限上界比例','有限区间平均宽度 / 天','平均尾部概率'],rows)


def figures(run, point, distribution, histories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(16,9), constrained_layout=True)
    axes=axes.ravel()
    for ax,mode,ylabel in [(axes[0],'point','Validation MAE (days)'),(axes[1],'distribution','Validation categorical NLL')]:
        hist=histories[mode]
        ax.plot(hist.epoch,hist.validation_score,marker='.' if len(hist)<3 else None)
        best=int(hist.loc[hist.validation_score.idxmin(),'epoch'])
        ax.axvline(best,ls='--',color='#c56d21',label=f'Selected epoch {best}')
        ax.set(xlabel='Epoch',ylabel=ylabel,title=f'{mode.title()}: selection on validation')
        ax.legend()
    truth=point.remaining_days.to_numpy(); pred=point.predicted_remaining_days.to_numpy()
    axes[2].scatter(truth,pred,s=7,alpha=.25)
    high=max(truth.max(),pred.max())
    axes[2].plot([0,high],[0,high],'--',color='grey')
    axes[2].set(xlabel='Label (integer days)',ylabel='Prediction (days)',title='Point model: held-out test')
    errors=np.sort(np.abs(pred-truth))
    axes[3].plot(errors,np.arange(1,len(errors)+1)/len(errors))
    axes[3].set(xlabel='Absolute error (days)',ylabel='Cumulative fraction',title='Test error distribution',ylim=(0,1))
    h=len([col for col in distribution if col.startswith('p_day_')])
    for day in (1,3,7):
        if day>h: continue
        p=distribution[[f'p_day_{i}' for i in range(1,day+1)]].sum(axis=1).to_numpy()
        actual=(distribution.remaining_days.to_numpy()<=day).astype(float)
        bins=np.minimum((p*10).astype(int),9)
        xs,ys=[],[]
        for b in range(10):
            mask=bins==b
            if mask.any(): xs.append(p[mask].mean()); ys.append(actual[mask].mean())
        axes[4].plot(xs,ys,'o-',label=f'P(R <= {day} days)')
    axes[4].plot([0,1],[0,1],'--',color='grey')
    axes[4].set(xlabel='Predicted probability',ylabel='Observed fraction',title='Test CDF reliability',xlim=(0,1),ylim=(0,1))
    axes[4].legend()
    axes[5].hist(distribution.p_tail,bins=20)
    axes[5].set(xlabel=f'P(R > {h} days)',ylabel='Test windows',title='Predicted tail mass; label support is limited')
    fig.savefig(run/'experiment_figures.png',dpi=160)
    plt.close(fig)


def render_report(run, summary, point_frames, distribution_frames, histories):
    figures(run,point_frames['test'],distribution_frames['test'],histories)
    image=base64.b64encode((run/'experiment_figures.png').read_bytes()).decode('ascii')
    data=summary['data_manifest']; point=summary['point']; dist=summary['distribution']; baseline=summary['baselines']
    protocol=data['protocol']; h=dist['target_spec']['horizon']
    horizon_origin = ('按实验设计事先固定，未根据验证或测试标签选择'
                      if dist['target_spec'].get('horizon_source') == 'fixed_experiment_design'
                      else f"由训练标签的 {100 * dist['target_spec']['horizon_quantile']:g}% 分位向上取整得到")
    title='DK 每日预测全量实验报告' if not data['smoke_only'] else 'DK 每日预测报告流程验证（非全量结果）'
    sample_rows=[]
    for key,label in [('train','训练'),('validation','验证'),('test','测试')]:
        item=data['used_sample_counts'][key]
        sample_rows.append([label,item['windows'],item['lots'],item['first_forecast_date'][:10],
                            item['last_forecast_date'][:10],f"{item['label_min']}–{item['label_max']}"])
    model_rows=[]
    for name,result in [('点预测',point),('NLL 分布预测',dist)]:
        state=result['state']
        model_rows.append([name,state['epochs_completed'],state['best_epoch'],
                           '早停' if state['early_stopped'] else '达到上限',f"{result['execution_minutes']:.1f}",
                           state['monitor'],result['parameters']])
    calibration=[]
    for name, metrics in [('Transformer NLL',dist['test']),
                          ('同商品周期＋已过天数',baseline['test']['distribution_product_elapsed']),
                          ('训练标签频率',baseline['test']['distribution_training_frequency'])]:
        for day,item in metrics['calibration_by_horizon_days'].items():
            calibration.append([name,f'剩余 ≤ {day} 天',f"{item['brier_score']:.4f}",f"{item['ece_10_bins']:.4f}"])
    horizon_rows=[]
    for group,metrics in point['test']['by_remaining_days'].items():
        horizon_rows.append([group,metrics['windows'],f"{metrics['mae_days']:.4f}",f"{metrics['rmse_days']:.4f}"])
    p_gain=1-point['test']['mae_days']/baseline['test']['point_product_cycle']['mae_days']
    n_gain=baseline['test']['distribution_product_elapsed']['coarsened_nll']-dist['test']['coarsened_nll']
    p_text=f'降低 {p_gain:.1%}' if p_gain>=0 else f'增加 {-p_gain:.1%}'
    n_text=f'降低 {n_gain:.4f}' if n_gain>=0 else f'增加 {-n_gain:.4f}'
    md=summary['run_manifest']
    sections=[f'<h1>{title}</h1><p class="muted">实验 {html.escape(run.name)} · 基于缩放修正版 114e023 的每日预测工作区</p>',
      '<div class="callout"><b>两种模式均已完成训练及最佳模型重载核验。</b>'
      f" 点预测测试 MAE 为 {point['test']['mae_days']:.4f} 天，相比训练商品周期基线{p_text}；"
      f"分布预测测试分箱 NLL 为 {dist['test']['coarsened_nll']:.4f}，相比同商品周期＋已过天数的条件分布基线{n_text}。"
      '两种损失衡量不同目标，不能通过直接比较 MAE 和 NLL 数值判断谁更好。</div>',
      '<h2>1. 实验设置与复现</h2><p>按用户授权先训练点预测，再训练 NLL 模式。两者独立初始化，使用相同每日样本、'
      '特征处理和时间划分；测试预测在两种训练都结束后生成，不用于早停、选模或调参。本次仅一个随机种子。</p>',
      table(['项目','设置'],[['开始时间',md.get('started_at','流程验证')],['模型','Transformer，2 层、4 heads、d_model=64、LayerNorm、last pooling'],
            ['训练配置',f"seed={protocol['seed']}，batch={protocol['batch_size']}，最多 {protocol['epochs']} 轮，patience={protocol['patience']}"],
            ['运行环境',f"Python {point['state']['python']} / Torch {point['state']['torch']} / CPU {protocol['threads']} 线程"],
            ['数据校验',data['data']['csv_sha256']],['源码/模型/预处理器','哈希与运行记录一致；工作区源码快照单独留存']]),
      table(['模式','实际轮数','最佳轮数','结束条件','约运行分钟','选模指标','参数量'],model_rows),
      '<p>各模式运行时间按配置保存至完成状态文件写入的时间差计算，包含最后一次最佳模型重载与验证，不含前置样本准备。</p>',
      '<h2>2. 样本、标签与时间划分</h2>',table(['集合','窗口数','Lot 数','首预测日','末预测日','标签范围 / 天'],sample_rows),
      '<p>每个样本是 root Lot＋规划日期。每日夜间使用当天结束前所有观测，保留日内 stage 和原始观测时间；'
      '历史窗口为 15 个日历日，无新观测时保留最后状态并加入观测间隔标记。完工日期由企业原快照规则推定；'
      '标签为完工日期减规划日期的整数，已完成批次不生成在制样本。特征不含完工标签。</p>',
      f"<p>训练截止 {protocol['train_end']}，验证截止 {protocol['validation_end']}，之后为测试预测日。"
      f"测试预测日期上界（不含）为 {protocol.get('forecast_end_exclusive') or '源数据最后一个不完整日期'}。"
      f"训练和验证标签需在各自截止前经过 {protocol['completion_margin_days']} 天观察缓冲；"
      '测试标签允许使用源数据截止前的后续观测确认，后续观测不进入此前预测输入。'
      '特征编码与缩放只拟合训练截止前观测；目标缩放和报告基线只拟合训练标签。</p>',
      '<h2>3. 点预测结果</h2>',point_table(point,baseline),
      f"<p>点预测采用 L1 损失；对外输出截断为非负，误差计算不取整。测试原始负输出有 {point['test']['negative_raw_predictions']} 个。"
      f"测试中位绝对误差 {point['test']['median_ae_days']:.4f} 天，P90 {point['test']['p90_ae_days']:.4f} 天，"
      f"偏差 {point['test']['bias_days']:.4f} 天（负值表示偏早）。主指标对窗口等权，另列 Lot 内先平均再对 Lot 等权的指标。</p>",
      '<p>周期基线先对训练中不同 Lot 的总日历周期按商品求均值，再减去预测日已观测天数并截断至零；'
      '未知商品使用训练 Lot 总体平均周期。它不读取验证或测试标签来估计周期。</p>',
      table(['测试标签范围 / 天','窗口数','MAE / 天','RMSE / 天'],horizon_rows),
      '<h2>4. 分布预测结果</h2>',probability_table(dist,baseline),
      table(['测试标签范围 / 天','窗口数','分箱 NLL'],
            [[group,item['windows'],f"{item['coarsened_nll']:.4f}"]
             for group,item in dist['test']['by_remaining_days'].items()]),
      f"<p>H={h}，{horizon_origin}；模型输出第 1～{h} 天与 &gt;{h} 尾部，共 {h+1} 类概率。"
      f"训练尾部样本 {dist['target_spec']['train_tail_count']:,} 个。NLL 为真实类别的负对数概率，使用自然对数。"
      '超过 H 的原始标签仍保留，只在计算分类损失时映射到独立尾部类别；不是截为第 H 天。</p>',
      '<p><b>主要基线：同商品历史周期＋已经过天数。</b>训练中每个 Lot 只计一次，以首次观测日期至推定完工日期的整数差作为周期 T。'
      '预测时计算已经过天数 e，只保留 T&gt;e 的历史批次，将 T−e 映射为逐日或尾部类别。输入仅为商品、首次观测日与预测日，不读取待预测样本的完工或剩余时间标签。</p>',
      '<p>为避免零概率 NLL，全局条件类别计数采用 Laplace(1) 平滑，得到 q(e)；商品条件计数为 n(r)，支持样本数为 n，'
      '最终 p(r)=[n(r)+q(r|e)]/(n+1)，即固定 1 个伪样本向全局条件分布平滑。无同商品存活样本时退回 q(e)，'
      '全局也没有 T&gt;e 样本时退回训练窗口频率分布。两个平滑常数均固定为 1，未据验证或测试结果搜索或调整。</p>',
      table(['集合','有同商品支持的窗口','回退全局条件的窗口','回退固定频率的窗口'],
            [[label,*[baseline[split]['distribution_product_elapsed']['support_source_windows'].get(key,0)
                       for key in ('product_conditional_smoothed','global_conditional_fallback','global_frequency_fallback')]]
             for split,label in [('validation','验证'),('test','测试')]]),
      '<p>固定频率保留为弱基线：对训练窗口的日/尾部类别计数，采用 Laplace(1)，所有样本使用同一分布。'
      '条件基线采用上述固定平滑和回退规则，只拟合本次训练集，不根据验证或测试结果调整。它仍共享已完成可标注样本的选择限制，尚未使用 stage。</p>',
      '<p>覆盖率针对离散类别的中央 90% 区间；尾部包含开放上界，因此同时报告有限上界比例和有限区间平均宽度（上界减下界）。'
      '有限区间宽度只对各方法自己的有限区间求平均；不能单凭高覆盖率认定校准更好。'
      f"模型的标称 90% 区间实际测试覆盖率为 {dist['test']['central_90_class_interval_coverage']:.1%}，概率校准仍需改进。</p>",
      table(['模型','测试概率事件','Brier score','10 桶 ECE'],calibration),
      '<p>分位数落入尾部时只有下界，没有有限上界；导出的 mean_days_lower_bound 也只是期望天数下界，'
      '不能作为精确平均完工时间。此模式没有估计尾部内部逐日分布。</p>',
      f'<img alt="Training curves, point test errors and distribution calibration" src="data:image/png;base64,{image}">',
      '<h2>5. 解释边界</h2><ul>'
      '<li>当前结果是完成批次上的时间留出评价。完工标签来自企业快照启发式，未等同于独立核验的真实完工/到货事件。</li>'
      '<li>保留用户认可的全历史质量筛选；未完成、边界批次和截止时标签未确认的样本被排除，尚未使用删失损失。'
      '这会偏向短周期，不能直接代表全部实时在制 Lot。</li>'
      f"<li>本次测试标签最大值为 {int(point_frames['test'].remaining_days.max())} 天，超过 H 的占比为 {dist['test']['actual_tail_fraction']:.2%}。"
      '若缺少尾部实际事件，就无法充分检验长尾预测及尾部概率校准。</li>'
      '<li>同一 Lot 可有多个日末窗口，窗口间并非独立；因此同时报告 Lot 等权指标。本次单次训练未估计跨随机种子的波动。</li>'
      '<li>原缩放修正版的 2.0424 天属于另一套窗口和标签口径，不与本报告数值直接比较。历史每日实验若采用不同时间切分，MAE／NLL 的变化也不能解释为模型改进比例；以本次相同测试窗口上的训练基线为参照。下游优化不在本次实验范围内。</li></ul>',
      '<h2>6. 产物与核验</h2><p>两模式的逐测试窗口 ID、Lot、规划日期、输入截止与原始标签逐项一致；'
      '已重新计算导出指标，并与选模日志及保存指标核对。输入 CSV 未修改，训练源码未在运行中改动；未 commit 或 push。</p>',
      table(['产物','路径'],[['综合指标',str(run/'evaluation_summary.json')],['点预测模型',str(run/'training/point')],
            ['NLL 模型',str(run/'training/distribution')],['逐窗口点预测',str(run/'evaluation/point/test_predictions.csv')],
            ['逐窗口分布预测',str(run/'evaluation/distribution/test_predictions.csv')],
            ['逐窗口基线概率与回退记录',str(run/'test_baseline_predictions.csv')],
            ['各 Lot 最晚测试窗口（合并两模式）',str(run/'test_latest_per_lot.csv')],
            ['点预测每 Lot 误差',str(run/'point_per_lot_metrics.csv')],['本次训练输入索引',str(run/'training/sample_index.csv')]])]
    if data['smoke_only']:
        sections.insert(1,'<div class="warning">此文件仅验证报告生成流程，使用 1 轮小样本运行，不是全量实验结果。</div>')
    if (run/'source_app_order_policy.yaml').exists():
        sections.insert(3, '<p>已单独冻结 APP 订单假设：提前28天知道材料与固定数量，以首次快照日期作为计划进入生产的代理，'
                        '以训练批次周期构造承诺日期。该假设没有加入本次 Transformer 输入；本报告不代表已重跑新版 APP。</p>')
    css="""body{font-family:'Microsoft YaHei','Segoe UI',sans-serif;max-width:1180px;margin:36px auto;padding:0 24px;color:#173042;line-height:1.8}h1{font-size:29px}h2{font-size:21px;margin-top:32px}table{width:100%;border-collapse:collapse;margin:16px 0;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #dde5ea;overflow-wrap:anywhere}th{background:#eef4f7}.callout{background:#edf6fa;border-left:5px solid #23809e;padding:18px}.warning{background:#fff2d9;padding:18px}.muted{color:#5a7280}img{width:100%}@media print{body{margin:0;max-width:none}table,img,.callout{break-inside:avoid}}"""
    (run/'experiment_report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+css+'</style><body>'+''.join(sections)+'</body></html>',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--allow-smoke',action='store_true',help='Explicitly label a report-generator validation fixture.')
    args=parser.parse_args()
    run=project_path(args.run_dir)
    manifest=read_json(run/'run_manifest.json')
    data=read_json(run/'training/data_manifest.json')
    if data['smoke_only'] and not args.allow_smoke:
        raise ValueError('A smoke run cannot be presented as a full experiment.')
    if not data['smoke_only']:
        for step in ('training','evaluate_point','evaluate_distribution'):
            if manifest['steps'][step]['exit_code'] != 0:
                raise ValueError(f'{step} did not finish successfully')
    if file_hash(run/'training/sample_index.csv') != data['sample_index_sha256']:
        raise ValueError('Sample index changed after training')
    current=code_hashes()
    if any(current.get(path)!=digest for path,digest in data['source_hashes'].items()):
        raise ValueError('Training source changed. Preserve the historical report and use its source snapshot.')
    if file_hash(dataset_paths('dk')['csv']) != data['data']['csv_sha256']:
        raise ValueError('Input CSV changed since training')
    index=read_frame(run/'training/sample_index.csv')
    point,pf,ph=validate_mode(run,'point',data,index)
    dist,df,dh=validate_mode(run,'distribution',data,index)
    if point['target_spec'] != dist['target_spec']:
        raise ValueError('Training target specs differ between modes')
    baseline,bf=baselines(index,dist['target_spec']['horizon'])
    for split,frame in bf.items():
        frame.to_csv(run/f'{split}_baseline_predictions.csv',index=False,encoding='utf-8-sig')
    point_test=pf['test'].copy()
    point_test['absolute_error_days']=np.abs(point_test.predicted_remaining_days-point_test.remaining_days)
    per_lot=point_test.groupby('lot_id').agg(windows=('sample_id','size'),mae_days=('absolute_error_days','mean'),
              bias_days=('error_days','mean'),first_forecast_date=('forecast_date','min'),last_forecast_date=('forecast_date','max'))
    per_lot.to_csv(run/'point_per_lot_metrics.csv',encoding='utf-8-sig')
    point_test.nlargest(50,'absolute_error_days').to_csv(run/'largest_point_errors.csv',index=False,encoding='utf-8-sig')
    dist_columns=['sample_id',*[c for c in df['test'] if c.startswith(('p_day_','q05_','q95_','median_'))],
                  'p_tail','mean_days_lower_bound','coarsened_nll']
    joined=point_test.merge(df['test'][dist_columns],on='sample_id',validate='one_to_one')
    baseline_columns=['sample_id',*[c for c in bf['test'] if c.startswith('conditional_')]]
    joined=joined.merge(bf['test'][baseline_columns],on='sample_id',validate='one_to_one')
    joined.sort_values('forecast_date').groupby('lot_id',sort=False).tail(1).to_csv(run/'test_latest_per_lot.csv',index=False,encoding='utf-8-sig')
    summary={'run_id':run.name,'generated_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
             'run_manifest':manifest,'data_manifest':data,'point':point,'distribution':dist,'baselines':baseline,
             'verification':{'same_samples_between_modes':True,'validation_selection_matches_checkpoint':True,
                             'exported_metrics_recomputed':True,'training_source_unchanged':True,'csv_unchanged':True,
                             'test_not_used_for_model_selection':True,
                             'conditional_baseline_training_only':True},
             'report_generator_sha256':file_hash(Path(__file__))}
    if (run/'preflight.json').exists():
        preflight=read_json(run/'preflight.json')
        for path,digest in preflight['frozen_input_hashes'].items():
            if file_hash(project_path(path)) != digest:
                raise ValueError(f'Frozen input or old artifact changed: {path}')
        if data['sample_index_sha256'] != preflight['expected_sample_index_sha256']:
            raise ValueError('Full training sample index differs from preflight.')
        summary['verification']['preflight_and_old_artifact_hashes_unchanged']=True
        summary['preflight']=preflight
    write_json(run/'evaluation_summary.json',summary)
    render_report(run,summary,pf,df,{'point':ph,'distribution':dh})
    if Path(__file__).resolve() != (run/'report_generator.py').resolve():
        shutil.copyfile(__file__,run/'report_generator.py')
    print(json.dumps({'report':str(run/'experiment_report.html'),
                     'point_test_mae_days':point['test']['mae_days'],
                     'distribution_test_coarsened_nll':dist['test']['coarsened_nll'],
                     'smoke_only':data['smoke_only']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'): stream.reconfigure(encoding='utf-8')
    main()
