"""Four-way comparison of saved results; no model/policy tuning here."""
from pathlib import Path
from html import escape
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

NAMES={'nll_expected_supply':'NLL 期望到货 PTO','point_transformer':'点预测 PTO',
       'historical_product_cycle':'历史商品周期基线','oracle':'全 T 完美信息 Oracle'}


def table(frame):return frame.to_html(index=False,border=0,float_format=lambda x:f'{x:,.4f}')


def render(out,s):
    out=Path(out);oracle=s['oracle'];metrics=s['distribution_metrics']
    comparison=pd.DataFrame(oracle['comparison'])
    comparison['method']=comparison.method.map(NAMES)
    comparison['regret_percent']=100*comparison.relative_regret_over_oracle
    comparison=comparison[['method','policy_cost','absolute_regret','regret_percent']]
    comparison=pd.concat([comparison,pd.DataFrame([{'method':NAMES['oracle'],'policy_cost':oracle['costs']['total_cost'],
                         'absolute_regret':0.,'regret_percent':0.}])],ignore_index=True)
    costs=pd.DataFrame([{'method':NAMES[m['method']],**m['costs']} for m in s['methods']]+
                      [{'method':NAMES['oracle'],**oracle['costs']}])
    totals={m['method']:m['costs']['total_cost'] for m in s['methods']}
    difference=totals['nll_expected_supply']-totals['point_transformer']
    if abs(difference)<1e-5:conclusion='本条轨迹上，NLL 期望到货与点预测 PTO 成本相同。'
    else:conclusion=f'NLL 期望到货比点预测 PTO 成本{"降低" if difference<0 else "增加"} {abs(difference):,.2f}（{abs(difference)/totals["point_transformer"]*100:.2f}%）。'
    frames={m['method']:pd.read_csv(out/m['method']/'period_costs.csv') for m in s['methods']}
    frames['oracle']=pd.read_csv(out/'oracle/oracle_period_costs.csv')
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    styles={'point_transformer':('#397bbb','--','o'),'historical_product_cycle':('#d78835',':','s'),
            'nll_expected_supply':('#00856e','-','D'),'oracle':('#8c6bab','-.','x')}
    for method,frame in frames.items():
        frame['cumulative_realized_cost']=frame.total_cost.cumsum()
        color,ls,marker=styles[method]
        for ax,col in zip(axes.flat,['cumulative_realized_cost','customer_backlog_end','secondary_ordered_die','production_total']):
            ax.plot(frame.period,frame[col],label=NAMES[method],color=color,linestyle=ls,marker=marker,markersize=4)
    for ax,title in zip(axes.flat,['累计实际成本','每期期末客户积压','当期第二货源下单','当期生产']):
        ax.set(title=title,xlabel='执行期',xticks=range(1,8));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(out/'experiment_figures.png',dpi=150);plt.close(fig)
    calibration=[]
    for day in (7,14,21,28):
        x=metrics['calibration_by_horizon_days'][str(day)]
        calibration.append({'horizon_days':day,'brier_score':x['brier_score'],'ece_10_bins':x['ece_10_bins']})
    forecasts=pd.read_csv(out/'distribution_forecasts.csv')
    observed=forecasts.has_snapshot
    quantity_rows=[]
    for period,g in forecasts.groupby('period'):
        quantity_rows.append({'period':period,'orders':len(g),'observed':int(g.has_snapshot.sum()),
            'known_quantity_die':g.fixed_quantity.sum(),'expected_within_28_days':g.expected_quantity_within_horizon.sum(),
            'expected_beyond_28_days':g.expected_quantity_beyond_horizon.sum()})
    nll=s['methods'][-1]
    weekly=pd.concat([f.assign(method=NAMES[m]) for m,f in frames.items()],ignore_index=True)
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NLL 期望到货：滚动 APP 四方比较</title><style>
body{{background:#f3f6f8;color:#18383e;font:15px/1.8 "Microsoft YaHei",sans-serif;margin:0}}main{{max-width:1250px;margin:28px auto;padding:0 24px}}
header,section{{background:white;padding:24px;margin-bottom:18px;border:1px solid #dce6e9;border-radius:10px}}header{{border-top:6px solid #00856e}}
h1{{font-size:28px}}h2{{font-size:21px}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:8px;border-bottom:1px solid #dfe8eb;text-align:right;white-space:nowrap}}th{{background:#edf4f4}}
.scroll{{overflow-x:auto}}img{{width:100%}}.note{{padding:15px;background:#fff4dd;border-left:4px solid #c68c34}}a{{color:#087d71}}code{{overflow-wrap:anywhere}}
</style><main><header><h1>NLL 期望到货：滚动 APP 四方比较</h1>
<p>3 材料、3 产品；2025-01-07 至 02-24；7 天一期，N=4、T=7，最后一次规划至 03-17。复用已完成训练的 NLL 最佳第 {s['distribution_best_epoch']} 轮，未重新训练。</p>
<p class="note">{conclusion}成本为合成实验单位。本次方案是确定性 APP 上的期望到货 PTO，不是两阶段或多阶段随机规划。</p></header>
<section><h2>相同测试轨迹上的成本</h2><div class="scroll">{table(comparison)}</div>
<p>点预测和历史基线复用昨日已完成且哈希核验通过的轨迹，需求、数量、BOM、产能、成本、初始状态与本次完全相同。Oracle 重新核验，其值与原值一致；新策略代入 Oracle 约束可行且成本一致。Regret 同时包含信息和滚动视野、末端近似的影响。</p>
<div class="scroll">{table(costs)}</div><img src="experiment_figures.png" alt="四种方法的累计成本、积压、采购和生产"></section>
<section><h2>如何使用分布</h2>
<p>每个 Lot 的剩余完成天数概率按 1–7、8–14、15–21、22–28 天求和。材料 m 在第 k 周的预测到货量为 Σ Q_l × P_l(第 k 周到货)。超过 28 天的概率与对应期望数量留在规划窗外，不归一化回前四周，也不当作第 29 天的确定到货。</p>
<p>这是供给数量的期望，不是假定实际拆批。期望可分布于多个周，但累计不超过固定数量；回放仍在真实完工周整批收一次。当期生产只用期初库存，预测本周到货只能用于下周生产。每次执行一期后，按实际收货结转状态并重新预测。</p>
<p>尚无快照的已公告订单使用与两基线完全相同的合成承诺日期；没有给这部分订单假造 NLL 分布。共 {len(forecasts):,} 条订单视图，其中 {int(observed.sum()):,} 条有快照、{int((~observed).sum()):,} 条仅有承诺。</p>
<div class="scroll">{table(pd.DataFrame(quantity_rows))}</div>
<p>期望到货不会保留整批到货不确定性的全部决策影响；在规划中可能出现实际不会发生的分散到货。因此这次评估回答的是“分布的期望供给是否改善该确定性 APP”，不能据此判断场景随机规划的效果。尾部校准问题保留，未使用测试结果调整模型或参数。</p></section>
<section><h2>周决策时点的 NLL 预测检查</h2>
<p>仅对有快照且标签可核实的 {metrics['windows']} 条预测评分：分箱 NLL <b>{metrics['coarsened_nll']:.4f}</b>，Lot 等权 NLL {metrics['lot_macro_coarsened_nll']:.4f}；标称 90% 类别区间覆盖率 {100*metrics['central_90_class_interval_coverage']:.2f}%，上界开放比例 {100*(1-metrics['central_90_finite_upper_fraction']):.2f}%。未知标签 {metrics['unknown_rows']} 条。</p>
<p>实际 >28 天比例 {100*metrics['actual_tail_fraction']:.3f}%，平均尾部预测概率 {100*metrics['mean_predicted_tail_probability']:.3f}%。真实标签只在滚动执行结束后用于评分，不进入规划输入。不将共用确定承诺的订单强行计算概率 NLL，也不以尾部期望下界充当精确均值。</p>
<div class="scroll">{table(pd.DataFrame(calibration))}</div><p>这里是三种材料在周决策时点的预测子集，与此前全部 DK 每日测试窗口不同，不能直接比较指标改善。</p></section>
<section><h2>执行与核验</h2><ul>
<li>NLL 新增 7 次 CPLEX 求解均最优；最大规划约束误差 {nll['max_planning_constraint_residual']:.3g}，当期执行约束违反 {nll['max_execution_violation']:.3g}。</li>
<li>独立从 CPLEX XML 重算三策略的计划、库存、积压、采购管道和费用，最大差异 {s['post_run_audit']['max_audit_numeric_difference']:.3g}（容差 2e-5）。实际到货一批一次、数量守恒。</li>
<li>原始 CSV、预测源码与模型、冻结 APP 数据及旧结果均未改变；当前 APP 源码与配置另存快照。NLL 使用同一订单集合、数量、可见历史和承诺规则。</li>
<li>第 7 期下单、第 8 期收货的采购费不进入 T 内实际成本，但仍计入当次规划目标。末期订单保留，未伪造到货。原材料持有成本与残值均为零。</li></ul>
<div class="scroll">{table(weekly[['method','period','production_total','customer_backlog_end','secondary_ordered_die','total_cost']])}</div></section>
<section><h2>文件与解释范围</h2><p>单条轨迹、单一需求种子，探索性材料选择，企业快照推定完工，全历史质量筛选，固定首次 dieQty 与合成提前订单。没有据测试成本调参；结果不代表其他规模和供需条件下必然排序。</p>
<p><a href="experiment_summary.json">汇总结果</a> · <a href="distribution_forecasts.csv">无标签分布预测</a> · <a href="distribution_evaluation_only.csv">事后预测评分</a> · <a href="nll_expected_supply/primary_forecasts.csv">期望供给面板</a> · <a href="nll_expected_supply/period_costs.csv">NLL 逐期成本</a> · <a href="post_run_audit.json">独立核验</a> · <a href="oracle/oracle_summary.json">Oracle</a></p>
<p>模型：{escape(s['distribution_checkpoint'])}<br>数据：{escape(s['dataset'])}<br>仅保存本地；未提交 Git。</p></section></main></html>'''
    (out/'experiment_report.html').write_text(text,encoding='utf-8')
