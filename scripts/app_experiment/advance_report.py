"""Report the advance-order pilot from saved results, without tuning or solving."""
from pathlib import Path
from html import escape
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

NAMES={'point_transformer':'快照点预测 PTO','historical_product_cycle':'历史商品周期基线'}


def table(frame):
    return frame.to_html(index=False,border=0,float_format=lambda x:f'{x:,.4f}')


def render(out,s):
    out=Path(out)
    data=Path(s['dataset'])
    params=json.loads((data/'shared/parameters.json').read_text(encoding='utf-8'))
    checks=json.loads((data/'advance_order_checks.json').read_text(encoding='utf-8'))
    oracle=s['oracle']
    methods=s['methods']
    audit=s['post_run_audit']
    rows=pd.DataFrame(oracle['comparison'])
    rows['method']=rows.method.map(NAMES)
    rows['relative_regret_percent']=100*rows.pop('relative_regret_over_oracle')
    baseline=next(m for m in methods if m['method']=='historical_product_cycle')['costs']['total_cost']
    point=next(m for m in methods if m['method']=='point_transformer')['costs']['total_cost']
    pct=100*(baseline-point)/baseline
    conclusion=f'本条轨迹中，快照 PTO 成本相对统计基线{"降低" if pct>=0 else "增加"} {abs(pct):.2f}%。'
    if abs(point-baseline)<1e-5:
        conclusion='本条轨迹中，两种方法实际成本相同。'
    costs=pd.DataFrame([{'method':NAMES[m['method']],**m['costs']} for m in methods]+
                       [{'method':'全 T 完美信息 Oracle',**oracle['costs']}])
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for m in methods:
        f=pd.read_csv(out/m['method']/'period_costs.csv')
        for ax,col in zip(axes.flat,['cumulative_realized_cost','customer_backlog_end','secondary_ordered_die','production_total']):
            ax.plot(f.period,f[col],label=NAMES[m['method']],marker='o',linestyle='-' if m['method']=='point_transformer' else '--')
    f=pd.read_csv(out/'oracle/oracle_period_costs.csv')
    f['cumulative_realized_cost']=f.total_cost.cumsum()
    for ax,col,title in zip(axes.flat,['cumulative_realized_cost','customer_backlog_end','secondary_ordered_die','production_total'],
                           ['累计实际成本','期末客户积压','当期第二货源下单','当期生产']):
        ax.plot(f.period,f[col],label='全 T Oracle',color='#7a5fa5',linestyle=':',marker='x')
        ax.set(title=title,xlabel='执行期',xticks=f.period)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9)
    fig.savefig(out/'experiment_figures.png',dpi=160)
    plt.close(fig)
    metrics=pd.DataFrame(s['forecast_metrics'])
    metrics['method']=metrics.method.map(NAMES)
    weekly=pd.concat([pd.read_csv(out/m['method']/'period_costs.csv').assign(method=NAMES[m['method']]) for m in methods])
    diagnostics=pd.read_csv(data/'test/diagnostics/material_balance.csv')
    unknown=sum(x['not_known_at_week_start_die'] for x in audit['primary_visibility'])
    peak_usage={}
    for m in methods:
        f=pd.read_csv(out/m['method']/'period_costs.csv')
        peak_usage[NAMES[m['method']]]=float(f.production_total.max()/params['capacity_product_units_per_period'])
    all_receipts=sum(x['receipt_die'] for x in audit['primary_visibility'])
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>提前28天订单：新版点PTO滚动APP</title><style>
body{{background:#f4f7f8;color:#183640;font:15px/1.75 "Microsoft YaHei",sans-serif;margin:0}}main{{max-width:1280px;margin:28px auto;padding:0 24px 32px}}
section,header{{background:#fff;padding:24px;border:1px solid #dce5e7;border-radius:10px;margin-bottom:18px}}header{{border-top:6px solid #126d83}}
h1{{font-size:29px}}h2{{font-size:21px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #dae3e5;padding:8px;text-align:right;white-space:nowrap}}
th{{background:#eef4f5}}.scroll{{overflow-x:auto}}img{{width:100%}}.note{{background:#fff4e1;padding:16px;border-left:4px solid #d39a41}}a{{color:#126d83}}code{{overflow-wrap:anywhere}}</style>
<main><header><h1>提前 28 天订单：3×3 点 PTO 滚动 APP</h1>
<p>测试 2025-01-07 至 2025-02-24；每 7 天决策，规划 N=4 期，只执行当前一期，评价 T=7 期。最后一次规划覆盖至 2025-03-17。</p>
<p>复用 10 周训练版点预测最佳第 {s['point_best_epoch']} 轮，未重新训练。CPLEX 完成两种策略各 7 次滚动求解及一次全 T Oracle 求解。</p>
<p class="note">{conclusion}成本为合成实验单位，不能解释为企业真实节省金额。</p></header>
<section><h2>实际成本与 regret</h2><div class="scroll">{table(rows)}</div>
<p>Regret = 策略的 7 期实际成本 − 同一需求、到货、库存、产能和成本下的全 7 期完美信息最优成本。它包含信息差异、滚动视野和末端近似的影响，不能全归因于预测误差。</p>
<div class="scroll">{table(costs)}</div><img src="experiment_figures.png" alt="累计成本、积压、第二货源与生产"></section>
<section><h2>信息规则与训练期校准</h2>
<p>首次快照日期作为计划进入生产日期代理，订单公告提前 28 天，数量固定首次 dieQty。公告后尚无快照时，两种方法共用训练商品周期构造的承诺日期；出现快照后分别使用 Transformer 点预测或历史商品平均周期减已过天数。承诺逾期但尚未收到的订单保留，预测到货最早放在当前期，不追补期初库存。订单信息假设不是企业真实订单流。</p>
<p>只有已公告且尚未收到的订单进入计划。未来真实完工记录与预测分开，执行器在固定当期生产和采购后才读取当期到货。日粒度预测按周聚合，一个 Lot 只贡献一次整批预测数量；下周重新预测不代表重复实际到货。</p>
<div class="scroll">{table(pd.DataFrame(checks['periods']))}</div>
<p>训练截止 2024-12-10，验证截止 2025-01-07。下游尺度使用 10 月 8 日至 12 月 2 日的 8 个完整参考周，标签在训练截止前确认；边界不完整周不作供给参考。需求为训练尺度的独立均匀扰动（0.8–1.2），规划窗内已知，seed=42。当前只评价测试，训练和验证 APP 未执行。</p>
<p>保持原设定：BOM、成本、产能目标利用率 85%、第二货源能力为平均材料消耗的 30%、有效提前期 1 期、初始原料为一周期平均消耗，原料持有与残值为 0。重新校准每期产能为 {params['capacity_product_units_per_period']:,.2f} 产品单位。未根据本次测试结果调整参数。</p></section>
<section><h2>预测与决策敏感性</h2><div class="scroll">{table(metrics)}</div>
<p>all_orders 包含共用承诺的未观测订单；snapshot_observed 才是有快照后的模型比较。真实标签仅在执行后合并，包含数据截止前确认、但完成时间可能晚于 T 的订单；未知结局单独计数。不能与全部每日窗口指标直接比较。</p>
<p>{audit['different_week_predictions']} 条预测的到货周不同，当期执行动作最大差额 {audit['max_executed_action_difference']:,.2f}，未来计划最大差额 {audit['max_future_plan_difference']:,.2f}。真实到货共 {all_receipts:,.0f} die，其中 {unknown:,.0f} die 在相应周决策时尚未公告。</p>
<p>各方法最高产能利用率：{escape(json.dumps(peak_usage,ensure_ascii=False))}。</p>
<div class="scroll">{table(pd.DataFrame(audit['procurement_saturation']))}</div>
<div class="scroll">{table(diagnostics)}</div></section>
<section><h2>时间、数量与求解核验</h2><ul>
<li>当期生产仅使用期初已知库存；当期第一货源到货下期可用，第二货源订单按正提前期进入期初库存。库存、积压、订单管道持续结转。</li>
<li>所有 CPLEX 解均最优，独立重算计划约束、执行状态和成本；最大审计差异 {audit['max_audit_numeric_difference']:.3g}，容差 2e-5。</li>
<li>两种滚动策略的执行轨迹代入全 T Oracle 均可行，目标值一致；Oracle 自身独立顺序回放通过。</li>
<li>实际收货不重复、原料与成品及采购管道数量守恒；原 CSV、预测模型、冻结新数据及历史结果哈希均未改变。</li>
<li>第 7 期采购到第 8 期收到，采购费不进入本窗口实际成本，但进入滚动规划目标。Oracle 取末期采购零作为 T 内等价代表解。第一货源未到订单保留，不伪造收货。</li></ul>
<div class="scroll">{table(weekly[['method','period','production_total','finished_inventory_end','customer_backlog_end','secondary_ordered_die','total_cost']])}</div></section>
<section><h2>解释范围与文件</h2><p>单条测试轨迹、单一需求种子；材料经过探索性筛选，完工沿用企业快照推定、全历史质量筛选、固定首次 dieQty、整批到货与无运输假设。预测的长尾和校准问题保留。当前运行的是点 PTO，不是 NLL 随机规划或决策导向训练。</p>
<p>新版时间划分、订单可见规则和训练供给尺度均变化，不能把与旧实验的成本差异解释为单一改动的效果。当前比较只在共享本次数据的两策略和 Oracle 之间进行。</p>
<p><a href="experiment_summary.json">完整结果</a> · <a href="weekly_predictions.csv">无标签周预测</a> · <a href="forecast_evaluation_only.csv">事后预测评价</a> ·
<a href="post_run_audit.json">独立核验</a> · <a href="oracle/oracle_summary.json">Oracle 与 regret</a> · <a href="point_transformer/period_costs.csv">PTO 逐期成本</a></p>
<p>数据：{escape(str(data))}<br>模型：{escape(s['point_checkpoint'])}<br>仅保存本地，未提交 Git。</p></section></main></html>'''
    (out/'experiment_report.html').write_text(content,encoding='utf-8')
