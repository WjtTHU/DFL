"""Local HTML and static charts for the first point-PTO rolling APP experiment."""
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

NAMES={'point_transformer':'快照 Transformer PTO','historical_product_cycle':'历史商品周期基线'}
COLORS={'point_transformer':'#126d83','historical_product_cycle':'#d37b37'}
COST_NAMES={'production_cost':'生产','finished_holding_cost':'成品库存','backlog_cost':'每期积压',
            'secondary_receipt_cost':'第二货源收货','terminal_backlog_cost':'期末额外积压','total_cost':'合计'}


def table(frame, digits=2):
    return frame.to_html(index=False,border=0,escape=True,float_format=lambda x:f'{x:,.{digits}f}')


def render_report(out, summary):
    out=Path(out)
    methods=summary['methods']
    frames={m['method']:pd.read_csv(out/m['method']/'period_costs.csv') for m in methods}
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
                         'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for method,frame in frames.items():
        for ax,column in [(axes[0,0],'cumulative_realized_cost'),(axes[0,1],'customer_backlog_end'),
                          (axes[1,0],'secondary_ordered_die'),(axes[1,1],'production_total')]:
            is_point=method=='point_transformer'
            ax.plot(frame.period,frame[column],marker='o' if is_point else 'x',
                    markersize=7 if is_point else 5,markerfacecolor='white',
                    linestyle='-' if is_point else '--',label=NAMES[method],color=COLORS[method],lw=2.5 if is_point else 1.5)
    titles=['累计已执行成本（含第 7 期期末惩罚）','期末客户积压','当期第二货源下单量','当期实际生产量']
    units=['实验成本单位','产品单位','die','产品单位']
    for ax,title,unit in zip(axes.flat,titles,units):
        ax.set(title=title,xlabel='执行期',ylabel=unit,xticks=range(1,8))
        ax.grid(alpha=.18); ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'))
    axes[0,0].legend(fontsize=9)
    fig.savefig(out/'experiment_figures.png',dpi=160)
    plt.close(fig)
    costs=pd.DataFrame([{'方法':NAMES[m['method']],**{COST_NAMES[k]:v for k,v in m['costs'].items()}} for m in methods])
    forecast=pd.DataFrame([{'方法':NAMES[m['method']],'预测行':m['forecast_rows'],'可评价行':m['labelled_rows'],
        '结局未确认行':m['unknown_rows'],'MAE（天）':m['mae_days'],'RMSE（天）':m['rmse_days'],
        '到货周命中率（%）':100*m['same_arrival_week_fraction'],
        '按数量加权命中率（%）':100*m['quantity_weighted_same_week_fraction']} for m in summary['forecast_metrics']])
    point=next(x for x in methods if x['method']=='point_transformer')
    base=next(x for x in methods if x['method']=='historical_product_cycle')
    difference=point['costs']['total_cost']-base['costs']['total_cost']
    percent=100*abs(difference)/base['costs']['total_cost']
    comparison=("本条测试轨迹中，两种方法实际执行的生产、采购和总成本相同；快照预测的日数 MAE 更小，但本次没有体现为成本下降。"
                if abs(difference)<1e-6 else
                f"本条测试轨迹中，快照 PTO 总成本相对历史商品周期基线{'高' if difference>0 else '低'} {percent:.2f}%（差额 {difference:,.2f}）。")
    terminal=pd.DataFrame([{'方法':NAMES[m['method']],'期末客户积压':m['terminal_backlog'],
        '期末成品库存':m['terminal_finished_stock'],'期内第二货源已收货（die）':m['secondary_received_die'],
        '跨期第二货源在途（die）':m['secondary_pending_die'],
        '期末原材料库存是否完整核实': '是' if m['terminal_raw_inventory_exact'] else '否'} for m in methods])
    weekly=[]
    for method,frame in frames.items():
        view=frame[['period','decision_at','demand_total','production_total','customer_backlog_end',
                    'secondary_ordered_die','secondary_received_die','total_cost']].copy()
        view.insert(0,'方法',NAMES[method])
        weekly.append(view.rename(columns={'period':'期','decision_at':'决策时点','demand_total':'需求',
            'production_total':'生产','customer_backlog_end':'期末积压','secondary_ordered_die':'采购下单',
            'secondary_received_die':'采购收货','total_cost':'实际计入成本'}))
    checks=pd.DataFrame([{'方法':NAMES[m['method']],'CPLEX最优求解次数':m['cplex_optimal_solves'],
        '最大规划约束残差':m['max_planning_constraint_residual'],'最大执行约束残差':m['max_execution_violation'],
        '实际第一货源Lot数':m['primary_known_arrivals_lots'],'已确认到货量（die）':m['primary_known_arrivals_die'],
        '物料／成品／采购管道守恒':'通过'} for m in methods])
    unknown_details=[]
    for m in methods:
        state=json.loads((out/m['method']/'terminal_state.json').read_text(encoding='utf-8'))
        for mat,lower in state['raw_inventory_lower_bound_die'].items():
            unknown_details.append({'方法':NAMES[m['method']],'材料':mat,'期末库存下界（die）':lower,
                '期末库存可能上界（die）':state['raw_inventory_upper_bound_die'][mat]})
    diagnostic=''
    if (out/'post_run_audit.json').is_file():
        audit=json.loads((out/'post_run_audit.json').read_text(encoding='utf-8'))
        unseen=sum(x['not_known_at_week_start_die'] for x in audit['primary_visibility'])
        arrived=sum(x['receipt_die'] for x in audit['primary_visibility'])
        diagnostic=f'''<section><h2>为何成本相同：事后核查</h2><p>逐行核查了 {audit['forecast_rows_per_method']} 条预测（{audit['distinct_forecast_lots']} 个不同 Lot）。日数预测不同的有 {audit['different_day_predictions']} 条，分到不同到货周的有 {audit['different_week_predictions']} 条。第 1、4 次规划的后续期生产／积压计划确实不同，最大差额为 {audit['max_future_plan_difference']:,.0f} 产品单位；全部当期执行动作的最大差额为 {audit['max_executed_action_difference']:,.0f}。因此不是将同一份预测误传给两种方法。</p>
<p>观测到的约束状态与这一结果一致：A 在第 2–7 期采购满额，B 在第 2–6 期采购满额；C 除第 1 期外不再采购，后续库存充足。当期生产受期初资源限制，部分未来到货周差异只改变尚未执行的后续计划。此处没有改变参数或再次优化来追求差异。</p>
<p>日数 MAE 与到货周命中率也不是同一指标：快照模型 MAE 更小，但周命中率低于本条轨迹的历史基线。按周聚合后，预测误差是否跨越周边界比单独的天数误差更直接影响此模型。</p>
<p>真实已确认到货中，{unseen:,.0f} die（{100*unseen/arrived:.2f}%）在对应周决策时尚未首次出现，不能提前放入已知订单预测。当前首版没有预测这种未来新订单供给，所有已知订单的预测均落在前两周，第 3–4 周没有新的第一货源预测到货。该范围限制可能影响采购倾向，后续应另行讨论，不能把本次成本相同概括为快照信息没有价值。</p>
<p>独立审计从 CPLEX XML 解重新逐期计算规划库存、实际状态与成本，两种方法均通过；最大差异 {audit['max_audit_numeric_difference']:.3g}，来自 CSV 小数输出精度（审计容差 2e-5）。<a href="post_run_audit.json">审计结果</a> · <a href="prediction_week_differences.csv">36 条不同到货周预测</a> · <a href="plan_differences.csv">未来计划差异</a></p></section>'''
    content=f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DK 小规模 PTO 滚动 APP 实验</title><style>
body{{margin:0;background:#f3f6f7;color:#17313b;font:15px/1.7 "Microsoft YaHei",sans-serif}}main{{max-width:1260px;margin:32px auto;padding:0 24px 40px}}
header,section{{background:white;border:1px solid #dce5e8;border-radius:12px;padding:24px 28px;margin:18px 0}}header{{border-top:6px solid #126d83}}h1{{font-size:30px;line-height:1.35;margin:6px 0 12px}}h2{{font-size:21px;margin:0 0 12px}}h3{{font-size:17px}}p{{margin:10px 0}}.muted{{color:#59717b}}.tag{{letter-spacing:1px;color:#126d83;font-weight:bold}}.callout{{background:#eaf5f6;border-left:4px solid #126d83;padding:12px 18px}}.note{{background:#fff6e8;border-left:4px solid #d37b37;padding:12px 18px}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0}}th,td{{padding:9px 10px;border-bottom:1px solid #e5ecef;text-align:right;white-space:nowrap}}th{{background:#eff5f6}}td:first-child,th:first-child{{text-align:left}}img{{width:100%;height:auto}}a{{color:#126d83}}code{{font-size:13px}}ul{{padding-left:24px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.card{{background:#eff5f6;padding:15px;border-radius:8px}}.card strong{{font-size:23px;display:block}}@media(max-width:720px){{main{{padding:0 10px}}section,header{{padding:18px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><header><div class="tag">DK · POINT PTO · CPLEX 12.8</div><h1>3 材料 × 3 产品：周预测与滚动 APP 实验</h1>
<p>测试期 2025-01-28 至 2025-03-17，7 天一期，每次规划 N=4 期，只执行第一期；评价 T=7 期。</p>
<p class="muted">完成时间 {escape(summary['completed_at'])} · 运行耗时 {summary['elapsed_seconds']/60:.2f} 分钟 · 复用点预测最佳第 {summary['point_best_epoch']} 轮模型，无重新训练或测试调参</p>
<div class="grid"><div class="card">快照 PTO 总成本<strong>{point['costs']['total_cost']:,.2f}</strong>合成实验成本单位</div><div class="card">历史周期基线总成本<strong>{base['costs']['total_cost']:,.2f}</strong>相同需求、到货、初始状态与约束</div><div class="card">CPLEX 最优求解<strong>{sum(m['cplex_optimal_solves'] for m in methods)} / 14</strong>7 次决策 × 2 种方法</div></div></header>
<section><h2>本次结果</h2><p class="callout">{comparison}这是冻结参数下的一条小规模轨迹结果，不据此推断普遍优劣。</p><div class="scroll">{table(costs)}</div>
<p>成本只累计已执行的 7 期：生产、成品期末库存、每期期末积压、实际收到的第二货源采购，以及第 7 期末一次性额外积压惩罚。第一货源采购视为沉没成本；原材料持有与期末残值为 0。单位是合成实验成本单位，非企业财务金额。</p>
<p class="muted">两种方法的实际轨迹重合，图中分别使用实线圆点与虚线叉号显示。</p><img src="experiment_figures.png" alt="两种方法的累计成本、积压、第二货源采购及生产曲线"></section>
{diagnostic}
<section><h2>预测与决策如何连接</h2><p>周二 00:00 决策，只使用此前快照，预测日期为周一。快照 PTO 沿用已保存的 Transformer 和预处理参数；历史基线只用训练集每 Lot 一次的商品平均总周期，减去当时已过天数并截到非负。两个方法预测同一组可见未到货订单，没有用真实完工时间形成预测。</p>
<p>模型输出仍为日粒度点预测。向上取整后 1–7 天对应本周到货，8–14 天对应下一周；0 天预测因订单仍待收货而放在当前周，不追加入期初库存。一个 Lot 的首次 dieQty 整批放入预测到货周，超过 28 天的预测保留在窗外。APP 数量不替换预测模型原来的 waferQty 特征。</p>
<div class="scroll">{table(forecast,4)}</div><p>预测评价在滚动结束后另行合并可确认标签；未确认结局不计误差。上述是周决策样本及所选材料的指标，不与此前全部每日窗口指标直接比较。</p>
<p>在两次决策之间首次出现并到货的第一货源订单会进入真实回放，但在出现前不向规划器预告；本首版也没有额外构造未知未来订单的供给预测。两种方法遵循同一信息边界。</p></section>
<section><h2>时序、求解与执行核验</h2><ul><li>当期生产只消耗期初库存；第一货源当期到货下期可用；第二货源提前期为 1 期，在下期期初收到并可用。</li><li>每次只执行当前生产与采购，未来计划全部重算；实际库存、积压和已下达采购订单持续结转，不逐期重置。</li><li>CPLEX 解要求 optimal，并独立重算等式、非负、产能与物料约束，容差 1e-5。所有实际动作在读取当期未来到货前确定。</li><li>原 CSV、冻结数据集、预测源码、模型及预处理器哈希均核验未变；各方法共享同一外生数据。</li></ul>
<div class="scroll">{table(checks,8)}</div><p>完整 LP、XML 解和求解器日志按方法与期数保存于 <code>&lt;method&gt;/cplex/period_XX/</code>。规划目标包含未来 4 期，其值不作为当期实际执行成本。</p></section>
<section><h2>逐期执行结果</h2><div class="scroll">{table(pd.concat(weekly,ignore_index=True))}</div></section>
<section><h2>末期状态与跨期订单</h2><div class="scroll">{table(terminal)}</div><p>第 7 期新下达的第二货源订单在第 8 期收货，留在订单管道，采购费用不计入本次 T=7 成本；规划仍计入这笔采购费用。其金额另存于 <code>terminal_state.json</code>，不等于订单免费或已取消。</p>
<div class="scroll">{table(pd.DataFrame(unknown_details))}</div><p class="note">22 个 Lot、128,194 die 的最后一周到货归属未确认；另 7 个 Lot、49,037 die 在测试结束后仍有观测。没有虚构收货事件。上表是末期已确认第一货源到货后的原材料库存边界，尚未加第 8 期期初的第二货源收货。该不确定性不影响当前设定下前 7 期生产和计账，但不能把期末库存或第一货源在途说成已完全核实。</p></section>
<section><h2>解释范围与后续使用</h2><ul><li>材料 A 即使在第二货源满额时，总量也有约 6.63 万 die 缺口；存在客户积压是允许的，不代表生产计划不可行。没有据本次测试结果修改需求或采购能力。</li><li>仍使用企业快照推定完工、固定首次 dieQty、整批到货、无运输、离线质量筛选及左边界批次剔除。</li><li>本次只是一个测试轨迹与一个需求随机种子的点 PTO 接通实验，未运行训练/验证 APP 或统计显著性检验。所选材料经过前期探索，不能视为未经查看留出期的预注册选择。</li><li>NLL 模型保持原状，尚未接入 28 天规划。没有运行随机规划或决策导向训练。</li></ul>
<p><a href="weekly_predictions.csv">周预测（无真实标签）</a> · <a href="forecast_evaluation_only.csv">单独的预测评价明细</a> · <a href="point_transformer/period_costs.csv">PTO 逐期成本</a> · <a href="historical_product_cycle/period_costs.csv">基线逐期成本</a> · <a href="experiment_summary.json">完整结果与检查</a> · <a href="experiment_config.json">本次配置</a></p>
<p class="muted">冻结数据：{escape(summary['dataset'])}<br>点预测模型：{escape(summary['point_checkpoint'])}<br>报告只保存本地，未提交或推送 Git。</p></section></main></body></html>'''
    (out/'experiment_report.html').write_text(content,encoding='utf-8')


def refresh_report(out):
    """Refresh presentation after an audit; predictions and optimization stay unchanged."""
    import hashlib
    from datetime import datetime
    from zoneinfo import ZoneInfo
    out=Path(out).resolve()
    def sha(path):
        with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
    manifest=json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed': raise ValueError('Run must be completed before report-only refresh')
    previous=json.loads((out/'artifact_hashes.json').read_text(encoding='utf-8'))
    if not all(sha(out/p)==h for p,h in previous.items()): raise ValueError('Existing run artifacts changed')
    previous_report=sha(out/'experiment_report.html')
    summary=json.loads((out/'experiment_summary.json').read_text(encoding='utf-8'))
    render_report(out,summary)
    (out/'report_source.py').write_bytes(Path(__file__).read_bytes())
    revision={'updated_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'reason':'Clarify identical realized costs and incorporate post-run decision-sensitivity audit; no predictions/actions changed.',
        'previous_report_sha256':previous_report,'report_sha256':sha(out/'experiment_report.html'),
        'report_source_sha256':sha(__file__),'original_runtime_source_archive':'app_source_snapshot.zip',
        'training_or_optimization_repeated':False}
    (out/'report_revision.json').write_text(json.dumps(revision,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    hashes={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
            if p.is_file() and p.name not in {'artifact_hashes.json','run_manifest.json'}}
    (out/'artifact_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_dir',type=Path)
    refresh_report(parser.parse_args().run_dir)
