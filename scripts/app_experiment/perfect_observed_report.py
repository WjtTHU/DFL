"""Report a privileged observed-Lot counterfactual without fitting or tuning."""
from pathlib import Path
from html import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from scripts.app_dataset.generate import write_csv

METHOD='perfect_observed_diagnostic'
NAMES={'point_transformer':'点预测 PTO','historical_product_cycle':'历史商品周期基线',
       'nll_expected_supply':'NLL 期望到货 PTO',METHOD:'已有快照 Lot 完美预测诊断',
       'oracle':'全 T 完美信息 Oracle'}


def table(frame):
    return frame.to_html(index=False,border=0,float_format=lambda x:f'{x:,.4f}')


def render(out,s):
    out=Path(out);oracle=s['oracle'];delta=s['diagnostic_comparison'];scope=s['truth_scope']
    comparison=pd.DataFrame(oracle['comparison'])
    comparison['regret_percent']=100*comparison.relative_regret_over_oracle
    comparison=comparison[['method','policy_cost','absolute_regret','regret_percent']]
    comparison=pd.concat([comparison,pd.DataFrame([{'method':'oracle','policy_cost':oracle['costs']['total_cost'],
                         'absolute_regret':0.,'regret_percent':0.}])],ignore_index=True)
    write_csv(out/'cost_comparison.csv',comparison)
    comparison['method']=comparison.method.map(NAMES)
    components=pd.DataFrame([{'method':NAMES[m['method']],**m['costs']} for m in s['methods']]+
                           [{'method':NAMES['oracle'],**oracle['costs']}])
    frames={m['method']:pd.read_csv(out/m['method']/'period_costs.csv') for m in s['methods']}
    frames['oracle']=pd.read_csv(out/'oracle/oracle_period_costs.csv')
    rows=[]
    for method,frame in frames.items():
        rows.append(frame.assign(method=method))
    write_csv(out/'weekly_cost_comparison.csv',pd.concat(rows,ignore_index=True))
    point=frames['point_transformer'].set_index('period')
    diag=frames[METHOD].set_index('period')
    differences=pd.DataFrame({'period':point.index,
        'point_cost':point.total_cost.to_numpy(),'diagnostic_cost':diag.total_cost.to_numpy(),
        'cost_reduction':(point.total_cost-diag.total_cost).to_numpy(),
        'point_secondary_ordered':point.secondary_ordered_die.to_numpy(),
        'diagnostic_secondary_ordered':diag.secondary_ordered_die.to_numpy(),
        'point_backlog_end':point.customer_backlog_end.to_numpy(),
        'diagnostic_backlog_end':diag.customer_backlog_end.to_numpy()})
    write_csv(out/'point_diagnostic_period_differences.csv',differences)
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    styles={'point_transformer':('#397bbb','--','o'),'historical_product_cycle':('#d78835',':','s'),
            'nll_expected_supply':('#8c6bab','-.','D'),METHOD:('#00856e','-','^'),
            'oracle':('#46545d',':','x')}
    for method,frame in frames.items():
        frame=frame.assign(cumulative_realized_cost=frame.total_cost.cumsum())
        color,ls,marker=styles[method]
        for ax,col in zip(axes.flat,['cumulative_realized_cost','customer_backlog_end','secondary_ordered_die','production_total']):
            ax.plot(frame.period,frame[col],label=NAMES[method],color=color,linestyle=ls,marker=marker,markersize=4)
    for ax,title in zip(axes.flat,['累计实际成本','每期期末客户积压','当期第二货源下单','当期生产']):
        ax.set(title=title,xlabel='执行期',xticks=range(1,8));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(out/'experiment_figures.png',dpi=150);plt.close(fig)
    change=delta['point_minus_diagnostic_cost']
    phrase=f'相比点预测 PTO，成本{"降低" if change>=0 else "增加"} {abs(change):,.2f}'
    diagnostic=next(m for m in s['methods'] if m['method']==METHOD)
    checks=pd.read_json(out/'information_scope_checks.json')
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>已有快照 Lot 完美预测诊断</title><style>
body{{background:#f3f6f8;color:#18383e;font:15px/1.8 "Microsoft YaHei",sans-serif;margin:0}}main{{max-width:1250px;margin:28px auto;padding:0 24px}}
header,section{{background:white;padding:24px;margin-bottom:18px;border:1px solid #dce6e9;border-radius:10px}}header{{border-top:6px solid #00856e}}
h1{{font-size:28px}}h2{{font-size:21px}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:8px;border-bottom:1px solid #dfe8eb;text-align:right;white-space:nowrap}}th{{background:#edf4f4}}
.scroll{{overflow-x:auto}}img{{width:100%}}.note{{padding:15px;background:#fff4dd;border-left:4px solid #c68c34}}a{{color:#087d71}}code{{overflow-wrap:anywhere}}
</style><main><header><h1>已有快照 Lot 完美预测诊断</h1>
<p>同一 3 材料、3 产品测试轨迹：2025-01-07 至 02-24；7 天一期、N=4、T=7。每次计划 4 期，只执行第一期，状态连续结转。</p>
<p class="note">{phrase}。相对全 T Oracle 的 regret 为 <b>{delta['diagnostic_regret_absolute']:,.2f}（{100*delta['diagnostic_regret_relative']:.2f}%）</b>。
这是使用未来标签的诊断实验，不是可上线的预测器；完工仍按企业快照规则推定。</p></header>
<section><h2>五种方法的同口径比较</h2><div class="scroll">{table(comparison)}</div>
<p>Regret = 方法在固定 7 期内的实际总成本 − 全 7 期完美信息最优成本；比例以 Oracle 成本为分母。金额为合成实验成本单位。
原点预测、NLL 和历史基线的已完成执行轨迹原样复制，未重训或重新调参。Oracle 再次求解后与原值一致。</p>
<div class="scroll">{table(components)}</div><img src="experiment_figures.png" alt="五种方法的实际累计成本、积压、采购和生产"></section>
<section><h2>只替换什么信息</h2>
<p>冻结每次决策的原订单集合与 has_snapshot 标记：{scope['exact_observed_rows']:,} 条已有快照的订单视图（{scope['exact_observed_lots']:,} 个不同 Lot）使用其完工标签，
按“完工日期减预测日期”的整数日差映射到到货周；{scope['promise_only_rows']:,} 条尚无快照的订单视图继续使用原合成承诺日期。
缺失已观测 Lot 的完工标签为 {scope['unknown_observed']} 条，因此没有删除、填补或回退。订单数量、公告时间、快照可见性均未变化。</p>
<p>对未观测订单不使用真实完工信息，也不把未来新出现的快照提前加入。真实完工标签保存在单独的特权信息核验表中；诊断预测文件明确标记 diagnostic_truth_used。
这不是声称生产环境能够提前获得这些标签。</p>
<div class="scroll">{table(checks)}</div>
<p>相对原点预测，共 {scope['point_vs_exact_different_week_rows']} 条已有快照订单的预测到货周发生变化；原点预测在该子集的 MAE 为 {scope['point_mae_on_exact_rows']:.4f} 天。
没有跨越周边界的日级预测修正不会改变本 APP 的供给周。</p></section>
<section><h2>这次结果能说明什么</h2>
<p>诊断相对点预测消除了本条轨迹上 {100*delta['fraction_of_point_oracle_gap_closed_on_this_trajectory']:.2f}% 的原始 Oracle 成本差距。
该数字是保持其他规则不变的反事实结果，不是可实现的改善保证，也不是算法的正式下界。</p>
<p>若诊断比点预测明显改善，说明这些已有快照批次的时间误差在当前参数下具有决策影响；即使如此，剩余差距仍混合了未观测订单承诺误差、尚未公告的订单信息、有限 N=4 滚动视野与末端处理。
不能仅凭剩余 regret 分辨每一项贡献，或把它解释为无法再降低的成本。有限视野下更准确的部分信息也不保证任意轨迹成本单调下降。</p>
<div class="scroll">{table(differences)}</div></section>
<section><h2>执行与核验</h2><ul>
<li>新增 7 次滚动 CPLEX 求解与一次 Oracle 求解均达到最优；诊断最大规划约束误差 {diagnostic['max_planning_constraint_residual']:.3g}，执行违反 {diagnostic['max_execution_violation']:.3g}。</li>
<li>从保存的 CPLEX XML 独立回放四条滚动轨迹，核验生产、库存、积压、采购管道和费用；最大差异 {s['post_run_audit']['max_audit_numeric_difference']:.3g}，容差 2e-5。</li>
<li>真实到货一批一次、固定数量不变；本期到货只可供下期生产，当前生产只用期初库存。诊断不是将未来物料加入当前可用库存。</li>
<li>BOM、需求、产能、成本、初始化、订单规则与第二货源提前期完全沿用。末期采购在第 8 期收货，不计入前 7 期实际成本，仍保留订单管道。</li>
<li>原始 CSV、预测源码与模型、冻结数据及旧结果的哈希均未变化；本次诊断源码及配置另存快照，未提交 Git。</li></ul>
<p>限制：单条测试轨迹、单一需求种子、探索性材料选择、全历史质量筛选、企业快照推定完工及合成提前订单。此次只诊断已有快照 Lot 的预测价值，不重新校准下游参数。</p></section>
<section><h2>结果文件</h2><p>
<a href="experiment_summary.json">汇总结果</a> · <a href="cost_comparison.csv">五方成本比较</a> ·
<a href="diagnostic_forecasts.csv">诊断供给输入</a> · <a href="privileged_completion_audit.csv">特权完工标签核验</a> ·
<a href="point_diagnostic_period_differences.csv">点预测与诊断逐期差异</a> · <a href="post_run_audit.json">独立回放核验</a> ·
<a href="oracle/oracle_summary.json">Oracle</a></p>
<p>数据：{escape(s['dataset'])}<br>参考结果：{escape(s['reference_run'])}<br>本次用时：{s['elapsed_seconds']:.1f} 秒（不含报告绘图）。</p></section></main></html>'''
    (out/'experiment_report.html').write_text(html,encoding='utf-8')
