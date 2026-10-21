# 版本迭代记录

“原版”固定指首次本地 CSV 基线 `abfce8f`。

2026-10-15 每日预测版至 10-21 诊断版是未单独提交的阶段记录，相关代码现统一纳入 `pto-baseline-20261021`；表中“未提交”保留当时状态。每个 Git 提交／标签均有对应的 README，模型、数据和实验结果另存本地。

| 日期 | 版本 | 主要变动 | 回溯定位 / 状态 |
| --- | --- | --- | --- |
| 2026-10-15 | 原版 | 整合企业 CSV 入口与配置，适配本地路径和启动方式；保留原预测逻辑。 | [`abfce8f`](https://github.com/WjtTHU/DFL/commit/abfce8f3ef54d6c6b6a56da314ca0f5bb543a96c)，已提交并推送 |
| 2026-10-15 | 缩放修正版 | 修复标签缩放不一致；回归测试、DK 全量重跑及对比报告已完成。 | `114e023`；tag：`scaler-fix-20261015`，已推送 |
| 2026-10-15 | 每日预测版 | 日末预测、整数日标签、独立时间测试集；保留日内快照，新增 NLL 模式及商品周期条件分布基线。 | 未提交；10 月 16 日实验及基线比较完成，测试 MAE 1.4901 天 / NLL 2.3838；[报告](../outputs/dk/daily/full_runs/20261016_1218/experiment_report.html) |
| 2026-10-17 | APP 时间划分版 | 训练／验证边界改为 2024-12-31／2025-01-28；完成点预测与 NLL 重跑，旧模型与结果保留。 | 未提交；测试 MAE 2.0937 天 / NLL 2.0289；[报告](../outputs/dk/daily/full_runs/20261017_1908_pto_split/experiment_report.html) |
| 2026-10-19 | APP 小规模数据版 | 新增独立的 3 材料、3 产品数据构造与检查；7 天一期、N=4、测试 T=7，预测模型不变。 | 未提交；[配置](../config/app_dk_small.yaml)、[说明](app_dataset.md)、[检查报告](../outputs/app/dk_3x3_v1_20261019/dataset_report.html) |
| 2026-10-19 | 点 PTO 滚动实验版 | 补齐周预测、接入 CPLEX，完成 7 期滚动执行及历史周期基线比较；两者成本相同。 | 未提交；[说明](app_pto_experiment.md)、[实验报告](../outputs/app/pto_runs/20261019_point_v1/experiment_report.html) |
| 2026-10-20 | 10 周训练＋28 天分布版 | 前移划分，完成点预测与 1–28 天加尾部 NLL 重跑；新增提前 28 天订单规则。 | 未提交；测试 MAE 2.6374 天／NLL 2.2076；[报告](../outputs/dk/daily/full_runs/20261020_train10w_h28/experiment_report.html)、[旧结果归档](../outputs/archives/dk_app_before_20261020.zip)，旧模型与报告保留 |
| 2026-10-20 | 提前订单 PTO 版 | 新划分下接入提前 28 天订单，完成点 PTO、历史基线和全 T Oracle；预测模型不变。 | 未提交；PTO 成本比基线低 3.96%，Oracle regret 14.72%；[报告](../outputs/app/pto_runs/20261020_point_advance28_v2/experiment_report.html)、[配置](../config/app_pto_10w_advance28.yaml) |
| 2026-10-21 | NLL 期望供给 PTO 版 | 周到货概率转换为期望数量，接入同一 APP；与点预测、历史基线及 Oracle 比较，无重训。 | 未提交；成本 747,176.12，比点 PTO 高 4.64%；[报告](../outputs/app/pto_runs/20261021_nll_expected_v1/experiment_report.html)、[配置](../config/app_nll_expected_28d.yaml) |
| 2026-10-21 | 已观测 Lot 完美预测诊断 | 已有快照批次改用完工标签，其余订单保留承诺；N=4 不变，完成五方比较。 | 未提交；成本 714,485.97、regret 14.79%，未改善点 PTO 总成本；[报告](../outputs/app/pto_runs/20261021_perfect_observed_v1/experiment_report.html)、[配置](../config/app_perfect_observed.yaml) |
| 2026-10-21 | PTO 完整基线版 | 汇总点预测／NLL、APP 数据与 CPLEX 滚动、历史基线、Oracle 和诊断；63 项测试通过。 | tag：[`pto-baseline-20261021`](https://github.com/WjtTHU/DFL/tree/pto-baseline-20261021)；[本版 README](https://github.com/WjtTHU/DFL/blob/pto-baseline-20261021/README.md)，代码发布，产物保留本地 |

后续重要版本每次追加一行，填写实际 commit 或 tag，便于回溯。
