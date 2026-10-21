# 本地运行说明

首次验证选用 DK。现有 CSV 和企业配置足以运行，不需要 PostgreSQL，也不需要重新人工清洗。

本页记录原版与缩放修正版。缩放修正版已提交并推送为 `114e023`（tag：`scaler-fix-20261015`）；后续每日预测和滚动 APP 统一保存于 `pto-baseline-20261021`。每日预测使用独立 `daily-*` 命令，见[每日预测使用说明](daily_prediction.md)。下文历史指标不代表每日预测版效果。

## 已准备的环境

2026-10-15 在当前 Windows 机器上完成检查与小规模验证：

| 项目 | 当前配置 |
| --- | --- |
| CPU | Intel Core i7-14700KF，28 个逻辑处理器 |
| 内存 | 约 32 GiB |
| 已检测到的实体显卡 | Intel Iris Xe Graphics；未检测到 NVIDIA 显卡 |
| 项目 Python | `.venv\Scripts\python.exe`，Python 3.12.14 |
| PyTorch | 2.9.0+cpu，CPU 张量计算验证通过 |
| 依赖 | `pip check` 通过；训练及预测入口所需模块齐全 |

系统默认 `python` 仍是 Python 3.10.11，低于项目要求。运行本项目时使用 `.venv` 中的 Python，不需要修改系统 Python。

本次 `.venv` 使用本机已有的 Python 3.12.14 创建，基础解释器位于 `C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`。虚拟环境不纳入 Git，也不应跨电脑复制；基础解释器位置失效时，用新的 Python 3.12 重新创建环境。

`requirements/local-cpu.txt` 提供 CSV 训练和推理所需的依赖，版本约束来自企业 `uv.lock`；PyTorch 保持 2.9.0，只改用官方 CPU 构建。该环境不安装完整数据库/API 服务依赖。原 `pyproject.toml` 和 `uv.lock` 保留原样，其中 Windows 的 Torch 源仍是 CUDA 12.8，因此此 CPU 环境按下述 pip 清单安装，不使用 `uv sync` 替代。

在新机器上准备 Python 3.12 后，从项目根目录执行：

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements/local-cpu.txt
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe run.py check --dataset dk --environment
```

如网络需要代理，通过 pip 的 `--proxy` 参数使用当地已配置的代理，不在代码中固定代理地址。

## 小规模流程验证

在 `E:\THU\Research\DFL_CODE\local_wip` 中运行：

```powershell
& .\.venv\Scripts\python.exe scripts/debug/smoke_csv_pipeline.py --dataset dk
```

该脚本先调用企业入口对全量 CSV 预处理，再抽取完整 Lot 历史进行功能验证。默认抽取 32 个训练候选 Lot、16 个验证候选 Lot，训练 1 个 epoch，batch size 32，CPU 线程数 4。为保证功能测试包含可构造窗口的历史，候选 Lot 要求观测时长大于 3 天；这只是验证样本的选择条件，不是正式训练新增的清洗规则。标签与训练/验证样本仍由原训练函数生成，原流程可能继续排除边界 Lot。

字段定义、时间分箱、产品一致性检查、模型结构、训练/验证时间切分和原 YAML 均不修改。小规模样本上的产品统计量和指标不代表全量训练结果。

2026-10-15 的验证结果：

- DK 原始输入 803,546 行，标准化后 799,319 行、11,332 个 Lot。
- 抽取 48 个完整 Lot 历史，共 3,915 行；原完成批次规则保留 46 个可用于构造标签的 Lot。
- 完成 1 轮训练和验证，生成 `best_model.pth`、`preprocessor.pkl`、`config.yaml`。
- 重新加载 checkpoint 后为 5 个 Lot 输出有限数值预测，结果与内存中的模型一致。
- 另起 Python 进程，通过 `run.py predict` 重新读取全量 DK CSV，预测命令成功退出。

本次结果在 `outputs/dk/smoke/20261015_172641_071452/`：

- `run_manifest.json`：样本选择、环境、覆盖范围、验证状态。
- `checkpoint/`：小规模验证模型与预处理器。
- `predictions.csv`：5 个 Lot 的预测剩余天数。
- `experiments/training.log`、`pipeline.log`：训练与处理日志。

上述预测仅用于确认流程通畅，不能用这一轮训练的指标或模型判断业务预测质量。

## 正式训练与预测

已按企业专用配置完成首次 DK 全量训练；重新运行的命令为：

```powershell
& .\.venv\Scripts\python.exe run.py train --dataset dk --no-plot
```

原配置为 batch size 128、最多 80 个 epoch，并保留原早停策略。`--no-plot` 关闭逐轮绘图，不修改模型、预处理或训练/验证切分。首次全量运行设置 `OMP_NUM_THREADS=4`、`MKL_NUM_THREADS=4`；2026-10-15 17:40:56 建立运行记录，19:10:40 正常结束，全程约 89.7 分钟，在第 38 轮触发早停。

### 首次全量结果与指标口径

- 训练窗口 73,901 个，来自 8,801 个 Lot；验证窗口 3,397 个，来自 1,750 个 Lot，训练与验证标签 Lot 不重叠。
- 原日志选出的最佳模型为第 8 轮，已在独立进程重新加载；按原训练器口径重算 MAE 1.365817，与日志 1.3658 一致。
- **存在目标缩放不一致：训练范围为 1–62 天，验证数据集自动扩展为 1–108 天，但训练器仍按训练范围还原验证标签。** 日志实际比较的标签是 `1 + (原始剩余天数 - 1) × 61 / 107`，因此日志指标不能直接当作原始天数误差。早停和选模同样受此口径影响。
- 保持模型预测不变，改为与原始 `RCT_in_days` 标签比较：MAE **2.1949 天**，RMSE **5.0334 天**，中位绝对误差 **1.0390 天**，82.16% 的窗口误差不超过 3 天。
- 仅按训练商品平均周期计算的基线 MAE 为 2.8109 天，本模型相对降低约 21.9%。每个 Lot 等权平均的 MAE 为 2.1397 天。
- 标签是快照规则推定的最后观测时间；验证集用于选模，且有 481 个验证窗口的参考快照时间不晚于训练截止。以上不是独立测试集或严格上线回测结果。首次实验的企业训练、清洗、标签和配置未作修改；后续经用户授权完成的缩放修正见下节。

本机产物（由 `.gitignore` 排除，远端 Git 仓库不含这些文件）：

- [实验报告](../outputs/dk/full_runs/20261015_174056/experiment_report.html)
- [评估指标 JSON](../outputs/dk/full_runs/20261015_174056/evaluation_summary.json)
- [全部验证窗口预测](../outputs/dk/full_runs/20261015_174056/validation_predictions.csv)
- [每个验证 Lot 的最晚窗口预测](../outputs/dk/full_runs/20261015_174056/validation_latest_per_lot.csv)
- [最佳模型目录](../outputs/dk/checkpoints/20261015_174243/)

报告脚本复用训练与验证数据集构造函数，并核验数据 SHA-256、配置文件和保存模型的日志指标。重建历史实验需要使用该次训练对应的源代码；当前脚本会检查运行记录中的源代码哈希，避免用修正后的代码覆盖原版本报告。以下历史命令需在恢复该次源代码的独立目录中使用：

```powershell
& .\.venv\Scripts\python.exe scripts/evaluation/report_dk_run.py --run-dir outputs/dk/full_runs/20261015_174056 --checkpoint outputs/dk/checkpoints/20261015_174243 --experiment outputs/dk/experiments/20261015_174243 --threads 4
```

### 标签缩放修正与重跑

2026-10-15 用户授权修正后，工作区做了两处功能修改：

- `LotwiseIdxRegDataset` 仅在 `fit_scaler=True` 时更新标签范围；验证与推理完整复用传入的训练参数，超出范围的标签不裁剪。
- 数据集统一入口检查传入范围与生成后范围是否一致，发现漂移立即报错。

`tests/test_target_scaling.py` 的 5 项测试全部通过，覆盖训练范围拟合、验证上下界外标签往返、训练器 MAE 的原始天数口径、推理参数复用以及范围漂移拒绝。修改前其中 3 项失败，验证了测试确实能捕获该问题。

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_target_scaling.py -v
```

全量重跑记录：`outputs/dk/full_runs/20261015_195345_scaler_fix/`；模型和训练日志分别写入 `outputs/dk/checkpoints/20261015_195531/`、`outputs/dk/experiments/20261015_195531/`。重跑采用原 CSV、原 YAML、seed 42、batch 128、最多 80 轮和 patience 30，CPU 计算线程仍为 4。2026-10-15 19:53:46 开始，21:35:42 正常结束，退出码为 0，耗时约 101.9 分钟；第 42 轮触发早停，最佳模型为第 12 轮。21:41 完成独立进程重载与报告核验。

两次实验的 CSV 哈希、源 YAML 及逐验证窗口的 Lot、参考快照时间和标签一致。训练与验证均使用固定范围 `[1, 62]`，验证原始标签仍为 1–108 天，没有裁剪。重载模型的原始天数 MAE 为 2.042418823，与训练器重算结果及日志最佳值 2.0424 一致。

| 对象 | 验证 MAE / 天 | RMSE / 天 | 窗口误差 ≤ 3 天 |
| --- | --- | --- | --- |
| 原版保存模型（第 8 轮，按原始标签重算） | 2.1949 | 5.0334 | 82.16% |
| 缩放修正版（第 12 轮） | 2.0424 | 4.9016 | 85.49% |
| 训练商品平均周期基线 | 2.8109 | 5.7894 | 70.86% |

修正版 MAE 比原版降低 6.9%，比商品周期基线降低 27.3%。中位绝对误差 0.9969 天，P90 为 3.7511 天；每个 Lot 等权 MAE 为 1.9957 天，每个 Lot 最晚窗口 MAE 为 1.8885 天。本次验证没有负预测，但网络输出并未限制非负。

长剩余时间表现仍弱：标签超过 14 天的 150 个窗口 MAE 为 16.5380 天，均偏早预测；标签不超过 3 天的窗口 MAE 为 0.6671 天，较原版 0.5402 天有所上升，因此改善并非覆盖所有时间段。

上述主指标对 3,397 个窗口等权，来自 1,750 个 Lot；同一 Lot 的窗口数可不同。验证集参与早停与选模，且仍有 481 个窗口的参考时间不晚于训练截止，不属于独立测试或严格时间回测。完工标签沿用用户认可的企业 root Lot 最后观测规则，并非经独立完工事件核验的真实完工时间。本次仍是原窗口定义及可含小数天的点预测实验；固定日末预测、整数日标签与独立 NLL 分布预测尚未实现，不能将本次指标直接当作后续研究版的结果。

本机产物：

- [修正实验报告](../outputs/dk/full_runs/20261015_195345_scaler_fix/experiment_report.html)
- [评估指标 JSON](../outputs/dk/full_runs/20261015_195345_scaler_fix/evaluation_summary.json)
- [全部验证窗口预测](../outputs/dk/full_runs/20261015_195345_scaler_fix/validation_predictions.csv)
- [每个验证 Lot 的最晚窗口预测](../outputs/dk/full_runs/20261015_195345_scaler_fix/validation_latest_per_lot.csv)
- [最佳模型目录](../outputs/dk/checkpoints/20261015_195531/)

本次报告已通过以下命令生成；仅在需要复现时运行。脚本按 `baseline_run` 读取首次实验产物比较，不重建或覆盖原版报告：

```powershell
& .\.venv\Scripts\python.exe scripts/evaluation/report_dk_run.py --run-dir outputs/dk/full_runs/20261015_195345_scaler_fix --checkpoint outputs/dk/checkpoints/20261015_195531 --experiment outputs/dk/experiments/20261015_195531 --threads 4
```

### 使用模型预测

使用本次保存的 checkpoint 运行预测：

```powershell
& .\.venv\Scripts\python.exe run.py predict --dataset dk --checkpoint "outputs/dk/checkpoints/20261015_195531" --at "2025-02-10 00:00:00" --lot "实际LotId"
```

该入口当前输出原模型定义的剩余天数。预计完工时刻按“该 Lot 输入窗口最后一条快照时间 + 预测剩余天数”换算，不能直接加到请求截止时间上。报告中的预测为历史验证窗口，不能视为当前实时在制品清单。模型、日志、CSV 和虚拟环境均由 `.gitignore` 排除；代码与说明的后续提交按用户指令执行。

CPU 构建参考：[PyTorch 2.9.0 官方安装说明](https://pytorch.org/get-started/previous-versions/#v290)。
