# PTO 完整基线版

版本标签：[`pto-baseline-20261021`](https://github.com/WjtTHU/DFL/tree/pto-baseline-20261021)。本版保存每日点预测与 NLL 分布预测、提前 28 天订单规则、3×3 APP 数据构造、CPLEX 滚动执行，以及历史基线、全周期 Oracle 和已有快照 Lot 完美预测诊断。

当前实验采用 10 周训练、1–28 天加尾部的 NLL，APP 为 7 天一期、N=4、测试 T=7。发布前全部 63 项测试通过。使用说明见[每日预测](doc/daily_prediction.md)、[APP 数据](doc/app_dataset.md)和[滚动实验](doc/app_pto_experiment.md)；本机 CPLEX 路径由 `config/app_solver.yaml` 配置。

每个提交／标签都包含当时的 `README.md`，可在 GitHub 切换标签查看。本版只提交代码、配置、测试和文档；数据、模型及 `outputs/` 内实验结果继续保留本地，报告链接仅在相应本地文件存在时可用。下方实验日志中的“未提交”描述当时状态，相关代码现统一纳入本标签；历史实验没有另外补造独立提交。

## 本地 CSV 入口

本地使用方法和路径索引见 [本地路径说明](doc/local_paths.txt)。统一入口为 `python run.py check`，数据路径配置位于 `config/local_paths.json`。

版本的主要变动与回溯提交号见 [版本迭代记录](doc/version_history.md)。“原版”固定指首次本地 CSV 基线 `abfce8f`。

代码仓库：https://github.com/WjtTHU/DFL 。仓库根目录为 `local_wip`。代码、配置、依赖锁定文件和文档纳入版本管理；CSV、原始压缩包、模型权重和运行输出保留在仓库外或由 `.gitignore` 排除。

日常修改先保留在工作区，收到提交指令后再创建 commit 并推送到 GitHub。重要版本可增加 tag，注明数据版本及验证情况。历史代码可通过 commit 或 tag 检出到独立分支；恢复已发布版本时优先使用 revert，以保留完整历史。未提交的修改，以及未纳入 Git 的数据和模型，不在代码版本历史中。

2026-10-21 已完成 **已有快照 Lot 完美预测诊断**：仅将 904 条有快照订单视图替换为企业口径完工标签，其余 3,775 条保留原承诺日期，N=4、T=7 和全部 APP 参数不变。成本 **714,485.97**、Oracle regret **14.79%**，相比点预测成本增加 472.49（0.066%）：采购费多 8,649，积压费少 8,176.51。这次仅消除已有快照批次的预测误差没有缩小总成本差距，不能将剩余 regret 全部归因于预测。34 项行为检查与独立回放、哈希核验通过，旧模型和结果不变，未提交 Git。见[五方诊断报告](outputs/app/pto_runs/20261021_perfect_observed_v1/experiment_report.html)、[配置](config/app_perfect_observed.yaml)、[说明](doc/app_pto_experiment.md)。

2026-10-21 已完成 **NLL 期望到货 PTO 与点预测、历史基线、Oracle 比较**：复用 NLL 最佳第 4 轮，将 1–28 天概率汇总为周概率并乘固定数量，尾部保留在窗外，沿用确定性 APP 和整批实际收货。相同 7 期测试下，NLL／点预测／历史基线／Oracle 成本为 **747,176.12／714,013.47／743,450.47／622,418.41**。NLL 比点预测高 4.64%，比历史基线高 0.50%，相对 Oracle regret 20.04%；采购费增加超过积压费节约，未据测试调参。904 条有快照周预测的 NLL 为 1.3509，90% 类别区间覆盖率 90.49%，不与全部每日测试窗口直接比较。28 项行为检查、7 次新增滚动求解和独立回放通过，旧轨迹与模型保留，未重训、未提交 Git。见[比较报告](outputs/app/pto_runs/20261021_nll_expected_v1/experiment_report.html)、[配置](config/app_nll_expected_28d.yaml)、[说明](doc/app_pto_experiment.md)。这是期望供给接入，尚非场景随机规划。

2026-10-20 **10 周训练＋28 天 NLL 版已完成**：18:19–20:24 完成串行训练、测试评价和[实验报告](outputs/dk/daily/full_runs/20261020_train10w_h28/experiment_report.html)，约 124.8 分钟。点预测第 46 轮早停、最佳第 16 轮，测试 MAE **2.6374 天**、RMSE **4.8612 天**、Lot 等权 MAE **1.8830 天**，比同窗口训练商品周期基线 3.4243 天低 23.0%。NLL 第 34 轮早停、最佳第 4 轮，测试分箱 NLL **2.2076**，低于商品周期条件基线 2.8682 和固定频率基线 2.7232；标称 90% 类别区间覆盖率为 **81.3%**，开放上界比例 **0.37%**。长周期仍低估：15–28 天及 >28 天窗口的点预测 MAE 分别为 10.1923／27.1495 天，尾部训练只有 58 个窗口，验证无尾部事件。独立协议 [daily_dk_10w_h28.yaml](config/daily_dk_10w_h28.yaml) 使用 2024-12-10／2025-01-07 切分，测试至 2025-02-24，NLL 为 1–28 天加尾部；新旧划分与分箱不同，不直接比较历史指标的改善比例。两模式窗口、最佳模型、指标、CSV／源码及旧产物哈希核验通过，保留企业快照推定完工、全历史质量筛选、完成批次选择和单一种子的限制。

[提前 28 天订单规则](config/app_order_information_28d.yaml)及 3,733 个 Lot 的订单元数据已保存，承诺周期只由训练标签拟合，未加入 Transformer 输入。395 个历史文件和修改前源码已压缩至[旧结果归档](outputs/archives/dk_app_before_20261020.zip)，CRC 和逐文件哈希验证通过；只清理已归档的重复 APP ZIP，保留主要历史模型、报告及新运行完整目录。归档与清理记录见[清单](outputs/archives/dk_app_before_20261020.json)。尚未提交或推送 Git。

2026-10-20 随后完成 **提前 28 天订单版点 PTO 滚动 APP**：3 材料、3 产品，测试 2025-01-07 至 02-24，7 天一期、N=4、T=7，最后规划覆盖至 03-17。复用最佳第 16 轮点模型，补跑 904 条有快照周预测，另有 3,775 条未观测订单共用合成承诺日期；未重新训练。PTO／历史周期基线／全 T Oracle 成本分别为 **714,013.47／743,450.47／622,418.41** 实验成本单位，PTO 比基线低 **3.96%**，相对 Oracle regret 为 **14.72%**。采购成本减少 35,019，积压成本增加 5,582，两策略最终积压均清零。15 次 CPLEX 求解均最优，21 项行为检查、独立状态和成本复核、两次生成的 65 文件逐字节复现及保护文件哈希核验通过。见[新报告](outputs/app/pto_runs/20261020_point_advance28_v2/experiment_report.html)、[实验说明](doc/app_pto_experiment.md)、[数据配置](config/app_dk_10w_advance28.yaml)。这是单条轨迹的点预测实验，尚未运行 NLL 随机规划；旧结果保留，未提交 Git。

初始提交 `abfce8f` 基于企业 v0.9.5 发布包及补充的 CSV 入口和配置。首次 DK 全量基线在第 38 轮早停，原日志选出的最佳模型为第 8 轮；按原始天数标签计算的验证 MAE 为 2.1949 天。原实现的验证标签缩放与训练范围不一致，日志中的 1.3658 不能直接解释为原始天数误差。原基线模型和[历史报告](outputs/dk/full_runs/20261015_174056/experiment_report.html)保留。

2026-10-15 已完成标签缩放修正及同配置全量重跑：第 42 轮早停，最佳模型为第 12 轮，耗时约 101.9 分钟。重载核验的验证 MAE 为 **2.0424 天**，RMSE 为 **4.9016 天**；MAE 比原版降低 6.9%，比训练商品平均周期基线降低 27.3%。训练和验证均固定使用 1–62 天的缩放范围，原始验证标签最高 108 天，未裁剪。5 项回归测试通过，两次数据、源配置与验证窗口一致。见[修正实验报告](outputs/dk/full_runs/20261015_195345_scaler_fix/experiment_report.html)及[本地运行说明](doc/local_run.md)。结果来自用于选模的验证集，尚非独立测试或严格时间回测；完工标签沿用企业快照规则。缩放修正版已提交并推送为 `114e023`，回溯标签为 `scaler-fix-20261015`。运行输出仅保留本地。

当前工作区新增 DK 每日日末预测入口：整数日标签、保留日内观测、按标签可用时间切分训练/验证/测试，支持分别训练点预测和 NLL 分布预测。见[每日预测使用说明](doc/daily_prediction.md)。19 项测试及真实数据小规模流程验证通过；2026-10-16 12:17–15:29 已完成两种模式的全量训练、测试评价和[实验报告](outputs/dk/daily/full_runs/20261016_1218/experiment_report.html)。点预测第 37 轮早停、选用第 7 轮，测试 MAE **1.4901 天**、RMSE **2.0906 天**、Lot 等权 MAE **1.1468 天**，MAE 比同测试窗口的训练商品周期基线低 24.1%。NLL 模式第 35 轮早停、选用第 5 轮，测试分箱 NLL **2.3838**（同商品周期＋已过天数条件基线 2.5625，固定训练频率弱基线 2.4265）。模型中央 90% 类别区间覆盖率仅 **74.5%**，条件基线为 89.2% 且区间更宽，概率校准仍需改进。条件基线在首轮报告后补充，固定平滑规则仅拟合训练数据，新增 5 项行为测试通过，原模型与预测未重跑。测试为 607 个 Lot 的 3,703 个日末窗口，未参与选模；保留全历史质量筛选及已完成可标注样本，标签仍为企业快照规则推定，不能视为全部实时在制品的上线回测。测试无超过 14 天的实际尾部事件，长尾能力尚未充分验证。新代码未提交，旧版 2.0424 天使用另一套口径，不直接比较。凌晨中断记录已按用户要求删除。

2026-10-17 已按后续 APP 实验需要调整时间划分：训练截至 2024-12-30，验证为 2024-12-31～2025-01-27，测试为 2025-01-28～2025-03-17。19:07–21:09 完成点预测与 NLL 串行重跑、测试评价和[实验报告](outputs/dk/daily/full_runs/20261017_1908_pto_split/experiment_report.html)，总计约 121.7 分钟。点预测第 33 轮早停、最佳第 3 轮，测试 MAE **2.0937 天**、RMSE **3.5374 天**、Lot 等权 MAE **1.6670 天**，相对同窗口商品周期基线 MAE 2.7686 天降低 24.4%。NLL 第 32 轮早停、最佳第 2 轮，测试分箱 NLL **2.0289**（商品周期条件基线 2.5401，固定频率基线 2.4881）；中央 90% 类别区间覆盖率 **90.3%**，但仅 71.5% 的区间有有限上界，不能据此认定概率完全校准。测试共 18,618 个窗口、2,485 个 Lot；其中 642 个窗口的剩余时间超过 14 天，点预测 MAE 为 10.4505 天，长尾低估仍明显。41 个训练源码文件和原 CSV 未变，24 项检查通过，旧模型及报告哈希核验一致。新旧时间划分不同，不直接比较模型改进幅度；保留完成批次选择、全历史质量筛选、企业快照推定完工及单一随机种子的限制。尚未构造 APP 数据或实现下游优化，本次未提交 Git。

2026-10-19 已生成独立的 **3 材料、3 产品 APP 小规模数据集**，采用 7 天一期、N=4，训练／验证／测试分别 13／4／7 期，需求仅以训练参考周校准。保留 3,733 个 Lot（含 29 个末尾未确认 Lot），快照、按时点订单、需求、BOM、产能、成本、初始状态与事后到货分别保存。见 [APP 数据说明](doc/app_dataset.md)、[数据目录](outputs/app/dk_3x3_v1_20261019/)及[检查报告](outputs/app/dk_3x3_v1_20261019/dataset_report.html)。测试材料 A 在第二货源满额下仍有约 6.63 万 die 的总量缺口，保留这一结果，不据测试数据反调需求。原数据及预测模型不变，未提交 Git。

2026-10-19 随后完成首版 **点 PTO 与 CPLEX 滚动 APP 实验**：复用原模型补跑 660 条周预测（647 个 Lot），按 N=4 规划、T=7 执行，与历史商品周期基线共享同一数据和参数。14 次求解均为最优，行为测试和独立状态／成本复核通过。两者总成本均为 **1,673,919.92** 实验成本单位，当期生产与采购完全相同；周预测 MAE 分别为 0.8194／1.3480 天，但本次没有体现成本改善。36 条预测的到货周不同，部分未来计划也不同，详见 [模型与实验说明](doc/app_pto_experiment.md) 和 [实验报告](outputs/app/pto_runs/20261019_point_v1/experiment_report.html)。仍保留末期未知到货和未来新订单不可预告的限制，不将单条小规模轨迹视为普遍结论。未重新训练、未改冻结数据、未提交 Git。

`config/data_sources.json` 记录本版三份 CSV 的大小和 SHA-256；实际文件需另外提供，并在 `config/local_paths.json` 中配置位置。

CSV 入口不需要数据库。若使用原包的 PostgreSQL/API 功能，请将 `config/postgres.example.yaml` 复制为 `config/postgres.yaml` 并填写本地连接信息；含连接口令的实际配置不纳入 Git。

以下保留原发布包的说明，包含数据库和 API 服务用法。

# WIP Prediction & HWM Generation

## Getting Started

This project uses Nix flakes for reproducible development environments.

**Supported platforms:**
- x86_64-linux
- aarch64-linux

**Tested on:**
```text
OS:      Arch Linux x86_64
Kernel:  Linux 6.17.1-arch1-1
CPU:     AMD Ryzen 5 5600 (12) @ 3.63 GHz
GPU:     NVIDIA GeForce RTX 3060 Lite Hash Rate [Discrete]
Mem:     32 GiB
```

We recommend using GNU/Linux for hosting. Behaviors on Windows/MacOS haven't been tested thoroughly.

## Documentation

Comprehensive documentation for all WIP analysis functionality is available in the `./doc/` directory:

- [System Overview](./doc/00-overview.md) - Architecture and core concepts
- [Lot-Level HWM](./doc/01-lot-level-hwm.md) - Individual lot tracking
- [Product-Level HWM](./doc/02-product-level-hwm.md) - Aggregate statistics
- [Stage Load Analysis](./doc/03-stage-load.md) - Throughput metrics
- [Stage Duration](./doc/04-stage-duration.md) - Cycle time analysis
- [Stage Capacity](./doc/05-stage-capacity.md) - Entry rate statistics

## Setup

Do the following under root dir of this repo:

### Prerequisites

- **Nix** with flakes enabled ([installation guide](https://nixos.org/download.html))
- **(Optional)** Docker for containerized deployment

### Prepare the Environment

#### 1️⃣ Enter Development Environment

```bash
# Enter the default dev shell (includes uv, prek, ruff, CUDA tools)
nix develop

# Or enter clean shell (production-like, no dev tools)
nix develop .#clean
```

The development shell provides:
- Python 3.12
- All project dependencies
- Development tools (uv, prek, ruff)
- CUDA toolkit and libraries

#### 3️⃣ Build Docker Images (Optional)

```bash
# Build API service image
nix build .#dockerApi
docker load < result

# Build training image
nix build .#dockerTrain
docker load < result
```

### Database Configuration

#### 1️⃣ Prepare Configuration File

Create your **`.yaml` configuration file** under `config/environments` (refer to `config/environments/hb.yaml` as a template), and create a symbolic link to it under `config`:

```bash
cd config
ln -sf environments/hb.yaml sql_base.yaml
```

Having the configuration file under `config` named `sql_base.yaml` saves repetition when calling the SQL template renderer.

#### 2️⃣ Populate Database Tables

This project depends on several database tables:

**If you already have a snapshot table:**
- Ensure your snapshot table has columns for: `product_id`, `lot_id`, `parent_lot_id`, `stage`, `lot_status`, `snapshot_time`
- Fill your table's name in `database.tables.raw_table`

**If you don't have a database/table yet:**
- Decide a name for the raw table and fill it in your configuration: `database.tables.raw_table`
- Decide column names for this raw_table
- Populate the table with: `nix develop -c ./scripts/init/init_schemas.py --config config/sql_base.yaml --schema raw_snapshot.sql.j2`

**Configure other tables:**
- `database.tables.snapshot_table`: Filtered subset of `raw_table` (removes incomplete/noise data)
- `database.tables.product_table`: Product information
- `database.tables.lot_progress_table`: Expanded lot manufacturing process

Ensure column names in the `Universal` and `Snapshot Table` sections align with your actual table columns.

**Populate remaining tables:**

```bash
# Enter nix shell first
nix develop

# Then run initialization
./scripts/init/init_schemas.py --config config/sql_base.yaml
```

**If your raw_table needs data:**
Use the script `./scripts/init/csv_to_db.py` (adjust according to your needs), then refresh the materialized view:

```sql
REFRESH MATERIALIZED VIEW view_name;
```

After setup, you should have:
1. `raw_table`
2. `product_table`
3. `lot_progress_table`
4. `snapshot_table` (materialized view)

Tables 2 & 3 will be empty initially and populated in the next step.

### Populate Data into Tables

To support lot-level and product/fab-level analysis, populate `lot_progress_table` and `product_table` using scripts in `scripts/populate`.

#### 1️⃣ Primary Path and Stage Durations

Use either `scripts/populate/product_summary.py` or `scripts/populate/product_summary_async.py`. The former runs directly, while the latter requires the backend API to be running (see [Start the Services](#start-the-services)).

For `product_summary_async.py`, uncomment these lines around row 21:

```python
        _ = await run_in_threadpool(get_pid_primary_durs, product_id)
        _ = await run_in_threadpool(get_pid_quickest_cycle, product_id)
```

And add corresponding imports:

```python
from wip_analysis.hwm.product_level import (
    get_pid_primary_durs,   # <-- added for populating
    get_pid_primary_path,
    get_pid_quickest_cycle, # <-- added for populating
    get_pid_stage_stats,
)
```

Run the scripts (adjust arguments according to your hardware and demand):

```bash
# Enter nix shell
nix develop

# Start API (in one terminal)
./scripts/api/start_api.py --log-level debug

# Run population script (in another terminal)
./scripts/populate/product_summary_async.py http://localhost:8724/api/v1 2 0
```

#### 2️⃣ Lot-Level HWM

```bash
# In nix shell
./scripts/populate/lot_progress_async.py
```

### Start the Services

#### Backend API

```bash
# In nix shell
./scripts/api/start_api.py --log-level debug
# Log files can be found in ./logs/backend_api
```

The API will be available at `http://localhost:8724`

#### Frontend (Demo)

```bash
cd frontend
pnpm run dev
```

## Docker Deployment

### Running API Service

```bash
# Load the image (if built with nix)
docker load < result

# Or pull from registry (if published)
# docker pull your-registry/wip-analysis-api:latest

# Run the container
docker run -d \
  --name wip-api \
  -p 8724:8724 \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  wip-analysis-api:latest
```

### Running Training Service

```bash
# Load the image
docker load < result

# Run training
docker run -d \
  --name wip-train \
  --gpus all \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/experiments:/app/experiments \
  wip-analysis-train:latest
```

### Environment Variables

Runtime settings can be provided through process environment variables, `.env`,
`.env.local`, or a custom file path in `WIP_ANALYSIS_ENV_FILE`. Process
environment variables always take precedence over dotenv files. See
`.env.example` for the full set of supported tuning variables.

| Variable | Description | Default |
|----------|-------------|---------|
| `WIP_ANALYSIS_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `WIP_ANALYSIS_PORT` | API listen port | 8724 |
| `WIP_ANALYSIS_ENV_FILE` | Optional dotenv file path or `:`-separated file list | unset |
| `WIP_ANALYSIS_PRESET_PRIMARY_PATHS` | JSON object of product IDs to authoritative stage arrays; supports a `*` fallback and skips clustering/path optimization for matches | unset |
| `PYTHONPATH` | Python module search path | /app/src |

### Volume Mounts

| Container Path | Purpose | Recommended Host Path |
|----------------|---------|----------------------|
| `/app/config` | Configuration files | `./config` (read-only) |
| `/app/logs` | Application logs | `./logs` |
| `/app/checkpoints` | Model checkpoints | `./checkpoints` |
| `/app/experiments` | Experiment results | `./experiments` |

## Development

### Available Nix Shells

- **`nix develop`** (default): Full development environment with all tools
- **`nix develop .#clean`**: Production-like environment without dev tools

### Running Scripts

After entering the nix shell, run scripts directly:

```bash
nix develop
./scripts/api/start_api.py
```

Or run without entering the shell:

```bash
nix develop -c ./scripts/api/start_api.py
```

### Code Quality

Development tools are available in the default nix shell:

```bash
# Install Git hooks
prek install

# Run hooks on all files
prek run --all-files

# Format code
ruff format .

# Lint code with fixes
ruff check --fix .

# Type checking (if configured)
# mypy src/
```

## Project Structure

```
.
├── config/              # Configuration files
│   └── environments/    # Environment-specific configs
├── doc/                 # Comprehensive documentation
├── frontend/            # Demo frontend application
├── scripts/             # Utility scripts
│   ├── api/             # API service scripts
│   ├── init/            # Database initialization
│   └── populate/        # Data population scripts
├── sql/                 # SQL templates
├── src/                 # Source code
│   ├── wip_api/         # FastAPI backend
│   ├── wip_analysis/    # Analysis logic
│   └── wip_pred/        # Prediction utils
├── flake.nix            # Nix flake configuration
└── pyproject.toml       # Python project configuration
```

## Troubleshooting

### Nix Flakes Not Enabled

If you get an error about experimental features:

```bash
# Enable flakes in your nix configuration
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
```

### Database Connection Issues

- Verify your `config/sql_base.yaml` has correct database credentials
- Ensure the database server is running and accessible
- Check firewall rules if connecting to remote database

### Port Already in Use

If port 8724 is already in use:

```bash
# Find process using the port
lsof -i :8724

# Kill the process or use a different port
./scripts/api/start_api.py --port 8725
```

