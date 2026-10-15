# 本地 CSV 入口

本地使用方法和路径索引见 [本地路径说明](doc/local_paths.txt)。统一入口为 `python run.py check`，数据路径配置位于 `config/local_paths.json`。

版本的主要变动与回溯提交号见 [版本迭代记录](doc/version_history.md)。“原版”固定指首次本地 CSV 基线 `abfce8f`。

代码仓库：https://github.com/WjtTHU/DFL 。仓库根目录为 `local_wip`。代码、配置、依赖锁定文件和文档纳入版本管理；CSV、原始压缩包、模型权重和运行输出保留在仓库外或由 `.gitignore` 排除。

日常修改先保留在工作区，收到提交指令后再创建 commit 并推送到 GitHub。重要版本可增加 tag，注明数据版本及验证情况。历史代码可通过 commit 或 tag 检出到独立分支；恢复已发布版本时优先使用 revert，以保留完整历史。未提交的修改，以及未纳入 Git 的数据和模型，不在代码版本历史中。

初始提交 `abfce8f` 基于企业 v0.9.5 发布包及补充的 CSV 入口和配置。首次 DK 全量基线在第 38 轮早停，原日志选出的最佳模型为第 8 轮；按原始天数标签计算的验证 MAE 为 2.1949 天。原实现的验证标签缩放与训练范围不一致，日志中的 1.3658 不能直接解释为原始天数误差。原基线模型和[历史报告](outputs/dk/full_runs/20261015_174056/experiment_report.html)保留。

2026-10-15 已完成标签缩放修正及同配置全量重跑：第 42 轮早停，最佳模型为第 12 轮，耗时约 101.9 分钟。重载核验的验证 MAE 为 **2.0424 天**，RMSE 为 **4.9016 天**；MAE 比原版降低 6.9%，比训练商品平均周期基线降低 27.3%。训练和验证均固定使用 1–62 天的缩放范围，原始验证标签最高 108 天，未裁剪。5 项回归测试通过，两次数据、源配置与验证窗口一致。见[修正实验报告](outputs/dk/full_runs/20261015_195345_scaler_fix/experiment_report.html)及[本地运行说明](doc/local_run.md)。结果来自用于选模的验证集，尚非独立测试或严格时间回测；完工标签沿用企业快照规则。固定日末预测、整数日标签和 NLL 分布预测仍待实施。运行输出仅保留本地，修正代码尚未提交。

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

