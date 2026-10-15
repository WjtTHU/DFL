# 适配新数据操作指导指南

适用于将非均匀、细粒度（快照间隔可能短于1天）的数据接入本预测系统的场景。

---

## 前提假设

- 新数据为 lot 级别的时间序列快照，每条记录包含时间戳、lot 标识、根批次 lot 标识、产品标识、工序/阶段等信息
- 快照间隔不固定（非均匀），粒度可能为分钟/小时级别
- 需要灵活配置输入特征字段

---

## 步骤总览

| # | 操作 | 涉及文件 |
|---|------|---------|
| 1 | 数据导入与字段映射 | `config/sql_base.yaml`, SQL或CSV |
| 2 | 关闭日历对齐模式 | `core_template.yaml` → `use_given_emb: false` |
| 3 | 配置时间粒度感知 | `core_template.yaml` → `temporal_feat_cols` |
| 4 | 配置模型特征字段 | `core_template.yaml` → `numeric_feat_cols` / `categorical_feat_cols` |
| 5 | 调整滑动窗口参数 | `core_template.yaml` → `window_days` / `step_days` |
| 6 | 调整时间分箱策略 | `normalize_df()` 调用处 → `do_time_binning` / `time_binning_freq` |
| 7 | 处理空值与数据质量 | 数据预处理脚本 |
| 8 | 超参数调优 | `scripts/training/tuning.py` |

---

## 详细步骤

### 步骤 1：数据导入与字段映射

1. 将 CSV 文件导入 PostgreSQL 数据库
2. 在 `config/sql_base.yaml` 的 `columns` 区域配置字段映射（逻辑名 → 实际列名）：

```yaml
columns:
  # 必需字段（不可省略）
  product_id: "your_product_col"
  lot_id: "your_lot_col"
  parent_lot_id: "your_parent_lot_col"
  created_at: "your_timestamp_col"
  lot_status: "your_status_col"
  stage: "your_stage_col"
  current_wafer_qty: "your_qty_col"

  # 新增特征字段（在 Snapshot table 区域加入）
  your_new_feature_1: "actual_db_column_name_1"
  your_new_feature_2: "actual_db_column_name_2"
  ...
```

3. 在 `database.tables` 中配置表名：

```yaml
tables:
  snapshot_table: "your_snapshot_table_name"
```

4. 配置 `config/postgres.yaml` 中的数据库连接信息

---

### 步骤 2：关闭日历对齐模式

在 `core_template.yaml` 中设置：

```yaml
embed:
  use_given_emb: false
```

**原理说明：**
- `use_given_emb: true` 时，系统会将时间戳归一化到"天"级别，每个 lot 每天最多保留一条记录，缺失的天会被填充（产生 `is_given` 列标记插值行）。适合每天一个快照的均匀数据。
- `use_given_emb: false` 时，系统保留原始不规则时间戳，不做日历对齐和缺失填充，模型通过序列长度构造 attention mask。适合非均匀间隔数据。

---

### 步骤 3：配置时间粒度感知

在 `core_template.yaml` 中配置 `temporal_feat_cols`：

```yaml
embed:
  temporal_feat_cols: ["year", "month", "day", "weekday", "hour", "minute"] # 配置模型能感知的时间分量
  temporal_embed_dim: 36  # 能被 len(temporal_feat_cols) 整除即可，每个时间分量的维度为 temporal_embed_dim // len(temporal_feat_cols)
```

**要点：**
- 可选值范围：`year`, `month`, `day`, `weekday`, `hour`, `minute`（会自动从时间戳列中提取相应的时间分量，无需事先构造对应的列）
- 如果数据粒度为分钟级，建议包含 `hour` 和 `minute`；如果为小时级，至少包含 `hour`
- `temporal_embed_dim` 必须满足 `temporal_embed_dim % len(temporal_feat_cols) == 0`

---

### 步骤 4：配置模型特征字段

#### 数值型特征

```yaml
embed:
  numeric_feat_cols: ["your_numeric_feat_1", "your_numeric_feat_2", ...]
```

#### 类别型特征

```yaml
embed:
  categorical_feat_cols:
    - name: "product_id"
      encoder: "embed"
      dim: 16
    - name: "stage"
      encoder: "embed"
      dim: 64
    - name: "lot_status"
      encoder: "embed"
      dim: 2
    - name: "your_categorical_feat_1"
      encoder: "embed"   # 或 "onehot"
      dim: 8             # 仅 encoder=="embed" 时生效
    - name: "your_categorical_feat_2"
      encoder: "embed"   # 或 "onehot"
      dim: 8             # 仅 encoder=="embed" 时生效
    ...
```

**要点：**
- 字段名必须使用 `sql_base.yaml` 中 `columns` 映射的 **key**（逻辑名），而不是 value（实际列名）
- `dim` 可不指定，系统默认为 `min(50, round(num_classes^0.5))`
- 类别数较少（如 lot_status 仅几种取值）时建议 `dim` 设小（2~4）
- 类别数较多（如 stage 可能上百种）时建议 `dim` 设大（16~64）
- `encoder: "onehot"` 不需要 `dim` 参数，输出维度等于类别数

---

### 步骤 5：调整滑动窗口参数（可选）

```yaml
dataset:
  type: "lotwise"
  params:
    window_days: 15    # 滑动窗口长度（天）
    step_days: 1       # 窗口滑动步长（天）
```

**适配建议：**
- `window_days` 控制输入序列覆盖的时间跨度。对于快照间隔很短的数据，同样的 `window_days` 会产生更长的序列——需要权衡序列长度与 GPU 显存。
- 如果 lot 生命周期较短（如几天~十几天），应相应缩短 `window_days`。
- `step_days` 控制训练样本密度（通过滑窗构造训练样本时的滑动步长），取值越小，训练样本数越多。
- `window_days` 和 `step_days` 取值可以为小数。

---

### 步骤 6：调整时间分箱策略（可选）

`normalize_df()` 中有两个与时间分箱相关的参数：

```python
normalize_df(
    ...,
    do_time_binning=True,          # 是否启用时间分箱
    time_binning_freq="10min",     # 分箱频率
    ...
)
```

**适配建议：**
- 时间分箱的作用是合并同一 lot 在极短时间内的重复快照（去噪）
- 对于分钟级数据：建议 `do_time_binning=True`, `time_binning_freq="10min"` 或 `"5min"`
- 对于小时级数据：建议 `do_time_binning=True`, `time_binning_freq="1h"`
- 如果原始数据已经是规整的（无重复/近似重复记录），可以设置 `do_time_binning=False`
- 分箱后，同一 lot 同一时间点的多条记录会只保留最后一条

> 注意：此参数在训练脚本的 `normalize_df()` 调用处设置，不在 YAML 配置文件中。

---

### 步骤 7：数据质量要求

`normalize_df()` 对输入数据有严格的非空要求：

> 传入 `normalize_df()` 的 DataFrame 中，所有列都不允许有空值（NaN）。如果有空值，程序会抛出 `ValueError`。

**处理建议：**
- 在调用 `normalize_df()` 之前，对可能含空值的列做预处理：
  - 填充默认值（推荐）：`df['col'] = df['col'].fillna('NULL')` 或 `df['col'].fillna(0)`
  - 删除含空行：`df = df.dropna(subset=['col1', 'col2'])`

---

## 配置检查清单

完成所有配置后，请对照以下清单确认：

- [ ] 数据已导入数据库或 CSV 文件已准备好
- [ ] `sql_base.yaml` 中必需的 7 个字段（product_id, lot_id, parent_lot_id, created_at, lot_status, stage, current_wafer_qty）均已正确映射
- [ ] 新增特征字段已在 `sql_base.yaml` 的 `columns` 区域添加
- [ ] `use_given_emb` 设为 `false`
- [ ] `temporal_feat_cols` 已包含需要的时间粒度（hour/minute）
- [ ] `temporal_embed_dim % len(temporal_feat_cols) == 0`
- [ ] 数值特征列在 `numeric_feat_cols` 中配置，类别特征在 `categorical_feat_cols` 中配置
- [ ] 特征名使用的是 `sql_base.yaml` 中的逻辑名（key）
- [ ] `window_days` 和 `step_days` 已根据数据特点调整
- [ ] 时间分箱策略（`do_time_binning` / `time_binning_freq`）已在训练脚本中确认
- [ ] 输入数据无空值，或已在预处理中处理
- [ ] `data.dates.train_end` 和 `data.dates.val_end` 已设置为合理的时间切分点

---

## 完整配置示例（以 DK 数据为参考）

```yaml
# core_template.yaml
data:
  dates:
    train_end: "2025-01-31"
    val_end: "2025-03-31"

dataset:
  type: "lotwise"
  params:
    window_days: 15
    step_days: 1

embed:
  numeric_feat_cols: ["current_wafer_qty"]
  categorical_feat_cols:
    - name: "product_id"
      encoder: "embed"
      dim: 16
    - name: "stage"
      encoder: "embed"
      dim: 2
    - name: "lot_status"
      encoder: "embed"
      dim: 2
  temporal_feat_cols: ["year", "month", "day", "weekday", "hour", "minute"]
  temporal_embed_dim: 36
  position_col: "created_at"
  pos_enc: null
  use_given_emb: false

model:
  type: "tsf"
  params:
    d_phi: 32
    d_summary: 16
    d_model: 64
    n_head: 4
    n_layers: 2
    d_ff: 256
    dropout: 0.2
    agg: "last"
    norm: "LayerNorm"
```

---

## 模型结构选择和超参数调优

换用新数据后，原有的超参数（学习率、batch size、window_days 等）通常需要进行针对性地调整（修改`config/pred_template/core_template.yaml`及`config/dk_pred_template/train_template.yaml`），已提供 `scripts/training/tuning.py` 进行自动调参。（也可以换用其它已经支持的模型）

注意：如果序列的有效长度波动较大，建议使用在`config/pred_template/core_template.yaml`中使用 LayerNorm 替换 BatchNorm。

### 工作原理

调优脚本基于 [Optuna](https://optuna.org/) 框架，采用两阶段策略：

1. **Stage 1（粗搜索）**：在搜索空间中运行 `n_trials` 次实验，每次用不同种子训练 30 个 epoch，通过 MedianPruner 提前终止表现差的 trial，筛选出 Top-K 候选参数组合。
2. **Stage 2（鲁棒性验证）**：对 Top-K 参数组合分别用多个随机种子重复训练，选取均值好且方差小的组合作为最终结果。

### 使用方式

#### 1. 定义搜索空间

编辑 `scripts/training/tuning.py` 末尾的 `search_stage` 函数，指定需要调优的参数及其范围：

```python
def search_stage(trial):
    return {
        # 数据集参数（需配合 rebuild_datasets=True）
        "dataset.params.window_days": trial.suggest_int(
            "dataset.params.window_days", 5, 30, step=5
        ),
        "dataset.params.step_days": trial.suggest_categorical(
            "dataset.params.step_days", [1, 2, 3, 5]
        ),
        # 训练参数
        "training.batch_size": trial.suggest_categorical(
            "training.batch_size", [32, 64, 128, 256]
        ),
        "optimizer.params.lr": trial.suggest_float(
            "optimizer.params.lr", 1e-5, 1e-2, log=True
        ),
        "optimizer.params.weight_decay": trial.suggest_float(
            "optimizer.params.weight_decay", 1e-6, 1e-2, log=True
        ),
    }
```

**参数命名规则：** 使用点号分隔的路径表示嵌套字典层级，例如 `"optimizer.params.lr"` 对应配置中的 `optimizer → params → lr`。字典的 key 和 `trial.suggest_*` 的第一个参数（name）必须一致。

#### 2. 配置调优入口

在同文件底部的 `if __name__ == "__main__"` 块中：

```python
# 加载你的配置文件
with open("config/your_pred_template/core_template.yaml") as f:
    core_config = yaml.safe_load(f)
with open("config/your_pred_template/train_template.yaml") as f:
    train_config = yaml.safe_load(f)

pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,   # 前 5 个 trial 不剪枝（积累基线）
    n_warmup_steps=20,    # 每个 trial 前 20 个 epoch 不剪枝
    interval_steps=1,
)

tune(
    core_config,
    train_config,
    df=None,              # None 表示从数据库读取；也可传入已预处理的 DataFrame
    train_end=None,       # None 表示从 core_config 中读取
    val_end=None,
    search_space=search_stage,
    pruner=pruner,
    rebuild_datasets=True,  # 见下方说明
)
```

#### 3. 关于 `rebuild_datasets` 参数

| 值 | 含义 | 适用场景 |
|----|------|---------|
| `True` | 每个 trial 重新构建数据集 | 调优涉及 `window_days`、`step_days` 等数据集参数时（这些参数会影响到数据集构建结果） |
| `False` | 数据集只构建一次，所有 trial 共用 | 仅调优训练参数（lr、batch_size、weight_decay）时，速度更快 |

#### 4. 运行

```bash
.venv/Scripts/python.exe scripts/training/tuning.py
```

#### 5. 应用调优结果

脚本运行结束后会打印最佳参数组合，例如：

```
Best Params:
mean=6.1386, std=0.42
{'training.batch_size': 64, 'optimizer.params.lr': 0.000308, 'optimizer.params.weight_decay': 1.6e-06}
```

将上述值手动更新到对应的 YAML 配置文件中即可。

### 调优策略建议

针对新数据，建议分阶段调优，每个阶段固定其他参数只调一组：

| 阶段 | 调优目标 | `rebuild_datasets` | 推荐搜索范围 |
|------|---------|-------------------|-------------|
| 1 | 数据集参数 | `True` | `window_days`: 5~60, `step_days`: 1~10 |
| 2 | 学习率与正则化 | `False` | `lr`: 1e-5~1e-2, `weight_decay`: 1e-6~1e-2 |
| 3 | batch size | `False` | [32, 64, 128, 256, 512] |
| 4 | 模型结构 | `False` | `d_model`, `n_layers`, `n_head`, `dropout` |

> 如果要调模型结构参数，需要确保在搜索空间中正确引用路径，如 `"model.params.d_model"`、`"model.params.n_layers"` 等。注意调整 `d_model` 时需满足 `d_model % n_head == 0`。

### 自定义调优控制参数

在 `tune()` 内部调用的 `tune_stage()` 中可以调整以下参数：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `n_trials` | Stage 1 总共尝试多少组参数 | 10 |
| `top_k` | 从 Stage 1 中选出多少组进入 Stage 2 | 3 |
| `n_repeats` | Stage 2 中每组参数用多少个随机种子验证 | 5 |
