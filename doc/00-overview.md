# WIP Analysis Backend Overview

This repository exposes WIP analysis through FastAPI routes in `src/wip_api/routes`
and shared calculation code in `src/wip_analysis`. Prediction model training and
inference support lives in `src/wip_pred`.

## Runtime

- API app: `src/wip_api/main.py`
- Route prefix: `/api/v1`
- Startup: initializes the PostgreSQL connection pool and logging.
- Shutdown: drains async writes, closes the connection pool, clears CUDA memory when
  available, and runs local cleanup hooks.
- Request middleware: records per-route in-flight counts, exposed by `GET /metrics`.

## Public Routes

| Route | Method | Source | Purpose |
| --- | --- | --- | --- |
| `/api/v1/health` | POST | `routes/health.py` | Liveness check. |
| `/api/v1/metrics` | GET | `routes/metrics.py` | Process-local in-flight request counts. |
| `/api/v1/lots/{lot_id}` | POST | `routes/hwm_lot.py` | Lot-level historical progress across snapshots. |
| `/api/v1/lots/{lot_id}/progress-z.html` | POST | `routes/hwm_lot.py` | Plotly HTML heatmap of lot progress duration z-scores. |
| `/api/v1/products/{product_id}` | POST | `routes/hwm_prod.py` | Product-level stage lot counts over calendar time and snapshot cuts. |
| `/api/v1/products/{product_id}/load_ratio` | POST | `routes/stage_load.py` | Product-scoped rolling wafer load ratios. |
| `/api/v1/stage_load_ratio` | POST | `routes/stage_load.py` | Body-scoped rolling wafer load ratios; `product_id` is optional. |
| `/api/v1/stage_dur` | POST | `routes/stage_dur.py` | Stage cycle-time rows with wafer count enrichment. |
| `/api/v1/stage_cap` | POST | `routes/stage_cap.py` | Stage entry lot and wafer counts by calendar date. |
| `/api/v1/products/{product_id}/date_out` | POST | `routes/pid_date_out.py` | Actual and predicted output counts for target dates. |
| `/api/v1/paths_stats` | POST | `routes/paths_stats.py` | Primary/secondary path clusters with lot, wafer, and start-time stats. |
| `/api/v1/products/{product_id}/paths.html` | POST | `routes/paths_stats.py` | Interactive HTML network graph for clustered product paths. |
| `/api/v1/products/{product_id}/super-hot-lot.html` | POST | `routes/super_hot_lot.py` | Interactive cumulative-stage-time plot for lots under a product. |
| `/api/v1/bottleneck_detection` | POST | `routes/bottleneck.py` | Bottleneck stage detection by one of three methods. |

## Core Tables

Schemas are rendered from `sql/schemas/*.sql.j2`.

- `product.sql.j2`: product primary path cache. Important columns are
  `product_id`, `path_start_time`, `path_end_time`, `primary_path`,
  `primary_path_durations`, `min_completion_time`, and `calculation_time`.
- `path.sql.j2`: clustered path candidate cache used by path stats and lot-level
  candidate matching.
- `lot_progress.sql.j2`: per-lot, per-snapshot stage progress cache. Many routes
  read from or ensure this table is populated.
- `raw_snapshot.sql.j2` and `clean_snapshot.sql.j2`: source snapshot data.
- `lot_completion_predictions.sql.j2`: model-backed predicted completion rows used
  by date-out.

## Shared Calculation Rules

- Product primary paths come from `get_pid_primary_path()`. The function first
  reads `product.sql.j2` rows through `get_pid_primary_paths.sql.j2`, selects the
  widest matching date window, checks coverage when needed, and only computes a new
  path when the DB cache is missing or does not cover required snapshots.
- Lot-level per-snapshot progress may use a matched path candidate from the path
  candidate table. This is per-row behavior. The top-level lot response
  `stage_order` is product primary path order from `product.sql.j2`, bounded only
  by `snapshot_end` so it does not use future data.
- Product and stage routes usually use `snapshot_start`/`snapshot_end` as inclusive
  filters. Routes reject windows where start is greater than or equal to end.
- Product-scoped routes that need `lot_progress` call readiness helpers that may
  trigger background population and block until data is ready or a timeout occurs.
- `refresh=true` on HTML/lot endpoints clears in-process LRU caches before the
  calculation. It does not truncate DB cache tables.

## Documentation Map

| File | Topic |
| --- | --- |
| `01-lot-level-hwm.md` | Lot progress JSON and z-score plot route. |
| `02-product-level-hwm.md` | Product HWM lot-count matrix. |
| `03-stage-load.md` | Rolling wafer load ratios. |
| `04-stage-duration.md` | Stage duration rows and z-score source data. |
| `05-stage-capacity.md` | Stage entry capacity rows. |
| `06-lot-quickest-finish.md` | Internal earliest-finish estimate helper and script. |
| `07-pid-date-out.md` | Product output counts by target date. |
| `08-plot-paths.md` | Path cluster stats and HTML graph route. |
| `09-super-hot-lot.md` | Super-hot lot cumulative plot route. |
| `10-primary-path.md` | Primary path cache and optimization flow. |
| `11-model-training.md` | Current model training and inference pipeline. |
| `12-bottleneck-candidates.md` | Bottleneck detection route and methods. |
| `13-time-to-end-variance.md` | Legacy/internal time-to-end variance visualization. |
| `14-lot-progress-data-readiness.md` | Lot-progress and date-out readiness helpers. |
