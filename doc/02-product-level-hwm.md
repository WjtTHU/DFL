# Product-Level HWM

Product-level HWM aggregates cached lot progress into stage-aligned lot-count
vectors. Each vector is aligned to the product primary path.

## Route

### `POST /api/v1/products/{product_id}`

Source:

- Route: `src/wip_api/routes/hwm_prod.py`
- Models: `src/wip_api/models/hwm_prod.py`
- Calculation: `src/wip_analysis/hwm/product_level/stage_lot_cnt.py`
- Primary path: `src/wip_analysis/hwm/product_level/primary_path.py`

There is no request body for this route.

Response:

```json
{
  "product_id": "PID1",
  "stage_order": ["STAGE_A", "STAGE_B"],
  "wip_stats": [
    {
      "calendar_time": "2024-01-05T00:00:00",
      "snapshot_cut": "2024-01-03T08:00:00",
      "stage_lot_cnt": [10, 4]
    }
  ],
  "reason": null
}
```

If no stage statistics are available, the route returns an empty `wip_stats`
list and a reason telling callers to populate lot progress first.

## Key Calculation Flow

`get_pid_stage_stats(product_id, snapshot_start=None, snapshot_end=None,
raise_on_error=True)` does the work:

1. Normalize `None` product to `"ALL"` internally.
2. For product-specific requests, call `ensure_lot_progress_populated()` so the
   latest per-lot progress rows exist.
3. Load `primary_path` with `get_pid_primary_path()`.
4. Query aggregated rows through `get_pid_stage_stats.sql.j2`.
5. For each `(calendar_time, snapshot_cut)` pair, create arrays of length
   `len(primary_path)`.
6. Fill `stage_lot_cnt` and `stage_wafer_sum` by primary-path index.
7. Return rows sorted by calendar time and snapshot cut.

The API response currently exposes `stage_lot_cnt`; `stage_wafer_sum` is used by
stage load calculation but is not returned by this route model.

## Primary Path Matching

`stage_order` comes from `get_pid_primary_path(product_id)`, which reads the
product path table rendered from `sql/schemas/product.sql.j2`. The path table
stores:

- `product_id`
- `path_start_time`
- `path_end_time`
- `primary_path`
- `primary_path_durations`
- `min_completion_time`
- `calculation_time`

When exact bounds are not supplied, the primary-path loader chooses the widest
available cached date range and falls back to a recomputation only if the DB
cache cannot satisfy the query.

## Readiness Dependency

Product HWM relies on `lot_progress` rows. If the product-specific readiness
check finds missing latest snapshots, the helper triggers background
`get_lot_prog_over_snapshots()` jobs and blocks until rows are available. A
timeout propagates as `504`; readiness `ValueError` propagates as `409`.

## Error Behavior

- `ValueError`: `409`
- `TimeoutError`: `504`
- Any other unexpected failure: `500`
