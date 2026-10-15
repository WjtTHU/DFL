# Stage Load Ratio

Stage load ratio normalizes rolling wafer volume by average stage duration.
The result is expressed as a wafer/day load metric for each stage and snapshot
cut.

## Routes

### `POST /api/v1/products/{product_id}/load_ratio`

Path-scoped product route. Request body is optional:

```json
{
  "product_id": "same as path, optional guard",
  "snapshot_start": "2024-01-01T00:00:00",
  "snapshot_end": "2024-01-31T23:59:59"
}
```

If the body `product_id` is present and does not match the path parameter, the
route returns `400`.

### `POST /api/v1/stage_load_ratio`

Body-scoped route. `product_id` is optional; omitted means `"ALL"`.

```json
{
  "product_id": "PID1",
  "snapshot_start": "2024-01-01T00:00:00",
  "snapshot_end": "2024-01-31T23:59:59"
}
```

Source:

- Route: `src/wip_api/routes/stage_load.py`
- Models: `src/wip_api/models/stage_load.py`
- Calculation: `src/wip_analysis/hwm/product_level/stage_load.py`

## Response Shape

```json
{
  "product_id": "PID1",
  "stage_order": ["STAGE_A", "STAGE_B"],
  "load_ratios": [
    {
      "snapshot_cut": "2024-01-03T08:00:00",
      "calendar_times": ["2024-01-01T00:00:00"],
      "stage_metrics": {
        "STAGE_A": {
          "avg_duration_seconds": 86400.0,
          "lot_counts": [2],
          "wafer_sums": [50],
          "rolling_7d": [50.0],
          "rolling_1m": [50.0],
          "rolling_3m": [50.0],
          "rolling_6m": [50.0],
          "rolling_1y": [50.0]
        }
      }
    }
  ],
  "reason": null
}
```

## Key Calculation Flow

`get_pid_stage_load(product_id, snapshot_start, snapshot_end)`:

1. Loads primary path with date bounds derived from the snapshot window.
2. Calls `get_pid_stage_stats()` to get calendar-time arrays of lot counts and
   wafer sums.
3. Loads per-stage duration data with `get_stage_duration()` and calculates
   average duration seconds for each primary-path stage.
4. If product-specific duration data is missing, calls
   `ensure_lot_progress_populated()` for the missing stages.
5. Explodes product HWM arrays into rows:
   `(snapshot_cut, calendar_time, stage_name, lot_count, wafer_sum)`.
6. For each `(snapshot_cut, stage_name)`, computes rolling wafer sums over
   `7d`, `1mo`, `3mo`, `6mo`, and `1y`.
7. Divides rolling wafer sum by `avg_duration_seconds / 86400`.
8. Builds a complete calendar axis per snapshot so all stage arrays align.

Formula:

```text
load_ratio = rolling_wafer_sum / (avg_duration_seconds / 86400)
```

If duration is missing or non-positive, the rolling load value is `null`.

## Validation And Errors

- `snapshot_start >= snapshot_end`: `400`
- Product mismatch on the path route: `400`
- Readiness `ValueError`: `409`
- Readiness timeout: `504`
- Unexpected failure: `500`

Empty calculations return a normal response with empty `load_ratios` and a
`reason`.

## Interpretation

- `lot_counts` and `wafer_sums` are raw stage volume arrays aligned with
  `calendar_times`.
- Rolling arrays are normalized by duration, so stages with longer typical
  duration require more rolling wafer volume to show the same load level.
- This route is not a direct capacity route. Use `/stage_cap` for entry counts
  by stage and date.
