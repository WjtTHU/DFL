# Stage Capacity

Stage capacity reports how many lots and wafers enter stages by calendar date.
It is an entry-volume metric, not a duration-normalized load metric.

## Route

### `POST /api/v1/stage_cap`

Request body is optional:

```json
{
  "stage_name": ["STAGE_A", "STAGE_B"],
  "product_id": "PID1",
  "snapshot_start": "2024-01-01T00:00:00",
  "snapshot_end": "2024-01-31T23:59:59"
}
```

`stage_name` may be:

- `null`: all stages
- string: one stage
- ordered list of strings: return series in the same order

Blank strings are normalized to no filter.

Source:

- Route: `src/wip_api/routes/stage_cap.py`
- Models: `src/wip_api/models/stage_cap.py`
- Calculation: `src/wip_analysis/hwm/stage/capacity.py`
- SQL: `sql/templates/get_pid_stages_transitions.sql.j2`

## Response Shape

```json
{
  "product_id": "PID1",
  "stage_name": ["STAGE_A", "STAGE_B"],
  "capacity_data": [
    {
      "stage_name": "STAGE_A",
      "wafer_data": [
        {
          "calendar_date": "2024-01-03",
          "stage_name": "STAGE_A",
          "entering_lot_count": 2,
          "entering_wafer_count": 50
        }
      ]
    }
  ],
  "reason": null
}
```

## Key Calculation Flow

`get_stage_capacity(stage_name, product_id, snapshot_start, snapshot_end)`:

1. Normalize blank product/stage filters.
2. For product-specific requests, call `ensure_lot_progress_populated()`.
3. Render `get_pid_stages_transitions.sql.j2`.
4. Fetch rows with:
   - `calendar_date`
   - `stage_name`
   - `entering_lot_count`
   - `entering_wafer_count`
5. The route groups rows by stage.
6. If `stage_name` was a list, preserve the request order and include empty
   series for requested stages that have no rows.
7. If all stages were requested, sort stage series alphabetically.

## Validation And Errors

- `snapshot_start >= snapshot_end`: `400`
- Product readiness `ValueError`: `409`
- Readiness timeout: `504`
- Unexpected failure: `500`

If no rows match the filters, the route returns an empty `capacity_data` list and
a `reason`.

## Interpretation

- `entering_lot_count` counts distinct lots entering the stage on that date.
- `entering_wafer_count` sums wafer quantities for those lots using each lot's
  latest snapshot at or before stage entry time.
- A low entry count may indicate an upstream constraint; it does not by itself
  prove the queried stage is the bottleneck.

Related routes:

- `/stage_dur`: cycle time per stage.
- `/products/{product_id}/load_ratio` and `/stage_load_ratio`: rolling wafer load
  normalized by duration.
- `/products/{product_id}`: WIP lot-count arrays aligned to primary path.
