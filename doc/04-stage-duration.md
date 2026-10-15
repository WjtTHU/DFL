# Stage Duration

Stage duration reports how long lots stayed in each stage. The API reads from
the lot-progress cache when possible and enriches rows with each lot's initial
wafer count.

## Route

### `POST /api/v1/stage_dur`

Request body is optional:

```json
{
  "stage_name": "STAGE_A",
  "product_id": "PID1",
  "snapshot_start": "2024-01-01T00:00:00",
  "snapshot_end": "2024-01-31T23:59:59"
}
```

All fields are optional.

Source:

- Route: `src/wip_api/routes/stage_dur.py`
- Models: `src/wip_api/models/stage_dur.py`
- Calculation: `src/wip_analysis/hwm/stage/duration.py`
- Cache helpers: `src/wip_analysis/hwm/stage/utils/fetch_cached_durs.py`
- Recalculation helpers: `src/wip_analysis/hwm/stage/utils/calculate_upsert.py`

## Response Shape

```json
{
  "product_id": "PID1",
  "stage_name": "STAGE_A",
  "duration_data": [
    {
      "product_id": "PID1",
      "lot_id": "LOT1",
      "stage_name": "STAGE_A",
      "entry_date": "2024-01-01",
      "start_time": "2024-01-01T08:00:00",
      "end_time": "2024-01-02T08:00:00",
      "wafer_count": 25,
      "duration_seconds": 86400.0
    }
  ],
  "reason": null
}
```

## Key Calculation Flow

The route calls `get_stage_duration()` with `lot_id=None`:

1. Resolve effective cut time. If `snapshot_end` is supplied, it is used as the
   cut time; otherwise the default cut is `"3000-12-31"`.
2. Try to read cached duration rows from `lot_progress`.
3. If rows are missing or incomplete, optionally check population readiness, then
   calculate stage end and duration from lot-progress history.
4. Upsert recalculated durations back into the cache table.
5. Normalize columns to a stable Polars schema.
6. Apply inclusive `snapshot_start`/`snapshot_end` filters to `start_time`.
7. If `scored=True`, add `duration_zscore` inside the calculation helper. The
   API response does not expose this field.
8. The route queries initial wafer counts with
   `get_lot_initial_wafers.sql.j2` and adds `wafer_count` to each response row.

## Duration Stats Used Elsewhere

`get_stage_duration_stats()` returns per-stage aggregate rows:

- `stage_name`
- `avg_duration_seconds`
- `std_duration_seconds`
- `sample_count`

Lot HWM uses these stats to compute `duration_z`; stage load uses duration rows
to compute per-stage average duration.

## Validation And Errors

- `snapshot_start >= snapshot_end`: `400`
- Readiness `ValueError`: `409`
- Timeout: `504`
- Unexpected failure: `500`

If no rows match the filters, the route returns an empty `duration_data` list and
a `reason`.

## Notes

- `snapshot_start` and `snapshot_end` filter stage `start_time`, not raw snapshot
  creation time.
- `duration_seconds` can serialize to `null` only if the internal value is not a
  finite float.
- `wafer_count` is from each lot's first snapshot occurrence for the queried
  product/lot, not necessarily from the stage entry timestamp.
