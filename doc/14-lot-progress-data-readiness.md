# Lot Progress Data Readiness

Several product and stage routes depend on `lot_progress` cache rows. Readiness
helpers make those dependencies explicit: they check whether required rows exist,
trigger background population for missing rows, and block until ready or timed
out.

Source: `src/wip_analysis/hwm/stage/utils/lot_progress_readiness.py`

## Main Helpers

### `ensure_lot_progress_populated()`

Used by:

- Product HWM
- Stage duration when cache population is required
- Stage capacity
- Stage load
- Super-hot lot plot
- Date-out finish-lot readiness

Readiness rule:

1. For the product/lot/snapshot scope, find each lot's latest raw snapshot at or
   before `snapshot_end`.
2. Require a `lot_progress` row for each required `(lot_id, latest snapshot)`.
3. If rows are missing, trigger background `get_lot_prog_over_snapshots()` for
   those exact snapshots.
4. Poll until all rows are ready.
5. Raise `TimeoutError` when `max_wait_seconds` is exceeded.

Arguments accepted for compatibility but not used in the current readiness rule:

- `stage_names`
- `stage_time_start`
- `stage_time_end`
- `require_duration`

`wait_if_missing` is retained for compatibility and currently ignored; readiness
is always blocking.

### `ensure_date_out_pred_populated()`

Used by `POST /api/v1/products/{product_id}/date_out`.

Rule:

1. For each target day, find each lot's latest raw snapshot before the end of
   that day.
2. Require a corresponding row in `lot_completion_predictions`.
3. Trigger `compute_lot_completion_predictions()` for missing rows.
4. Poll until predictions are ready or timeout.

### `ensure_date_out_lot_progress_populated_for_product()`

Narrow date-out helper. It checks latest lot-progress rows for target days, not
all product progress history.

The current date-out route directly calls `ensure_lot_progress_populated()` for
finish lots after querying date-out lot candidates, and uses prediction readiness
before that.

## Background Work

Readiness helpers maintain process-local in-flight scope keys so duplicate API
requests do not start duplicate background work for the same scope.

Worker count for lot progress recalculation:

```text
min(task_count, WIP_ANALYSIS_LOT_PROGRESS_RECALC_WORKERS or 8)
```

Background population calls `get_lot_prog_over_snapshots()` with
`refresh=False`, so it can reuse existing caches where valid.

## Returned Readiness Dict

Successful helpers return:

```json
{
  "expected_snapshot_cuts": 10,
  "ready_snapshot_cuts": 10,
  "missing_snapshot_cuts": 0,
  "triggered_lot_count": 3
}
```

If no raw snapshots exist for the scope, all counts are `0`.

## SQL Templates

- `get_scope_latest_lot_progress_readiness.sql.j2`
- `get_scope_lot_ids.sql.j2`
- `get_scope_lot_progress_cuts.sql.j2`
- `get_scope_snapshot_times.sql.j2`
- `get_pid_date_out_lot_progress_readiness.sql.j2`
- `get_pid_date_out_prediction_readiness.sql.j2`
- `upsert_lot_completion_predictions.sql.j2`

## Route-Level Effects

- `ValueError` from readiness commonly maps to `409`.
- `TimeoutError` maps to `504`.
- Long-running first requests may block while background rows are computed.
- Subsequent requests should reuse DB rows and in-process LRU caches.

## Operational Notes

- Readiness triggers in-process background threads; it is not a durable job
  queue.
- If the process restarts, in-flight scope tracking is lost but DB cache rows
  remain.
- For debugging actual DB state, use `scripts/debug/run_sql.py`.
