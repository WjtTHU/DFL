# Lot Quickest Finish

Lot quickest finish is an internal estimator plus CLI script. It is not exposed
as a FastAPI route in the current app.

## Entry Points

- Function: `src/wip_analysis/hwm/lot_level/quickest_finish.py`
- Script: `scripts/analysis/lot_quickest_finish.py`

CLI:

```bash
python scripts/analysis/lot_quickest_finish.py LOT_ID \
  --duration-quantile 0.0 \
  --capacity-window 7d
```

Options:

- `--duration-quantile`: quantile in `[0, 1]`; default `0.0`.
- `--capacity-window`: `7d` or `all`; default `7d`.
- `--no-queue`: disable queue-wait estimate.

## Function

```python
get_lot_quickest_finish(
    lot_id: str,
    *,
    as_of: datetime | None = None,
    config: LotQuickestFinishConfig | None = None,
) -> dict[str, Any]
```

`LotQuickestFinishConfig`:

- `duration_quantile`: default `0`.
- `capacity_window`: `"7d"` or `"all"`, default `"7d"`.
- `include_queue_wait`: default `True`.

## Output Shape

Successful result:

```json
{
  "lot_id": "LOT1",
  "product_id": "PID1",
  "as_of": "2024-01-10T08:00:00",
  "current_stage": "STAGE_A",
  "remaining_stages": ["STAGE_A", "STAGE_B"],
  "config": {
    "duration_quantile": 0.0,
    "capacity_window": "7d",
    "include_queue_wait": true
  },
  "earliest_finish_at": "2024-01-12T08:00:00",
  "total_seconds": 172800.0,
  "stages": [
    {
      "stage_name": "STAGE_A",
      "entry_time": "2024-01-10T08:00:00",
      "duration_floor_seconds": 3600.0,
      "elapsed_seconds": 1200.0,
      "remaining_process_seconds": 2400.0,
      "capacity_lots_per_day": 20.0,
      "wip_lots": 5,
      "queue_wait_seconds": 17280.0,
      "stage_total_seconds": 19680.0,
      "exit_time": "2024-01-10T13:28:00"
    }
  ]
}
```

Error result keeps the same top-level keys where possible and adds `error`.

## Key Calculation Flow

1. Load the lot's latest raw snapshot and product id.
2. Use `as_of` if supplied; otherwise use the latest snapshot timestamp.
3. Load product primary path with `get_pid_primary_path(product_id)`.
4. Get current lot stage details with `get_lot_prog_on_snapshot(lot_id, as_of)`.
5. Determine the current stage as the last actual/interpolated stage on the
   primary path.
6. Load historical durations with `get_stage_duration(..., cut_time=as_of)`.
7. For each primary-path stage, choose the configured duration quantile as a
   processing-time floor. Missing durations become `0`.
8. Load stage capacity with `get_stage_capacity(product_id=product_id)`.
9. For each stage, use max observed `entering_lot_count` as capacity:
   - last 7 days when `capacity_window="7d"`
   - all returned history when `capacity_window="all"`
10. Walk remaining stages sequentially, carrying each exit time into the next
    stage's entry time.
11. If queue wait is enabled, estimate:

```text
lots_ahead = max(wip_lots - 1, 0) for the current stage
lots_ahead = wip_lots for later stages
queue_wait_seconds = lots_ahead / capacity_lots_per_day * 86400
```

12. Return the final projected exit time and per-stage details.

## Boundaries

- This is an optimistic planning estimate, not the model-backed date-out
  prediction route.
- Capacity is derived from observed entry counts, not equipment constraints.
- Future WIP is approximated by product HWM rows through `_stage_wip_counts()`.
- Missing capacity disables queue wait for that stage.
