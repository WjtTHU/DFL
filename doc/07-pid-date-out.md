# Product Date-Out

Product date-out returns actual and model-predicted output counts for requested
target dates.

## Route

### `POST /api/v1/products/{product_id}/date_out`

Source:

- Route: `src/wip_api/routes/pid_date_out.py`
- Models: `src/wip_api/models/pid_date_out.py`
- Readiness: `src/wip_analysis/hwm/stage/utils/lot_progress_readiness.py`
- Prediction persistence:
  `src/wip_analysis/hwm/stage/utils/lot_completion_predictions.py`
- Inference: `src/wip_pred/inference/predict.py`

Request:

```json
{
  "target_dates": ["2024-01-31", "2024-02-01"]
}
```

Backward-compatible request alias:

```json
{
  "target_date": "2024-01-31"
}
```

The response preserves the requested date order.

## Response Shape

```json
{
  "product_id": "PID1",
  "target_dates": ["2024-01-31"],
  "lot_counts": [12],
  "wafer_counts": [300],
  "actual_lot_counts": [8],
  "actual_wafer_counts": [200],
  "predicted_lot_counts": [4],
  "predicted_wafer_counts": [100],
  "date_outputs": [
    {
      "target_date": "2024-01-31",
      "final": {
        "actual": {"lot_count": 8, "wafer_count": 200},
        "predicted": {"lot_count": 4, "wafer_count": 100}
      },
      "stages": [
        {
          "stage_name": "STAGE_A",
          "actual": {"lot_count": 2, "wafer_count": 50},
          "predicted": {"lot_count": 1, "wafer_count": 25}
        }
      ]
    }
  ],
  "reason": null
}
```

`lot_counts` and `wafer_counts` are the sums of `actual` and `predicted` final
counts for each target date.

## Key Calculation Flow

`_calculate_pid_date_out(product_id, target_days)`:

1. Calls `ensure_date_out_pred_populated()` to make sure every lot's latest
   raw snapshot before each target day has a model prediction row in
   `lot_completion_predictions`.
2. Queries candidate finish lots with `get_pid_date_out_lots.sql.j2`.
3. If no finish lots are found, checks latest wafer data for the product and
   returns all-zero counts when snapshot data exists.
4. For actual finish lots, calls `ensure_lot_progress_populated()` for each
   required `(lot_id, snapshot_cut)` pair.
5. Queries final/stage output rows with `get_pid_date_out_outputs.sql.j2`.
6. Builds per-date `actual` and `predicted` splits.
7. Aggregates final split totals into the flat count arrays.

## Prediction Population

Missing completion predictions are computed by
`compute_lot_completion_predictions()`:

1. Group missing lots by `snapshot_cut`.
2. Load raw snapshots for the product in the model window ending at that
   snapshot cut.
3. Use `WipInferenceService.predict_remaining_days()` with
   `DEFAULT_MODEL_DIR = checkpoints/_release`.
4. Persist `(lot_id, snapshot_cut, remaining_days, predicted_completion_time,
   model_dir)` rows through `upsert_lot_completion_predictions.sql.j2`.

The readiness helper blocks by default. A timeout returns `504`.

## Errors

- Missing snapshot wafer data for product: `404`.
- Lot-progress readiness `ValueError`: `409` when the message indicates
  `lot_progress_table data is not ready yet`.
- Completion prediction or readiness timeout: `504`.
- Unexpected failure: `500`.

## Related Script

`scripts/analysis/pid_date_out.py` remains available for command-line analysis,
but the API route above is the current service entry point.
