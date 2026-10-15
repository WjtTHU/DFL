# Lot-Level HWM

Lot-level HWM describes one lot's progress over multiple raw snapshot cuts.
The API returns one row per snapshot, with actual/interpolated/predicted stage
details and rolling comparison metrics.

## Routes

### `POST /api/v1/lots/{lot_id}`

Source:

- Route: `src/wip_api/routes/hwm_lot.py`
- Models: `src/wip_api/models/hwm_lot.py`
- Main calculation: `src/wip_analysis/hwm/lot_level/all_history.py`
- Per-snapshot calculation: `src/wip_analysis/hwm/lot_level/single_step.py`

Request body is optional:

```json
{
  "product_id": "optional product guard",
  "refresh": false,
  "snapshot_start": "2024-01-01T00:00:00",
  "snapshot_end": "2024-01-31T23:59:59"
}
```

Response shape:

```json
{
  "lot_id": "LOT1",
  "product_id": "PID1",
  "stage_order": ["STAGE_A", "STAGE_B"],
  "snapshots": [
    {
      "snapshot_at": "2024-01-02T08:00:00",
      "path_id": 123,
      "path_edit_distance": null,
      "stage_details": [
        {
          "stage_name": "STAGE_A",
          "start_time": "2024-01-01T12:00:00",
          "stage_type": "actual",
          "cumulate_h": 4.5,
          "duration_z": null,
          "comparisons": {
            "dev_vs_1d": 0.0,
            "avg_cum_1d": 4.5
          }
        }
      ]
    }
  ]
}
```

### `POST /api/v1/lots/{lot_id}/progress-z.html`

Returns a full Plotly HTML document for a heatmap of `duration_z` values.
The route accepts query parameters:

- `refresh`: boolean, default `false`
- `snapshot_start`: optional datetime
- `snapshot_end`: optional datetime

Source:

- Route: `src/wip_api/routes/hwm_lot.py`
- Plot builder: `src/wip_analysis/hwm/lot_level/stage_dur_z.py`
- CLI script: `scripts/analysis/lot_progress_z.py`

## Validation And Errors

- `lot_id` is checked with `check_lot_id_legality()`.
- Missing lot returns `404`.
- Illegal lot id returns `400`.
- If request `product_id` does not match the lot's product, the route returns `400`.
- `snapshot_start >= snapshot_end` returns `400`.
- Unexpected calculation failure returns `500`.

## Key Calculation Flow

1. Validate the lot and optional product guard.
2. Load all raw snapshot times for the lot through
   `get_lid_snapshot_times.sql.j2`.
3. For each snapshot cut, call `get_lot_prog_on_snapshot()`.
4. Preload cached `lot_progress` rows for bounded ranges when possible.
5. Build or reuse stage details for each snapshot.
6. Compute cumulative hours from the lot start time.
7. Compute rolling cumulative averages and deviations for windows
   `1d`, `7d`, `1m`, `1q`, `6m`, and `1y`.
8. Join stage duration averages/stddevs from `get_stage_duration_stats()` and
   compute `duration_z`.
9. Return rows sorted by `snapshot_at`.

## Stage Order Semantics

There are two distinct orders:

- Per-row order: each `snapshots[*].stage_details` row is ordered by cumulative
  time. The row may have been generated from a selected path candidate, and the
  row exposes `path_id` plus `path_edit_distance`.
- Top-level `stage_order`: this is the product primary path from
  `get_pid_primary_path(actual_product_id, path_start_time=None, path_end_time)`.
  It is sourced from the product primary path table (`product.sql.j2`) when a
  matching row exists. It is not copied from the last lot-progress row.

The route passes `snapshot_end` through for the top-level product primary path
lookup. It does not pass `snapshot_start`, because doing so would reject wider
historical paths that are valid for the same non-future end bound.

## Path Candidate Matching

`get_lot_prog_on_snapshot()` loads path candidates with
`get_pid_path_candidates()` and compares candidates against actual stage names
for the lot:

- Candidates below coverage threshold are ignored.
- Levenshtein distance is the default primary distance.
- Jaccard distance is used as tie-breaker by default.
- If no candidate is suitable, the product primary path is used.

This per-snapshot selection is intentionally separate from top-level
`stage_order`.

## Persistence

Computed per-snapshot stage details are persisted through
`upsert_lid_progress_up_to_cut.sql.j2`. Persisted rows include stage start/end,
duration seconds, stage type, and optional `path_id`.
