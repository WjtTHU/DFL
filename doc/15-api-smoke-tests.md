# API Smoke Tests

These checks validate that live API routes respond with structurally reasonable
results. They do not assert business golden values because the configured
database is live data and expected counts change with cache state.

## Script

Run against a running local backend:

```bash
python -B scripts/debug/smoke_api_routes.py
```

Useful options:

```bash
python -B scripts/debug/smoke_api_routes.py \
  --base-url http://localhost:8724 \
  --product-id '5H0mKnT7qf6hZjrm9r4SfA==' \
  --lot-id '+EzHRhFT3SjBC3CqHWQSOw==' \
  --snapshot-start '2025-02-05 08:30:25' \
  --snapshot-end '2025-02-05 09:30:26'
```

The script writes response artifacts under `/tmp/wip-api-smoke` by default and
prints a JSON summary. HTML routes are stored as HTML artifacts; JSON routes are
summarized by status, response size, and key list lengths.

`POST /api/v1/products/{product_id}` is included by default with the same
bounded `snapshot_start` / `snapshot_end` smoke window as the lot and load
routes. The unbounded all-history product HWM request is intentionally excluded
because it is heavy. Include it explicitly:

```bash
python -B scripts/debug/smoke_api_routes.py --include-full-product-hwm
```

## Default Route Coverage

- `POST /api/v1/health`
- `POST /api/v1/lots/{lot_id}`
- `POST /api/v1/lots/{lot_id}/progress-z.html`
- `POST /api/v1/products/{product_id}`
- `POST /api/v1/products/{product_id}/load_ratio`
- `POST /api/v1/stage_load_ratio`
- `POST /api/v1/stage_cap`
- `POST /api/v1/stage_dur`
- `POST /api/v1/paths_stats`
- `POST /api/v1/products/{product_id}/paths.html`
- `POST /api/v1/products/{product_id}/date_out`
- `POST /api/v1/bottleneck_detection`
- `POST /api/v1/products/{product_id}/super-hot-lot.html`

For routes where the selected window may naturally have no data, structured
`404` is accepted as a smoke pass when the response has a reason.

## Observed Product HWM Cost

Using product `5H0mKnT7qf6hZjrm9r4SfA==`, the first product HWM call after
rebuilding the product path table exceeded a 300 second client timeout. The
server continued warming caches after the client disconnected.

The slow path was:

1. `ensure_lot_progress_populated()` populated latest lot HWM rows for many lots.
2. Product primary-path cache rows were generated for many exact snapshot end
   timestamps.
3. `get_pid_stage_stats.sql.j2` returned about 1.1 million rows for the full
   product-wide request.
4. Python then assembled dense stage arrays and serialized a large response.

After warmup, the same product HWM route returned `200`, but still took about
88 seconds and returned about 16 MB. This means the endpoint is correct enough
for smoke validation, but the full-product default is too expensive for routine
interactive use.

Implemented immediate mitigation:

- `POST /api/v1/products/{product_id}` accepts optional `snapshot_start` /
  `snapshot_end` request body fields so callers can avoid the all-history path.
- Product HWM result assembly now scans fetched stage-stat rows once instead of
  repeatedly filtering the whole result per `(calendar_time, snapshot_cut)`.

Remaining optimization candidates:

- Push the stage-array aggregation into SQL or a vectorized Polars pivot instead
  of fetching per-stage rows and looping per `(calendar_time, snapshot_cut)`.
- Consider pagination or a summary mode for product HWM responses.
- Add cancellation or bounded wait behavior around product-wide readiness
  warmup so disconnected clients do not leave long-running work behind.
