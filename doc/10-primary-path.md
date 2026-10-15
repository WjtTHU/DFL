# Primary Path

The primary path is the product-level ordered stage sequence used to align HWM,
stage load, plots, and several planning calculations.

## Main Function

```python
get_pid_primary_path(
    pid: str | None,
    path_start_time: datetime | date | None = None,
    path_end_time: datetime | date | None = None,
    force_refresh: bool = False,
) -> list[str]
```

Source: `src/wip_analysis/hwm/product_level/primary_path.py`

`pid=None` is normalized to `"ALL"`.

## Product Path Table

Schema template: `sql/schemas/product.sql.j2`

Important columns:

- `product_id`
- `path_start_time`
- `path_end_time`
- `primary_path`
- `primary_path_durations`
- `min_completion_time`
- `calculation_time`

Primary key:

```text
(product_id, path_start_time, path_end_time)
```

`product_id = 'ALL'` represents the cross-product path.

## DB-First Flow

1. Unless `force_refresh=True`, render `get_pid_primary_paths.sql.j2`.
2. Fetch candidate rows for the product and optional timestamp bounds.
3. Select the widest row by:
   - largest `path_end_time - path_start_time`
   - latest `path_end_time`
   - earliest `path_start_time`
4. If product is not `"ALL"` and `path_end_time` is supplied, query raw snapshot
   bounds with `get_pid_snapshot_bounds.sql.j2`.
5. If the cached row does not cover required snapshot timestamps, treat it as a cache
   miss.
6. Return `primary_path` as `list[str]`.

The shared SQL ordering fragment is `sql/templates/_product_path_order.sql.j2`.

## Recalculation Flow

On cache miss, or when `force_refresh=True`:

1. Serialize recalculation per `(pid, path_start_time, path_end_time)` with a
   process-local lock.
2. Recheck the DB cache after acquiring the lock.
3. Query raw snapshots with `get_pid_snapshots.sql.j2`.
4. Convert snapshots to lot recipe sequences with `snapshot_to_seq()`.
5. Solve the representative path with `get_primary_path()`.
6. Resolve effective start/end times from actual snapshot min/max times, falling
   back to supplied bounds when necessary.
7. Upsert through `upsert_pid_primary_path.sql.j2`.

## Optimization

`get_primary_path()` lives in
`src/wip_analysis/hwm/product_level/utils/stage_permutate.py`.
It takes a mapping of `lot_id -> recipe_seq` and solves a representative stage
order. Solver timeout is controlled by callers; in path stats it defaults to
`WIP_ANALYSIS_SOLVER_TIMEOUT` or `300` seconds.

## Durations

`get_pid_primary_durs()` in `primary_path_dur.py` uses the same product path
table and fills `primary_path_durations`.

Flow:

1. Load matching product path rows.
2. Select the widest cached row.
3. Validate `primary_path_durations` exists, is an array, is non-empty, and
   matches path length.
4. If invalid or missing, solve durations from stage duration data.
5. Upsert with `upsert_pid_primary_path_durations.sql.j2`.

## Current API Consumers

- `POST /api/v1/products/{product_id}`: response `stage_order`.
- `POST /api/v1/lots/{lot_id}`: response top-level `stage_order`.
- Stage load: aligns load arrays to primary path.
- Super-hot plot: stage axis ordering.
- Bottleneck detection: includes `primary_path` in each result.
- Lot quickest finish: determines remaining stages.
- Path stats: may reuse product primary path for a single non-outlier cluster.

## Distinction From Path Candidates

The product primary path is stored in the product table. Path candidates are
stored separately by the path stats pipeline and can represent secondary routes
or cluster-specific routes. Lot-level progress may select a path candidate for a
single snapshot row, but top-level product/lot `stage_order` should use
`get_pid_primary_path()` with no `path_start_time` bound for lot-level windows,
and only a `path_end_time` bound when one is needed to avoid future data.

## Cache Clearing

`clear_pid_primary_path_cache()` clears only the in-process LRU cache.
`clear_refresh_lru_caches()` calls it together with related lot/path/stage
caches. It does not delete rows from the database.
