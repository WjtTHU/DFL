import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from statistics import mean, median
from typing import Any, Iterable, Literal, Mapping, cast

import polars as pl
from loguru import logger

from wip_analysis.data_loader.postgres_loader import fetch_data, get_conn_pool
from wip_analysis.data_loader.sql_template import render_sql
from wip_analysis.hwm.product_level.clustering import cluster_recipe_seq
from wip_analysis.hwm.product_level.primary_path import (
    get_pid_primary_path,
    get_preset_primary_path,
    get_preset_primary_paths,
)
from wip_analysis.hwm.product_level.utils.snapshot_to_seq import snapshot_to_seq
from wip_analysis.hwm.product_level.utils.stage_permutate import get_primary_path
from wip_analysis.utils.env_num import env_num

_PATH_STATS_LOCKS: dict[tuple[Any, ...], threading.Lock] = {}
_PATH_STATS_LOCKS_GUARD = threading.Lock()


@dataclass
class ClusterTimeStats:
    """Start-time statistics for lots in a cluster."""

    earliest_start: datetime | None
    latest_start: datetime | None
    mean_start: datetime | None
    median_start: datetime | None


@dataclass
class ClusterPrimaryPathInfo:
    """Primary path and stats for a single cluster."""

    cluster_id: int
    lot_count: int
    primary_path: list[str]
    time_stats: ClusterTimeStats


def _compute_cluster_time_stats(
    lot_ids: Iterable[str],
    lot_start_time_map: Mapping[str, datetime | None],
) -> ClusterTimeStats:
    """Compute start-time stats for a cluster.

    Args:
        lot_ids: Lot IDs in the cluster.
        lot_start_time_map: Mapping from lot_id to its start time.

    Returns:
        A ClusterTimeStats object. If no valid start_time exists, all fields are None.
    """

    start_times = [t for lid in lot_ids if (t := lot_start_time_map.get(lid)) is not None]

    if not start_times:
        return ClusterTimeStats(
            earliest_start=None,
            latest_start=None,
            mean_start=None,
            median_start=None,
        )

    start_times_sorted = sorted(start_times)
    earliest = start_times_sorted[0]
    latest = start_times_sorted[-1]

    # Return mean/median as datetime for easier downstream consumption.
    timestamps = [t.timestamp() for t in start_times_sorted]
    mean_ts = mean(timestamps)
    median_ts = median(timestamps)

    return ClusterTimeStats(
        earliest_start=earliest,
        latest_start=latest,
        mean_start=datetime.fromtimestamp(mean_ts, tz=earliest.tzinfo),
        median_start=datetime.fromtimestamp(median_ts, tz=earliest.tzinfo),
    )


def _build_lot_start_time_map(raw_df: pl.DataFrame) -> dict[str, datetime | None]:
    """Build mapping of lot_id -> start_time from snapshot data.

    The start_time is defined as the earliest `created_at` for that lot.

    Args:
        raw_df: Snapshot data. Must include: lot_id, created_at.

    Returns:
        Dict mapping lot_id to start_time.
    """

    logger.debug("Building lot -> start_time mapping")

    df = raw_df.with_columns(pl.col("created_at").alias("start_time"))

    lot_start = (
        df.group_by("lot_id", maintain_order=False)
        .agg(pl.col("start_time").min())
        .select("lot_id", "start_time")
    )

    lot_ids = lot_start["lot_id"].to_list()
    start_times = lot_start["start_time"].to_list()

    return dict(zip(lot_ids, start_times, strict=True))


def _build_lot_last_wafer_qty_map(raw_df: pl.DataFrame) -> dict[str, int]:
    """Build mapping of lot_id -> current_wafer_qty from the last record of each lot.

    The last record is defined as the row with the maximum created_at for that lot.

    Args:
        raw_df: Snapshot data. Must include: lot_id, current_wafer_qty, created_at.

    Returns:
        Dict mapping lot_id to wafer quantity (0 if null).
    """
    logger.debug("Building lot -> last-record wafer_qty mapping")

    lot_last_df = (
        raw_df.sort("created_at", descending=True)
        .unique(subset=["lot_id"], keep="first")
        .select("lot_id", "current_wafer_qty")
        .with_columns(pl.col("current_wafer_qty").fill_null(0).cast(pl.Int64))
    )

    return dict(
        zip(
            lot_last_df["lot_id"].to_list(),
            lot_last_df["current_wafer_qty"].to_list(),
            strict=True,
        )
    )


def _build_cache_key_rules(
    rules: Mapping[str, Any],
) -> tuple[Literal["dbscan", "hierarchical", "hdbscan"], dict[str, Any], str]:
    clustering_method_raw = str(rules.get("clustering_method") or "hdbscan")
    clustering_method = cast(Literal["dbscan", "hierarchical", "hdbscan"], clustering_method_raw)
    if clustering_method not in {"dbscan", "hierarchical", "hdbscan"}:
        raise ValueError(f"Unsupported clustering_method: {clustering_method}")

    clustering_params = {k: v for k, v in rules.items() if k != "clustering_method"}
    clustering_params_json = json.dumps(clustering_params, sort_keys=True)
    return clustering_method, clustering_params, clustering_params_json


@lru_cache(maxsize=512)
def _load_product_snapshot_bounds(
    product_id: str,
    snapshot_end: datetime,
) -> tuple[datetime | None, datetime | None]:
    query = render_sql(
        "get_pid_snapshot_bounds.sql.j2",
        {
            "product_id": product_id,
            "snapshot_end": snapshot_end,
        },
    )
    bounds_df = fetch_data(query)
    if bounds_df.is_empty():
        return None, None

    min_created_at = bounds_df.item(0, "min_created_at")
    max_created_at = bounds_df.item(0, "max_created_at")
    min_t = min_created_at if isinstance(min_created_at, datetime) else None
    max_t = max_created_at if isinstance(max_created_at, datetime) else None
    return min_t, max_t


def clear_paths_stats_caches() -> None:
    """Clear in-process path statistics helper caches."""
    _load_product_snapshot_bounds.cache_clear()


def _get_paths_stats_lock(key: tuple[Any, ...]) -> threading.Lock:
    with _PATH_STATS_LOCKS_GUARD:
        lock = _PATH_STATS_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PATH_STATS_LOCKS[key] = lock
        return lock


def _cached_path_covers_required_snapshots(
    cached_rows: list[dict[str, Any]],
    required_min_t: datetime | None,
    required_max_t: datetime | None,
) -> bool:
    if not cached_rows or required_min_t is None or required_max_t is None:
        return bool(cached_rows)

    cached_start = cached_rows[0].get("start_time")
    cached_end = cached_rows[0].get("end_time")
    if not isinstance(cached_start, datetime) or not isinstance(cached_end, datetime):
        return False

    return cached_start <= required_min_t and cached_end >= required_max_t


def _save_cached_paths(
    output: dict[str, list[dict[str, Any]]],
    snapshot_start: datetime | None,
    snapshot_end: datetime,
    clustering_method: str,
    clustering_params_json: str,
) -> dict[str, list[dict[str, Any]]]:
    if not output:
        return output

    rows_to_insert: list[tuple[str, dict[str, Any], tuple[Any, ...]]] = []
    for pid, paths in output.items():
        for path_info in paths:
            time_stats_json = json.dumps(
                path_info.get("time_stats") or {}, default=str, sort_keys=True
            )
            rows_to_insert.append(
                (
                    pid,
                    path_info,
                    (
                        pid,
                        path_info["path"],
                        int(path_info.get("lot_count", 0)),
                        int(path_info.get("wafer_qty", 0)),
                        time_stats_json,
                        path_info.get("type") == "Primary",
                        clustering_method,
                        clustering_params_json,
                        snapshot_start,
                        snapshot_end,
                    ),
                )
            )

    if rows_to_insert:
        insert_query = render_sql("insert_pid_path_returning.sql.j2")
        for _pid, path_info, params in rows_to_insert:
            try:
                path_id = _insert_cached_path_returning_id(insert_query, params)
            except Exception:
                logger.exception(
                    "Failed to save path cache row; continuing without path_id | pid={} | path_len={}",
                    _pid,
                    len(path_info.get("path") or []),
                )
                continue
            if path_id is not None:
                path_info["path_id"] = path_id
    logger.debug(
        "Inserted path cache rows (dedup by unique key) | target_pids={} | attempted_rows={}",
        len(output),
        len(rows_to_insert),
    )
    return output


def _insert_cached_path_returning_id(query: str, params: tuple[Any, ...]) -> Any:
    pool = get_conn_pool()
    with pool.connection() as conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                returned = cursor.fetchone()
                conn.commit()
                return returned[0] if returned is not None else None
        except Exception:
            conn.rollback()
            raise


def _load_cached_paths(
    snapshot_start: datetime | None,
    snapshot_end: datetime,
    product_ids: list[str] | None,
    clustering_method: str,
    clustering_params_json: str,
) -> dict[str, list[dict[str, Any]]]:
    query = render_sql(
        "get_pid_paths.sql.j2",
        {
            "start_time": snapshot_start,
            "end_time": snapshot_end,
            "clustering_method": clustering_method,
            "clustering_params": clustering_params_json,
            "product_ids": product_ids,
        },
    )
    cached_df = fetch_data(query)
    if cached_df.is_empty():
        return {}

    output: dict[str, list[dict[str, Any]]] = {}
    for row in cached_df.iter_rows(named=True):
        raw_time_stats = row.get("time_stats")
        if isinstance(raw_time_stats, str):
            try:
                time_stats = json.loads(raw_time_stats)
            except json.JSONDecodeError:
                time_stats = {}
        elif isinstance(raw_time_stats, dict):
            time_stats = raw_time_stats
        else:
            time_stats = {}

        time_stats_payload = {
            "earliest_start": time_stats.get("earliest_start"),
            "latest_start": time_stats.get("latest_start"),
            "mean_start": time_stats.get("mean_start"),
            "median_start": time_stats.get("median_start"),
        }

        pid = row["product_id"]
        output.setdefault(pid, []).append(
            {
                "cluster_id": 0,
                "path_id": row.get("path_id"),
                "lot_count": int(row.get("lot_count") or 0),
                "wafer_qty": int(row.get("wafer_qty") or 0),
                "path": list(row["path"] or []),
                "time_stats": time_stats_payload,
                "type": "Primary" if row["is_primary_path"] else "Secondary",
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
            }
        )

    for _pid, rows in output.items():
        rows.sort(key=lambda x: 1 if x["type"] == "Primary" else 0, reverse=True)
        for idx, item in enumerate(rows, start=1):
            item["cluster_id"] = idx

    return output


def get_paths_with_stats_for_products(
    snapshot_start: datetime | None,
    snapshot_end: datetime,
    product_ids: Iterable[str] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Get primary/secondary paths with single-flight protection for identical requests."""
    product_ids_list = list(product_ids) if product_ids is not None else None
    product_ids_key = tuple(sorted(set(product_ids_list))) if product_ids_list is not None else None
    rules_key = json.dumps(dict(rules or {}), sort_keys=True)
    lock_key = (
        snapshot_start.isoformat() if snapshot_start is not None else None,
        snapshot_end.isoformat(),
        product_ids_key,
        rules_key,
    )

    with _get_paths_stats_lock(lock_key):
        return _get_paths_with_stats_for_products_locked(
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
            product_ids=product_ids_list,
            rules=rules,
        )


def _get_paths_with_stats_for_products_locked(
    snapshot_start: datetime | None,
    snapshot_end: datetime,
    product_ids: Iterable[str] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Get primary/secondary paths (by clusters) with per-cluster stats.

    This function follows the steps:
    - Build recipe sequences from snapshots.
    - Cluster lot recipe sequences per product.
    - Solve a primary path for each cluster.

    Notes:
        - Cluster id 0 is treated as outliers.
        - The "primary" route for a product is typically the cluster with the largest `lot_count`.
          Other clusters can be considered "secondary" routes.

    Args:
        snapshot_start: Start time of snapshots to analyze (inclusive).
            Use None for an unbounded lower side.
        snapshot_end: End time of snapshots to analyze (inclusive).
        product_ids: Product IDs to analyze. If None, analyze all products in the time range.
        rules: Identification rules. This dict may include:
            - Any subset of `cluster_recipe_seq` parameters, e.g.
              `clustering_method`, `distance_metric`, `eps`, `min_samples`,
              `min_cluster_size`, `cluster_selection_epsilon` (HDBSCAN),

            - `sol_timeout` (float, seconds): Timeout passed into `get_primary_path`.
              Default is 300 seconds.

    Returns:
        Dict[product_id, List[cluster_info]] where each `cluster_info` contains:
            - cluster_id: int
            - lot_count: int
            - wafer_qty: int (sum of current_wafer_qty from each lot's last record)
            - primary_path: List[str]
            - time_stats: Dict with keys: earliest_start/latest_start/mean_start/median_start
    """

    rules = dict(rules or {})
    if "sol_timeout" in rules:
        sol_timeout = float(rules.pop("sol_timeout"))
    else:
        sol_timeout = env_num("WIP_ANALYSIS_SOLVER_TIMEOUT", 300.0)

    clustering_method, clustering_params, clustering_params_json = _build_cache_key_rules(rules)

    product_ids_list = list(product_ids) if product_ids is not None else None
    preset_paths = get_preset_primary_paths()
    preset_product_ids = (
        {pid for pid in product_ids_list if get_preset_primary_path(pid, preset_paths) is not None}
        if product_ids_list is not None
        else set(preset_paths) - {"*"}
    )
    cache_product_ids = (
        [pid for pid in product_ids_list if pid not in preset_product_ids]
        if product_ids_list is not None
        else None
    )
    if "*" in preset_paths or cache_product_ids == []:
        cached_paths = {}
    else:
        cached_paths = _load_cached_paths(
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
            product_ids=cache_product_ids,
            clustering_method=clustering_method,
            clustering_params_json=clustering_params_json,
        )
    if preset_product_ids:
        cached_paths = {
            pid: rows for pid, rows in cached_paths.items() if pid not in preset_product_ids
        }
    if product_ids_list is not None and snapshot_start is None:
        covered_cached_paths: dict[str, list[dict[str, Any]]] = {}
        for pid, rows in cached_paths.items():
            required_min_t, required_max_t = _load_product_snapshot_bounds(pid, snapshot_end)
            if _cached_path_covers_required_snapshots(rows, required_min_t, required_max_t):
                covered_cached_paths[pid] = rows
            else:
                logger.debug(
                    "Ignoring path stats cache for pid={} because cached range [{}, {}] does not cover required [{}, {}]",
                    pid,
                    rows[0].get("start_time") if rows else None,
                    rows[0].get("end_time") if rows else None,
                    required_min_t,
                    required_max_t,
                )
        cached_paths = covered_cached_paths

    if product_ids_list is None:
        if cached_paths and not preset_product_ids:
            logger.debug("Using cached path stats for all products in requested window")
            return cached_paths
        target_product_ids = list(preset_product_ids) if cached_paths else None
    else:
        target_product_ids = [pid for pid in product_ids_list if pid not in cached_paths]
        if not target_product_ids:
            logger.debug("Using fully cached path stats for all requested product_ids")
            return cached_paths

    logger.debug("Fetching snapshots in range [{}, {}]", snapshot_start, snapshot_end)

    # Fetch snapshot data in the given range.
    query = render_sql(
        "get_lid_snapshots_ranged.sql.j2",
        ctx={
            "snapshot_start": snapshot_start,
            "snapshot_cut": snapshot_end,
            "product_ids": target_product_ids,
        },
    )
    raw_df = fetch_data(query)

    if target_product_ids is not None:
        logger.debug("Filtering data for missing product_ids: {}", target_product_ids)
        raw_df = raw_df.filter(pl.col("product_id").is_in(target_product_ids))

    if raw_df.is_empty():
        logger.warning("No snapshot data found for given conditions.")
        return cached_paths

    effective_snapshot_start = raw_df["created_at"].min()
    if effective_snapshot_start is None:
        logger.warning("Snapshot data missing created_at values; skipping path stats computation.")
        return cached_paths

    # Build lot_id -> start_time mapping (for per-cluster statistics).
    lot_start_time_map = _build_lot_start_time_map(raw_df)
    # Build lot_id -> last-record wafer_qty mapping.
    lot_last_wafer_qty_map = _build_lot_last_wafer_qty_map(raw_df)

    # Products with a configured path still derive their descriptive statistics from
    # snapshots, but skip both clustering and per-cluster path optimization.
    output: dict[str, list[dict[str, Any]]] = {}
    configured_pid_set: set[str] = set()
    for pid in raw_df.get_column("product_id").unique().to_list():
        preset_path = get_preset_primary_path(str(pid), preset_paths)
        if preset_path is None:
            continue

        configured_pid_set.add(str(pid))
        pid_raw_df = raw_df.filter(pl.col("product_id") == pid)
        lot_ids = [str(lot_id) for lot_id in pid_raw_df.get_column("lot_id").unique().to_list()]
        logger.info(
            "Using preset primary path for product_id={} (len={}); skipping clustering and solver",
            pid,
            len(preset_path),
        )
        output[str(pid)] = [
            {
                "cluster_id": 1,
                "lot_count": len(lot_ids),
                "wafer_qty": sum(lot_last_wafer_qty_map.get(lid, 0) for lid in lot_ids),
                "path": preset_path,
                "time_stats": asdict(_compute_cluster_time_stats(lot_ids, lot_start_time_map)),
                "type": "Primary",
            }
        ]

    auto_raw_df = (
        raw_df.filter(~pl.col("product_id").is_in(list(configured_pid_set)))
        if configured_pid_set
        else raw_df
    )
    if auto_raw_df.is_empty():
        seq_df = pl.DataFrame()
    else:
        # Convert only non-preset products to recipe sequences. Preset products do not
        # need usable recipe history because their path comes from configuration.
        seq_df = snapshot_to_seq(raw_df=auto_raw_df)

    # Build direct single-lot clusters first, then cluster the remaining products.
    auto_seq_df = seq_df
    if auto_seq_df.is_empty():
        single_pid_set: set[str] = set()
    else:
        seq_counts = auto_seq_df.group_by("product_id").len().rename({"len": "seq_count"})
        single_pid_rows = seq_counts.filter(pl.col("seq_count") == 1)
        single_pid_set = set(single_pid_rows["product_id"].to_list())

    for pid in single_pid_set:
        pid_seq_df = auto_seq_df.filter(pl.col("product_id") == pid)
        if pid_seq_df.is_empty():
            continue

        lot_id = str(pid_seq_df.item(0, "lot_id"))
        recipe_seq_raw = pid_seq_df.item(0, "recipe_seq")
        if isinstance(recipe_seq_raw, pl.Series):
            recipe_seq_values = recipe_seq_raw.to_list()
        elif recipe_seq_raw is None:
            recipe_seq_values = []
        elif isinstance(recipe_seq_raw, list):
            recipe_seq_values = recipe_seq_raw
        else:
            recipe_seq_values = [recipe_seq_raw]
        recipe_seq = [str(stage_name) for stage_name in recipe_seq_values]

        logger.debug(
            "Skip clustering for product_id={} because only one recipe sequence is available",
            pid,
        )

        output[pid] = [
            {
                "cluster_id": 1,
                "lot_count": 1,
                "wafer_qty": int(lot_last_wafer_qty_map.get(lot_id, 0)),
                "path": recipe_seq,
                "time_stats": asdict(_compute_cluster_time_stats([lot_id], lot_start_time_map)),
                "type": "Primary",
            }
        ]

    cluster_seq_df = (
        auto_seq_df.filter(~pl.col("product_id").is_in(list(single_pid_set)))
        if single_pid_set
        else auto_seq_df
    )

    # Cluster recipe sequences for products with 2+ sequences.
    if cluster_seq_df.is_empty():
        cluster_results: dict[str, list[dict[str, Any]]] = {}
    else:
        logger.debug("Clustering recipe sequences for products")
        cluster_results = cluster_recipe_seq(
            cluster_seq_df, clustering_method=clustering_method, **clustering_params
        )

    # Solve primary path per cluster and compute stats.

    for pid, clusters in cluster_results.items():
        pid_clusters: list[dict[str, Any]] = []

        non_outlier_clusters = [c for c in clusters if int(c["cluster_id"]) != 0]
        single_cluster_path: list[str] | None = None
        if len(non_outlier_clusters) == 1:
            try:
                single_cluster_path = get_pid_primary_path(
                    pid,
                    path_start_time=(
                        effective_snapshot_start
                        if isinstance(effective_snapshot_start, datetime)
                        else None
                    ),
                    path_end_time=snapshot_end,
                )
                logger.debug(
                    "Single-cluster product_id={}: reusing non-clustered primary path "
                    "(len={}) instead of solving per-cluster CP-SAT",
                    pid,
                    len(single_cluster_path),
                )
            except Exception as e:
                logger.debug(
                    "Failed to load non-clustered primary path for product_id={}: {}; "
                    "falling back to per-cluster CP-SAT solve",
                    pid,
                    e,
                )
                single_cluster_path = None

        for cluster in clusters:
            cluster_id = int(cluster["cluster_id"])

            # Keep consistent with plot_primary_paths: treat cluster 0 as outliers.
            if cluster_id == 0:
                logger.debug(
                    f"Skip outlier cluster 0 for product_id={pid} (containing {len(cluster['cluster_lots'])} lots)"
                )
                continue

            cluster_lots: dict[str, list[str]] = cluster["cluster_lots"]
            lot_ids = list(cluster_lots.keys())

            # Skip the per-cluster CP-SAT solve in the single-cluster fast path.
            if single_cluster_path is not None:
                primary_path = single_cluster_path
            else:
                primary_path = get_primary_path(cluster_lots, sol_timeout=sol_timeout)

            # Start-time statistics for lots in this cluster.
            time_stats = _compute_cluster_time_stats(lot_ids, lot_start_time_map)

            info = ClusterPrimaryPathInfo(
                cluster_id=cluster_id,
                lot_count=len(lot_ids),
                primary_path=primary_path,
                time_stats=time_stats,
            )

            wafer_qty = sum(lot_last_wafer_qty_map.get(lid, 0) for lid in lot_ids)

            pid_clusters.append(
                {
                    "cluster_id": info.cluster_id,
                    "lot_count": info.lot_count,
                    "wafer_qty": wafer_qty,
                    "path": info.primary_path,
                    "time_stats": asdict(info.time_stats),
                }
            )

        if pid_clusters:
            # Sort by lot count descending, so "primary" cluster appears first.
            pid_clusters.sort(key=lambda x: x["lot_count"], reverse=True)
            pid_clusters[0]["type"] = "Primary"
            for i in range(1, len(pid_clusters)):
                pid_clusters[i]["type"] = "Secondary"
            output[pid] = pid_clusters

    # Persist only derived paths. Environment presets stay reversible: removing the
    # variable must not leave a preset path masquerading as an algorithm result in DB.
    preset_output = {pid: output[pid] for pid in configured_pid_set if pid in output}
    derived_output = {pid: rows for pid, rows in output.items() if pid not in configured_pid_set}
    if derived_output:
        output = _save_cached_paths(
            output=derived_output,
            snapshot_start=effective_snapshot_start,
            snapshot_end=snapshot_end,
            clustering_method=clustering_method,
            clustering_params_json=clustering_params_json,
        )
    else:
        output = {}
    output.update(preset_output)

    if cached_paths:
        merged = dict(cached_paths)
        merged.update(output)
        return merged
    return output
