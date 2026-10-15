import json
import os
import threading
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Any, Mapping, Optional

from loguru import logger

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql
from wip_analysis.utils.env_num import env_num

from .aggregate_paths_stats import get_paths_with_stats_for_products
from .primary_path import get_preset_primary_path

_path_candidate_locks_guard = threading.Lock()
_path_candidate_locks: dict[tuple[Any, ...], threading.Lock] = {}


def _get_path_candidate_lock(key: tuple[Any, ...]) -> threading.Lock:
    with _path_candidate_locks_guard:
        lock = _path_candidate_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _path_candidate_locks[key] = lock
        return lock


def _resolve_candidate_end_time(
    snapshot_end: Optional[datetime],
    path_end_date: Optional[date],
) -> Optional[datetime]:
    # Prefer day-level bound when provided so callers can query paths with
    # "end_time <= <date>" semantics instead of an exact snapshot timestamp.
    if path_end_date is not None:
        return datetime.combine(path_end_date, time.max)
    if snapshot_end is not None:
        return snapshot_end
    return None


def _select_widest_cached_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None

    def _row_key(row: dict[str, Any]) -> tuple[float, float, float, bool, int]:
        start = row.get("start_time")
        end = row.get("end_time")
        if isinstance(start, datetime) and isinstance(end, datetime):
            span = (end - start).total_seconds()
            end_ts = end.timestamp()
            start_ts = start.timestamp()
        else:
            span = -1.0
            end_ts = float("-inf")
            start_ts = float("inf")

        return (
            span,
            end_ts,
            -start_ts,
            bool(row.get("is_primary_path")),
            int(row.get("lot_count") or 0),
        )

    return max(rows, key=_row_key)


@lru_cache(maxsize=512)
def _load_snapshot_bounds(
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


def _has_uncovered_snapshots(
    cached_row: dict[str, Any],
    required_min_t: datetime | None,
    required_max_t: datetime | None,
    staleness_tolerance: timedelta = timedelta(0),
) -> bool:
    if required_min_t is None or required_max_t is None:
        return False

    cached_start = cached_row.get("start_time")
    cached_end = cached_row.get("end_time")
    if not isinstance(cached_start, datetime) or not isinstance(cached_end, datetime):
        return True

    # A cached row "covers" the cut if it spans [required_min_t, required_max_t].
    # `staleness_tolerance` relaxes ONLY the recent (end) bound: a path computed up to
    # this many days before the cut is still accepted. This is the temporally-safe
    # direction — the cached window ends in the PAST relative to the cut, so it never
    # pulls future data into an earlier snapshot (preserves the no-time-travel design).
    # The start bound stays strict so a too-narrow (late-starting) window still recomputes.
    return cached_start > required_min_t or cached_end < required_max_t - staleness_tolerance


def _filter_covering_cached_rows(
    rows: list[dict[str, Any]],
    required_min_t: datetime | None,
    required_max_t: datetime | None,
    staleness_tolerance: timedelta = timedelta(0),
) -> list[dict[str, Any]]:
    if required_min_t is None or required_max_t is None:
        return rows

    return [
        row
        for row in rows
        if not _has_uncovered_snapshots(
            row,
            required_min_t,
            required_max_t,
            staleness_tolerance,
        )
    ]


def get_pid_path_candidates(
    pid: Optional[str],
    snapshot_end: Optional[datetime] = None,
    path_end_date: Optional[date] = None,
    rules: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Load path candidates for a product with DB-first behavior.

    Args:
        pid: Target product id. Uses "ALL" if None.
        snapshot_end: Upper bound timestamp for candidate rows.
        path_end_date: Optional date bound; used when snapshot_end is absent.
        rules: Optional clustering rules. Defaults to hdbscan when not provided.
    Returns:
        Candidate rows with at least fields: path_id, path, lot_count, wafer_qty,
        is_primary_path, start_time, end_time.
    """
    pid = pid or "ALL"
    effective_end = _resolve_candidate_end_time(snapshot_end, path_end_date)
    preset_path = get_preset_primary_path(pid)
    if preset_path is not None:
        logger.info(
            "Using preset path candidate from WIP_ANALYSIS_PRESET_PRIMARY_PATHS for pid={} "
            "(len={}); skipping path-cache lookup, clustering, and solver",
            pid,
            len(preset_path),
        )
        return [
            {
                "path_id": None,
                "path": preset_path,
                "lot_count": 0,
                "wafer_qty": 0,
                "is_primary_path": True,
                "start_time": None,
                "end_time": effective_end,
            }
        ]

    selected_rules: dict[str, Any] = dict(rules or {})
    clustering_method = str(selected_rules.get("clustering_method") or "hdbscan")
    clustering_params = {k: v for k, v in selected_rules.items() if k != "clustering_method"}
    clustering_params_json = json.dumps(clustering_params, sort_keys=True)

    return [
        dict(row)
        for row in _get_pid_path_candidates_cached(
            pid,
            effective_end,
            clustering_method,
            clustering_params_json,
            id(fetch_data),
            id(render_sql),
            id(get_paths_with_stats_for_products),
            os.environ.get("PYTEST_CURRENT_TEST", ""),
        )
    ]


@lru_cache(maxsize=1024)
def _get_pid_path_candidates_cached(
    pid: str,
    effective_end: datetime | None,
    clustering_method: str,
    clustering_params_json: str,
    _fetch_data_token: int,
    _render_sql_token: int,
    _compute_token: int,
    _test_token: str,
) -> tuple[dict[str, Any], ...]:
    clustering_params = json.loads(clustering_params_json)
    selected_rules = {
        "clustering_method": clustering_method,
        **clustering_params,
    }
    product_ids = None if pid == "ALL" else [pid]
    cache_key = (
        pid,
        effective_end,
        clustering_method,
        clustering_params_json,
    )
    lock = _get_path_candidate_lock(cache_key)
    with lock:
        rows = _get_pid_path_candidates_locked(
            pid=pid,
            effective_end=effective_end,
            product_ids=product_ids,
            selected_rules=selected_rules,
            clustering_method=clustering_method,
            clustering_params_json=clustering_params_json,
        )
    return tuple(rows)


def _get_pid_path_candidates_locked(
    *,
    pid: str,
    effective_end: datetime | None,
    product_ids: list[str] | None,
    selected_rules: dict[str, Any],
    clustering_method: str,
    clustering_params_json: str,
) -> list[dict[str, Any]]:
    cached_rows: list[dict[str, Any]] = []
    selected_row: dict[str, Any] | None = None

    try:
        query = render_sql(
            "get_pid_paths.sql.j2",
            {
                "product_ids": product_ids,
                "clustering_method": clustering_method,
                "clustering_params": clustering_params_json,
                "end_time_upper": effective_end,
            },
        )
        rows = fetch_data(query)
        if not rows.is_empty():
            cached_rows = rows.to_dicts()

        if cached_rows and pid != "ALL":
            required_min_t, required_max_t = _load_snapshot_bounds(pid, effective_end)
            staleness_tolerance = timedelta(
                days=env_num("WIP_ANALYSIS_PATH_STALENESS_TOLERANCE_DAYS", 0.0)
            )
            covering_rows = _filter_covering_cached_rows(
                cached_rows,
                required_min_t,
                required_max_t,
                staleness_tolerance,
            )
            selected_row = _select_widest_cached_row(covering_rows)
            if selected_row is not None:
                logger.debug(
                    "Path candidates cache hit (widest-covered) pid={} required=[{}, {}] cached=[{}, {}] rows={}",
                    pid,
                    required_min_t,
                    required_max_t,
                    selected_row.get("start_time"),
                    selected_row.get("end_time"),
                    len(covering_rows),
                )
                covering_rows.sort(
                    key=lambda row: (
                        row.get("end_time") or datetime.min,
                        bool(row.get("is_primary_path")),
                        int(row.get("lot_count") or 0),
                    ),
                    reverse=True,
                )
                return covering_rows

        if cached_rows and pid == "ALL":
            selected_row = _select_widest_cached_row(cached_rows)
            if selected_row is not None:
                cached_rows.sort(
                    key=lambda row: (
                        row.get("end_time") or datetime.min,
                        bool(row.get("is_primary_path")),
                        int(row.get("lot_count") or 0),
                    ),
                    reverse=True,
                )
                return cached_rows

        logger.debug(
            "Path candidates cache miss/gap pid={} end_time={}, will compute",
            pid,
            effective_end,
        )
    except Exception as e:
        logger.exception(f"Failed to fetch path candidates from DB for {pid}: {e}")

    if effective_end is None:
        return []

    # Fallback: compute paths for a recent window and use the same clustering method.
    fallback_rules = dict(selected_rules)
    fallback_rules["clustering_method"] = clustering_method

    snapshot_start = None
    computed = get_paths_with_stats_for_products(
        snapshot_start=snapshot_start,
        snapshot_end=effective_end,
        product_ids=product_ids,
        rules=fallback_rules,
    )
    logger.debug(
        "Computed path candidates pid={} end_time={} keys={}",
        pid,
        effective_end,
        list(computed.keys()),
    )

    result_pid = pid if pid != "ALL" else next(iter(computed), None)
    if result_pid is None:
        return []

    candidates: list[dict[str, Any]] = []
    for row in computed.get(result_pid, []):
        candidates.append(
            {
                "path_id": row.get("path_id"),
                "path": row.get("path") or [],
                "lot_count": row.get("lot_count") or 0,
                "wafer_qty": row.get("wafer_qty") or 0,
                "is_primary_path": row.get("type") == "Primary",
                "start_time": snapshot_start,
                "end_time": effective_end,
            }
        )

    candidates.sort(
        key=lambda row: (
            row.get("end_time") or datetime.min,
            bool(row.get("is_primary_path")),
            int(row.get("lot_count") or 0),
        ),
        reverse=True,
    )
    return candidates


def clear_pid_path_candidate_caches() -> None:
    """Clear in-process path candidate result caches."""
    _load_snapshot_bounds.cache_clear()
    _get_pid_path_candidates_cached.cache_clear()
