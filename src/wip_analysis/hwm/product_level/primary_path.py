import os
import threading
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Callable, Mapping, Optional

import polars as pl
from loguru import logger

from wip_analysis.data_loader.postgres_loader import execute_query_async, fetch_data
from wip_analysis.data_loader.sql_template import render_sql
from wip_analysis.utils.env_num import env_json, env_num
from wip_analysis.utils.logging import format_sequence

from .utils.snapshot_to_seq import snapshot_to_seq
from .utils.stage_permutate import get_primary_path

_PRIMARY_PATH_LOCKS: dict[tuple[str, datetime | None, datetime | None], threading.Lock] = {}
_PRIMARY_PATH_LOCKS_GUARD = threading.Lock()
PathTime = datetime | date
_PRESET_PRIMARY_PATHS_ENV = "WIP_ANALYSIS_PRESET_PRIMARY_PATHS"


def get_preset_primary_paths() -> dict[str, list[str]]:
    """Return validated product -> primary-path overrides from the environment."""
    raw = os.environ.get(_PRESET_PRIMARY_PATHS_ENV)
    if raw is None or not raw.strip():
        return {}

    configured: object = env_json(_PRESET_PRIMARY_PATHS_ENV, None)
    if not isinstance(configured, dict):
        raise ValueError(f"{_PRESET_PRIMARY_PATHS_ENV} must be a JSON object")

    paths: dict[str, list[str]] = {}
    for product_id, path in configured.items():
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError(
                f"{_PRESET_PRIMARY_PATHS_ENV} contains an empty or non-string product_id"
            )
        if not isinstance(path, list) or not path:
            raise ValueError(
                f"{_PRESET_PRIMARY_PATHS_ENV}[{product_id!r}] must be a non-empty JSON array"
            )
        if any(not isinstance(stage, str) or not stage.strip() for stage in path):
            raise ValueError(
                f"{_PRESET_PRIMARY_PATHS_ENV}[{product_id!r}] must contain only non-empty strings"
            )
        if len(set(path)) != len(path):
            raise ValueError(
                f"{_PRESET_PRIMARY_PATHS_ENV}[{product_id!r}] contains duplicate stages"
            )
        paths[product_id] = list(path)
    return paths


def get_preset_primary_path(
    pid: Optional[str],
    preset_paths: Mapping[str, list[str]] | None = None,
) -> list[str] | None:
    """Resolve an exact product override, then the optional ``*`` fallback."""
    product_id = pid or "ALL"
    configured = preset_paths if preset_paths is not None else get_preset_primary_paths()
    path = configured.get(product_id)
    if path is None:
        path = configured.get("*")
    return list(path) if path is not None else None


def _primary_path_staleness_tolerance() -> timedelta:
    # Same knob as path_candidates: reuse a product primary path / durations computed up to
    # N days before the cut instead of re-deriving it (over thousands of lots) for every
    # nearby snapshot. Older data only — never future, so no time-travel. 0 = strict.
    return timedelta(days=env_num("WIP_ANALYSIS_PATH_STALENESS_TOLERANCE_DAYS", 0.0))


def _coerce_path_start_time(value: object) -> datetime | None:
    return _coerce_path_time(value, end_of_day=False)


def _coerce_path_end_time(value: object) -> datetime | None:
    return _coerce_path_time(value, end_of_day=True)


def _coerce_path_time(value: object, *, end_of_day: bool) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.max if end_of_day else time.min)
    if isinstance(value, str):
        try:
            parsed_dt = datetime.fromisoformat(value)
        except ValueError:
            try:
                parsed_date = date.fromisoformat(value)
            except ValueError:
                return None
            return datetime.combine(parsed_date, time.max if end_of_day else time.min)
        if "T" not in value and " " not in value and len(value) == 10:
            return datetime.combine(parsed_dt.date(), time.max if end_of_day else time.min)
        return parsed_dt
    return None


def _resolve_start_time_value(
    value: object,
    fallback: object,
    default_value: datetime,
) -> datetime:
    return _coerce_path_start_time(value) or _coerce_path_start_time(fallback) or default_value


def _resolve_end_time_value(
    value: object,
    fallback: object,
    default_value: datetime,
) -> datetime:
    return _coerce_path_end_time(value) or _coerce_path_end_time(fallback) or default_value


def _is_degenerate_path_window(start_time: datetime, end_time: datetime) -> bool:
    """Whether a computed path window is too degenerate to cache.

    A zero/negative-span window (``path_end_time <= path_start_time``) only arises when the
    snapshot data used for the computation collapses to a single instant (every observation at
    one ``created_at``). In that case no stage-to-stage transition exists, the solved durations
    fall back to a meaningless uniform vector, and the resulting products row is keyed by
    ``(pid, t, t)`` -- it can never satisfy any request wider than that exact instant and only
    pollutes the cache. Writers should skip persisting such windows (the live result is still
    returned to the caller).
    """
    return end_time <= start_time


def _select_widest_primary_path_row(rows: list[dict[str, object]]) -> dict[str, object] | None:
    if not rows:
        return None

    def _to_timestamp(value: object, default: float, *, end_of_day: bool = False) -> float:
        coerced = _coerce_path_time(value, end_of_day=end_of_day)
        if coerced is None:
            return default
        return coerced.timestamp()

    def _row_key(row: dict[str, object]) -> tuple[float, float, float]:
        start_ts = _to_timestamp(row.get("path_start_time"), float("inf"))
        end_ts = _to_timestamp(row.get("path_end_time"), float("-inf"), end_of_day=True)
        span = end_ts - start_ts if end_ts >= start_ts else -1.0
        return (span, end_ts, -start_ts)

    return max(rows, key=_row_key)


def _load_snapshot_bounds_in_range(
    pid: str,
    path_start_time: PathTime | None,
    path_end_time: PathTime | None,
    renderer: Callable[[str, dict[str, object]], str] | None = None,
    fetcher: Callable[[str], pl.DataFrame] | None = None,
) -> tuple[datetime | None, datetime | None]:
    query = (renderer or render_sql)(
        "get_pid_snapshot_bounds.sql.j2",
        {
            "product_id": pid,
            "snapshot_start": _coerce_path_start_time(path_start_time),
            "snapshot_end": _coerce_path_end_time(path_end_time),
        },
    )
    bounds_df = (fetcher or fetch_data)(query)
    if bounds_df.is_empty():
        return None, None

    min_created_at = bounds_df.item(0, "min_created_at")
    max_created_at = bounds_df.item(0, "max_created_at")
    min_time = min_created_at if isinstance(min_created_at, datetime) else None
    max_time = max_created_at if isinstance(max_created_at, datetime) else None
    return min_time, max_time


def _has_uncovered_snapshots_for_primary_path(
    cached_row: dict[str, object],
    required_min_time: datetime | None,
    required_max_time: datetime | None,
    staleness_tolerance: timedelta = timedelta(0),
) -> bool:
    if required_min_time is None or required_max_time is None:
        return False

    cached_start = _coerce_path_start_time(cached_row.get("path_start_time"))
    cached_end = _coerce_path_end_time(cached_row.get("path_end_time"))
    if cached_start is None or cached_end is None:
        return True

    # Relax ONLY the recent (end) bound by staleness_tolerance: a primary path/dur computed
    # up to N days before the cut is still accepted (older data only — never future, so no
    # time-travel). The start bound stays strict so a too-narrow window still recomputes.
    return cached_start > required_min_time or cached_end < required_max_time - staleness_tolerance


def _select_covering_primary_path_row(
    rows: list[dict[str, object]],
    required_min_time: datetime | None,
    required_max_time: datetime | None,
    staleness_tolerance: timedelta | None = None,
) -> dict[str, object] | None:
    if required_min_time is None or required_max_time is None:
        return _select_widest_primary_path_row(rows)

    if staleness_tolerance is None:
        staleness_tolerance = _primary_path_staleness_tolerance()
    covering_rows = [
        row
        for row in rows
        if not _has_uncovered_snapshots_for_primary_path(
            row,
            required_min_time,
            required_max_time,
            staleness_tolerance,
        )
    ]
    return _select_widest_primary_path_row(covering_rows)


def _get_primary_path_recalc_lock(
    pid: str,
    path_start_time: datetime | None,
    path_end_time: datetime | None,
) -> threading.Lock:
    key = (pid, path_start_time, path_end_time)
    with _PRIMARY_PATH_LOCKS_GUARD:
        lock = _PRIMARY_PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PRIMARY_PATH_LOCKS[key] = lock
        return lock


def _load_primary_path_from_db(
    pid: str,
    path_start_time: PathTime | None,
    path_end_time: PathTime | None,
) -> list[str]:
    resolved_start_time = _coerce_path_start_time(path_start_time)
    resolved_end_time = _coerce_path_end_time(path_end_time)
    query = render_sql(
        "get_pid_primary_paths.sql.j2",
        {
            "product_id": pid,
            "path_start_time": resolved_start_time,
            "path_end_time": resolved_end_time,
        },
    )
    result = fetch_data(query)
    if result.is_empty():
        raise ValueError("No primary_path found in DB")

    rows = result.to_dicts()
    required_min_time = None
    required_max_time = None
    if pid != "ALL" and resolved_end_time is not None:
        required_min_time, required_max_time = _load_snapshot_bounds_in_range(
            pid,
            path_start_time=resolved_start_time,
            path_end_time=resolved_end_time,
        )

    selected = _select_covering_primary_path_row(rows, required_min_time, required_max_time)
    if selected is None:
        logger.debug(
            "Primary path cache gap for pid={} required=[{}, {}], will recompute",
            pid,
            required_min_time,
            required_max_time,
        )
        raise ValueError("Primary path cache does not cover requested snapshot range")

    primary_path = selected.get("primary_path")
    if not isinstance(primary_path, list):
        raise ValueError("No primary_path found in DB")

    logger.debug(
        "Primary path cache hit pid={} range=[{}, {}]",
        pid,
        selected.get("path_start_time"),
        selected.get("path_end_time"),
    )
    return [str(stage_name) for stage_name in primary_path]


@lru_cache(maxsize=128)
def get_pid_primary_path(
    pid: Optional[str],
    path_start_time: Optional[PathTime] = None,
    path_end_time: Optional[PathTime] = None,
    force_refresh: bool = False,
) -> list[str]:
    pid = pid or "ALL"
    preset_path = get_preset_primary_path(pid)
    if preset_path is not None:
        logger.info(
            "Using preset primary path from {} for pid={}; skipping DB lookup and solver. (path is {})",
            _PRESET_PRIMARY_PATHS_ENV, pid, preset_path,
        )  # fmt:skip
        return preset_path

    resolved_start_time = _coerce_path_start_time(path_start_time)
    resolved_end_time = _coerce_path_end_time(path_end_time)
    #################################
    # 1. Primary: Try to load from DB
    #################################
    if not force_refresh:
        try:
            return _load_primary_path_from_db(pid, resolved_start_time, resolved_end_time)
        except Exception as e:
            logger.warning(f"Failed to load primary path from DB: {e}")
    else:
        logger.info("Force refresh requested for pid={}, skipping DB cache lookup", pid)

    lock = _get_primary_path_recalc_lock(pid, resolved_start_time, resolved_end_time)
    with lock:
        if not force_refresh:
            try:
                return _load_primary_path_from_db(pid, resolved_start_time, resolved_end_time)
            except Exception as e:
                logger.debug("Primary path still missing after waiting for lock: {}", e)

        #####################################
        # 2. Fallback: Calculate from scratch
        #####################################
        logger.warning("Calculating primary path from scratch")
        query = render_sql(
            "get_pid_snapshots.sql.j2",
            {
                "product_id": pid if pid != "ALL" else None,
                "include_parent_filter": False,
                "path_start_time": resolved_start_time,
                "path_end_time": resolved_end_time,
            },
        )
        raw_df = fetch_data(query)
        product_df = snapshot_to_seq(raw_df=raw_df)

        min_created_at = None
        max_created_at = None
        if not raw_df.is_empty():
            min_created_at = raw_df.select(pl.col("created_at").min()).item()
            max_created_at = raw_df.select(pl.col("created_at").max()).item()
        effective_start_time = _resolve_start_time_value(
            min_created_at,
            resolved_start_time,
            datetime.min,
        )
        effective_end_time = _resolve_end_time_value(
            max_created_at,
            resolved_end_time,
            datetime.max,
        )

        lot_ids = product_df.get_column("lot_id").to_list()
        seqs = product_df.get_column("recipe_seq").to_list()
        primary_path = get_primary_path(dict(zip(lot_ids, seqs, strict=True)))

        logger.opt(lazy=True).trace("Lot_ids: {}", lambda: format_sequence(lot_ids))
        logger.opt(lazy=True).trace("Recipe_seqs: {}", lambda: format_sequence(seqs))
        logger.opt(lazy=True).trace("Primary_path: {}", lambda: format_sequence(primary_path))

        # Insert calculated primaryPath into database.
        # Skip persisting a degenerate zero/negative-span window: single-instant snapshot data
        # would otherwise create a (pid, t, t) row with a meaningless uniform-duration profile
        # that can only ever be served to an exact-instant early-snapshot query (cache pollution
        # with invalid data). The path is still returned to the live caller.
        if _is_degenerate_path_window(effective_start_time, effective_end_time):
            logger.warning(
                "Skipping primary_path cache write for pid={}: degenerate window [{}, {}] "
                "(single-instant snapshot data); returning result without caching.",
                pid,
                effective_start_time,
                effective_end_time,
            )
        else:
            try:
                insert_query = render_sql("upsert_pid_primary_path.sql.j2")
                execute_query_async(
                    insert_query,
                    (pid, effective_start_time, effective_end_time, primary_path),
                )
                logger.info(f"Inserted primary_path for product {pid} into database")
            except Exception as insert_error:
                logger.error(f"Failed to insert primary_path into database: {insert_error}")

        return primary_path


def clear_pid_primary_path_cache() -> None:
    """Clear in-process primary path result cache."""
    get_pid_primary_path.cache_clear()
