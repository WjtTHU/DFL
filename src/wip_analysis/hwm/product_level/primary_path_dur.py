import threading
from datetime import datetime
from functools import lru_cache
from typing import Optional

import polars as pl
from loguru import logger

from wip_analysis.data_loader.postgres_loader import execute_query_async, fetch_data
from wip_analysis.data_loader.sql_template import render_sql

from .primary_path import (
    PathTime,
    _coerce_path_end_time,
    _coerce_path_start_time,
    _is_degenerate_path_window,
    _load_snapshot_bounds_in_range,
    _resolve_end_time_value,
    _resolve_start_time_value,
    _select_covering_primary_path_row,
    get_pid_primary_path,
    get_preset_primary_path,
)
from .utils.calc_stage_durs import solve_stage_durations

_PRIMARY_DURS_LOCKS: dict[tuple[str, datetime | None, datetime | None], threading.Lock] = {}
_PRIMARY_DURS_LOCKS_GUARD = threading.Lock()


def _load_primary_path_times(
    pid: str,
    path_start_time: Optional[PathTime],
    path_end_time: Optional[PathTime],
) -> tuple[datetime | None, datetime | None]:
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
        return None, None

    rows = result.to_dicts()
    required_min_time = None
    required_max_time = None
    if pid != "ALL" and resolved_end_time is not None:
        required_min_time, required_max_time = _load_snapshot_bounds_in_range(
            pid,
            path_start_time=resolved_start_time,
            path_end_time=resolved_end_time,
            renderer=render_sql,
            fetcher=fetch_data,
        )

    selected = _select_covering_primary_path_row(rows, required_min_time, required_max_time)
    if selected is None:
        return None, None

    return (
        _coerce_path_start_time(selected.get("path_start_time")),
        _coerce_path_end_time(selected.get("path_end_time")),
    )


def _get_primary_durs_recalc_lock(
    pid: str,
    path_start_time: datetime | None,
    path_end_time: datetime | None,
) -> threading.Lock:
    key = (pid, path_start_time, path_end_time)
    with _PRIMARY_DURS_LOCKS_GUARD:
        lock = _PRIMARY_DURS_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PRIMARY_DURS_LOCKS[key] = lock
        return lock


@lru_cache(maxsize=128)
def get_pid_primary_durs(
    pid: Optional[str],
    path_start_time: Optional[PathTime] = None,
    path_end_time: Optional[PathTime] = None,
    force_refresh: bool = False,
) -> list[float]:
    pid = pid or "ALL"
    resolved_start_time = _coerce_path_start_time(path_start_time)
    resolved_end_time = _coerce_path_end_time(path_end_time)
    preset_path = get_preset_primary_path(pid)
    primary_path_row_start_time: datetime | None = None
    primary_path_row_end_time: datetime | None = None

    def _load_durs_from_db() -> list[float]:
        nonlocal primary_path_row_start_time, primary_path_row_end_time

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
            raise ValueError("No primary_path_durations records found in DB")

        rows = result.to_dicts()
        required_min_time = None
        required_max_time = None
        if pid != "ALL" and resolved_end_time is not None:
            required_min_time, required_max_time = _load_snapshot_bounds_in_range(
                pid,
                path_start_time=resolved_start_time,
                path_end_time=resolved_end_time,
                renderer=render_sql,
                fetcher=fetch_data,
            )

        if preset_path is not None:
            rows = [
                row
                for row in rows
                if isinstance(row.get("primary_path"), list)
                and [str(stage_name) for stage_name in row["primary_path"]] == preset_path
            ]

        selected = _select_covering_primary_path_row(rows, required_min_time, required_max_time)
        if selected is None:
            logger.debug(
                "Primary path duration cache gap for pid={} required=[{}, {}], will recompute",
                pid,
                required_min_time,
                required_max_time,
            )
            raise ValueError("Primary path duration cache does not cover requested time range")

        primary_path_row_start_time = _coerce_path_start_time(selected.get("path_start_time"))
        primary_path_row_end_time = _coerce_path_end_time(selected.get("path_end_time"))
        durations = selected.get("primary_path_durations")
        if durations is None:
            raise ValueError("primary_path_durations is null in DB")
        if not isinstance(durations, list):
            raise ValueError("primary_path_durations is not an array in DB")
        durations_list = [float(v) for v in durations]
        primary_path = selected.get("primary_path")
        if not isinstance(primary_path, list):
            raise ValueError("No primary_path found in DB")
        path_len = len(primary_path)
        if not durations_list and path_len > 0:
            raise ValueError("primary_path_durations is empty in DB")
        if len(durations_list) != path_len:
            raise ValueError(
                "primary_path_durations length mismatch in DB: "
                f"path_len={path_len}, durations_len={len(durations_list)}"
            )
        return durations_list

    if not force_refresh:
        try:
            return _load_durs_from_db()
        except Exception as e:
            logger.debug("Error loading primary path intervals from DB: {}", e)
    else:
        logger.info("Force refresh requested for pid={}, skipping DB duration lookup", pid)

    lock = _get_primary_durs_recalc_lock(pid, resolved_start_time, resolved_end_time)
    with lock:
        if not force_refresh:
            try:
                return _load_durs_from_db()
            except Exception as e:
                logger.debug("Primary path intervals still missing after waiting for lock: {}", e)

        logger.warning("Calculating primary path intervals from scratch")

        primary_path = get_pid_primary_path(
            pid,
            path_start_time=resolved_start_time,
            path_end_time=resolved_end_time,
            force_refresh=force_refresh,
        )
        if preset_path is None and (
            primary_path_row_start_time is None or primary_path_row_end_time is None
        ):
            path_row_start_time, path_row_end_time = _load_primary_path_times(
                pid,
                resolved_start_time,
                resolved_end_time,
            )
            primary_path_row_start_time = path_row_start_time or primary_path_row_start_time
            primary_path_row_end_time = path_row_end_time or primary_path_row_end_time

        stg_to_idx = {stg: i for i, stg in enumerate(primary_path)}
        duration_start_time = primary_path_row_start_time or resolved_start_time
        duration_end_time = primary_path_row_end_time or resolved_end_time

        query = render_sql(
            "get_pid_snapshots.sql.j2",
            {
                "product_id": pid,
                "include_parent_filter": True,
                "path_start_time": duration_start_time,
                "path_end_time": duration_end_time,
            },
        )
        raw_df = fetch_data(query)
        res = solve_stage_durations(raw_df, stg_to_idx)
        durs: list[float] = res.select(pl.col("duration")).to_series().to_list()

        min_created_at = None
        max_created_at = None
        if not raw_df.is_empty():
            min_created_at = raw_df.select(pl.col("created_at").min()).item()
            max_created_at = raw_df.select(pl.col("created_at").max()).item()

        effective_start_time = primary_path_row_start_time or _resolve_start_time_value(
            min_created_at,
            duration_start_time,
            datetime.min,
        )
        effective_end_time = primary_path_row_end_time or _resolve_end_time_value(
            max_created_at,
            duration_end_time,
            datetime.max,
        )

        # Insert calculated primary_path_durations into database.
        # Skip a degenerate zero/negative-span window: single-instant snapshot data yields no
        # solvable durations (uniform fallback), so caching it under a (pid, t, t) key only
        # pollutes the products table with invalid data. See get_pid_primary_path.
        if preset_path is not None:
            logger.info(
                "Skipping primary_path_durations cache write for pid={}: primary path is "
                "configured by WIP_ANALYSIS_PRESET_PRIMARY_PATHS",
                pid,
            )
        elif _is_degenerate_path_window(effective_start_time, effective_end_time):
            logger.warning(
                "Skipping primary_path_durations cache write for pid={}: degenerate window "
                "[{}, {}] (single-instant snapshot data); returning result without caching.",
                pid,
                effective_start_time,
                effective_end_time,
            )
        else:
            try:
                insert_query = render_sql("upsert_pid_primary_path_durations.sql.j2")
                execute_query_async(
                    insert_query,
                    (pid, effective_start_time, effective_end_time, durs),
                )
                logger.info(f"Inserted primary_path_durations for product {pid} into database")
            except Exception as insert_error:
                logger.error(
                    f"Failed to insert primary_path_durations into database: {insert_error}"
                )

        return durs


def clear_pid_primary_durs_cache() -> None:
    """Clear in-process primary path duration result cache."""
    get_pid_primary_durs.cache_clear()
