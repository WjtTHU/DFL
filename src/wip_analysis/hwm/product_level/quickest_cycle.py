from datetime import datetime

import polars as pl
from loguru import logger

from wip_analysis.data_loader.postgres_loader import execute_query_async, fetch_data
from wip_analysis.data_loader.sql_template import render_sql

from .primary_path import (
    PathTime,
    _coerce_path_end_time,
    _coerce_path_start_time,
    _is_degenerate_path_window,
    _resolve_end_time_value,
    _resolve_start_time_value,
)


def get_pid_quickest_cycle(
    pid: str,
    path_start_time: PathTime | None = None,
    path_end_time: PathTime | None = None,
) -> float:
    resolved_start_time = _coerce_path_start_time(path_start_time)
    resolved_end_time = _coerce_path_end_time(path_end_time)
    try:
        query = render_sql(
            "get_pid_min_completion_time.sql.j2",
            {
                "product_id": pid,
                "path_start_time": resolved_start_time,
                "path_end_time": resolved_end_time,
            },
        )
        result = fetch_data(query)
        if result.is_empty():
            raise ValueError("No min_completion_time records found in DB")
        min_time = result.item(0, "min_completion_time")
        if min_time is None:
            raise ValueError("min_completion_time is null in DB")
        return min_time

    except Exception as e:
        logger.error(f"Error loading min completion time from DB: {e}")

    logger.warning("Calculating min completion time from scratch")
    query = render_sql(
        "get_pid_snapshots.sql.j2",
        {
            "product_id": pid,
            "include_parent_filter": True,
            "path_start_time": resolved_start_time,
            "path_end_time": resolved_end_time,
        },
    )
    df = fetch_data(query)
    min_time = (
        df.group_by("lot_id")
        .agg(
            (pl.col("created_at").max() - pl.col("created_at").min())
            .dt.total_days()
            .alias("duration_days")
        )
        .select(pl.col("duration_days").min())
        .item()
    )
    logger.info(f"Min completion time for {pid}: {min_time} days.")

    min_created_at = None
    max_created_at = None
    if not df.is_empty():
        min_created_at = df.select(pl.col("created_at").min()).item()
        max_created_at = df.select(pl.col("created_at").max()).item()

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

    # Insert calculated min_completion_time into database.
    # Skip a degenerate zero/negative-span window: a (pid, t, t) row from single-instant
    # snapshot data is unusable cache pollution (same guard as the primary_path writers).
    if _is_degenerate_path_window(effective_start_time, effective_end_time):
        logger.warning(
            "Skipping min_completion_time cache write for pid={}: degenerate window [{}, {}] "
            "(single-instant snapshot data); returning result without caching.",
            pid,
            effective_start_time,
            effective_end_time,
        )
    else:
        try:
            insert_query = render_sql("upsert_pid_min_completion_time.sql.j2")
            execute_query_async(
                insert_query,
                (pid, effective_start_time, effective_end_time, min_time),
            )
            logger.info(f"Inserted min_completion_time for product {pid} into database")
        except Exception as e:
            logger.exception(f"Failed to insert min_completion_time into database: {e}")

    return min_time
