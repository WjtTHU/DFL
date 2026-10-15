from datetime import datetime
from functools import lru_cache
from threading import Lock
from typing import Optional

import polars as pl
from loguru import logger

from wip_analysis.hwm.stage.utils.calculate_upsert import calc_stage_dur, upsert_to_cache
from wip_analysis.hwm.stage.utils.lot_progress_readiness import ensure_lot_progress_populated

from .utils.fetch_cached_durs import fetch_cached_stg_dur, fetch_cached_stg_dur_stats

_STAGE_DURATION_SCHEMA: dict[str, type[pl.DataType]] = {
    "product_id": pl.Utf8,
    "lot_id": pl.Utf8,
    "stage_name": pl.Utf8,
    "stage_type": pl.Utf8,
    "start_time": pl.Datetime,
    "end_time": pl.Datetime,
    "duration_seconds": pl.Float64,
    "duration_zscore": pl.Float64,
}

_STAGE_DURATION_STATS_SCHEMA: dict[str, type[pl.DataType]] = {
    "stage_name": pl.Utf8,
    "avg_duration_seconds": pl.Float64,
    "std_duration_seconds": pl.Float64,
    "sample_count": pl.Int64,
}

_stage_duration_stats_cache_lock = Lock()


def _normalize_stage_duration_df(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(
            {
                col_name: pl.Series([], dtype=dtype)
                for col_name, dtype in _STAGE_DURATION_SCHEMA.items()
            }
        )

    exprs: list[pl.Expr] = []
    for col_name, dtype in _STAGE_DURATION_SCHEMA.items():
        if col_name in df.columns:
            exprs.append(pl.col(col_name).cast(dtype, strict=False).alias(col_name))
        else:
            exprs.append(pl.lit(None).cast(dtype).alias(col_name))
    return df.select(exprs)


def _normalize_stage_duration_stats_df(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(
            {
                col_name: pl.Series([], dtype=dtype)
                for col_name, dtype in _STAGE_DURATION_STATS_SCHEMA.items()
            }
        )

    exprs: list[pl.Expr] = []
    for col_name, dtype in _STAGE_DURATION_STATS_SCHEMA.items():
        if col_name in df.columns:
            exprs.append(pl.col(col_name).cast(dtype, strict=False).alias(col_name))
        else:
            exprs.append(pl.lit(None).cast(dtype).alias(col_name))
    return df.select(exprs)


def _filter_stage_duration_window(
    df: pl.DataFrame,
    snapshot_start: Optional[datetime],
    snapshot_end: Optional[datetime],
) -> pl.DataFrame:
    if df.is_empty() or "start_time" not in df.columns:
        return df

    if snapshot_start is not None:
        df = df.filter(pl.col("start_time") >= snapshot_start)
    if snapshot_end is not None:
        df = df.filter(pl.col("start_time") <= snapshot_end)
    return df


@lru_cache(maxsize=512)
def _get_stage_duration_stats_cached(
    stage_name: Optional[str],
    product_id: Optional[str],
    lot_id: Optional[str],
    effective_cut_time: str,
    snapshot_start: Optional[datetime],
    snapshot_end: Optional[datetime],
) -> pl.DataFrame:
    logger.debug(
        f"Getting duration stats for stage='{stage_name}', pid='{product_id}', "
        f"lid='{lot_id}', cut_time='{effective_cut_time}', "
        f"snapshot_start='{snapshot_start}', snapshot_end='{snapshot_end}'"
    )
    df = fetch_cached_stg_dur_stats(
        stage_name,
        product_id,
        lot_id,
        effective_cut_time,
        snapshot_start=snapshot_start,
        snapshot_end=snapshot_end,
    )
    df = _normalize_stage_duration_stats_df(df)
    logger.debug("Using duration stats from lot_progress_table: {} stage rows", len(df))
    return df


def get_stage_duration_stats(
    stage_name: Optional[str],
    product_id: Optional[str],
    lot_id: Optional[str],
    cut_time: str = "3000-12-31",
    snapshot_start: Optional[datetime] = None,
    snapshot_end: Optional[datetime] = None,
) -> pl.DataFrame:
    """Get per-stage duration averages/stddevs without materializing every duration row."""
    effective_cut_time = snapshot_end.isoformat() if snapshot_end is not None else cut_time
    # functools.lru_cache is threadsafe for its data structure but not single-flight;
    # serialize misses so parallel API requests do not stampede identical DB aggregates.
    with _stage_duration_stats_cache_lock:
        return _get_stage_duration_stats_cached(
            stage_name,
            product_id,
            lot_id,
            effective_cut_time,
            snapshot_start,
            snapshot_end,
        )


def clear_stage_duration_caches() -> None:
    """Clear in-process stage duration result caches."""
    _get_stage_duration_stats_cached.cache_clear()
    get_stage_duration.cache_clear()


def _check_stage_duration_population(
    stage_name: Optional[str],
    product_id: Optional[str],
    snapshot_start: Optional[datetime],
    snapshot_end: Optional[datetime],
) -> None:
    """
    Check if stage duration data is populated in lot_progress_table for a given product/stage/window.

    Raises:
        ValueError: If lot_progress table has no data or no duration_seconds populated for the stage.
    """
    if product_id is None:
        return

    ensure_lot_progress_populated(
        product_id=product_id,
        stage_names=[stage_name] if stage_name else None,
        snapshot_start=snapshot_start,
        snapshot_end=snapshot_end,
        context="stage duration",
    )


@lru_cache(maxsize=512)
def get_stage_duration(
    stage_name: Optional[str],
    product_id: Optional[str],
    lot_id: Optional[str],
    cut_time: str = "3000-12-31",
    scored: bool = True,
    snapshot_start: Optional[datetime] = None,
    snapshot_end: Optional[datetime] = None,
    populate_cache: bool = False,
    # refresh: bool = False,
) -> pl.DataFrame:
    """
    Get the duration statistics of (a specific) stage for (a given) product. Takes raw data from lot_progress table and calculate stage_end and duration_seconds if necessary

    This function reads stage duration data directly from lot_progress_table:
    1. First tries to fetch existing results with duration_seconds calculated, would find the latest record up till cut_time (prediction included)
    2. If data is missing or incomplete, calculates from lot_progress_table
    3. Updates the duration_seconds column in lot_progress_table for future queries

    Args:
        stage_name (str, optional): Name of the stage to query. If None, returns all stages.
        product_id (str, optional): Product ID to filter by. If None, returns all products.
        lot_id (str, optional): Lot ID to filter by. If None, returns all lots.
        cut_time (str): Cut-off time to filter history data. default to use full history available.
        scored (bool): Whether to calculate a z-score for the durations.
        snapshot_start (datetime, optional): Lower bound for returned stage start times.
        snapshot_end (datetime, optional): Upper bound for returned stage start times.
        refresh (bool): If True, forces recalculation from raw data and updates lot_progress_table.

    Returns:
        out (pl.DataFrame): A DataFrame containing the duration statistics for the specified stage
        and product. Columns include
          - product_id: identifier of the product
          - lot_id: identifier of the lot
          - stage_name: name of the stage
          - stage_type: type of the stage
          - start_time: start time of the stage
          - end_time: end time of the stage
          - duration_seconds: duration of the stage in seconds
          - duration_zscore: duration of the stage in z-score (if scored=True)
    """
    effective_cut_time = snapshot_end.isoformat() if snapshot_end is not None else cut_time
    logger.debug(
        f"Getting {'scored ' if scored else ''}durations for stage='{stage_name}', "
        f"pid='{product_id}', lid='{lot_id}', cut_time='{effective_cut_time}', "
        f"snapshot_start='{snapshot_start}', snapshot_end='{snapshot_end}'"
    )

    # 1. Try to fetch from lot_progress_table first
    # if not refresh:
    try:
        df = fetch_cached_stg_dur(
            stage_name,
            product_id,
            lot_id,
            effective_cut_time,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )
    except Exception as e:
        logger.exception(f"Failed to fetch from lot_progress_table: {e}, should recalculate...")
        df = pl.DataFrame(
            {
                "product_id": pl.Series([], dtype=pl.Utf8),
                "lot_id": pl.Series([], dtype=pl.Utf8),
                "stage_name": pl.Series([], dtype=pl.Utf8),
                "stage_type": pl.Series([], dtype=pl.Utf8),
                "start_time": pl.Series([], dtype=pl.Datetime),
                "end_time": pl.Series([], dtype=pl.Datetime),
                "duration_seconds": pl.Series([], dtype=pl.Float64),
                "duration_zscore": pl.Series([], dtype=pl.Float64),
            }
        )

    if (
        not df.is_empty()
        and df.select(
            pl.col("start_time").is_not_null().all() & pl.col("end_time").is_not_null().all()
        ).item()
    ):
        logger.debug("Using data from lot_progress_table: {} rows", len(df))
    else:
        logger.debug(
            f"No valid duration data found in lot_progress_table for stage='{stage_name}',"
            f"pid='{product_id}', lid='{lot_id}', cut_time='{effective_cut_time}'"
        )

        # Check if data should be populated before proceeding
        if populate_cache:
            _check_stage_duration_population(stage_name, product_id, snapshot_start, snapshot_end)

        # 2. Data Missing / Error: Re-Calculate
        logger.debug("Calculating stage-end and duration-times from scratch")
        df = calc_stage_dur(stage_name, product_id, lot_id, effective_cut_time)

        if not df.is_empty():
            upsert_to_cache(df)

    df = _normalize_stage_duration_df(df)
    df = _filter_stage_duration_window(df, snapshot_start, snapshot_end)

    if scored and not df.is_empty():
        avg_dur = df["duration_seconds"].mean()
        std_dur = df["duration_seconds"].std()
        df = df.with_columns(
            ((pl.col("duration_seconds") - avg_dur) / std_dur).alias("duration_zscore")
        )

    return df
