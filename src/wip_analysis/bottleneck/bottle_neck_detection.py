from datetime import datetime
from typing import Any, Optional

import polars as pl
from loguru import logger

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql
from wip_analysis.hwm.product_level.primary_path import get_pid_primary_path


def detect_bottleneck_by_hold_time(
    start_time: datetime,
    end_time: datetime,
    product_ids: Optional[list[str]] = None,
    min_wafer_count: int = 10,
    min_hold_days: int = 3,
) -> list[dict[str, Any]]:
    """
    Detect bottleneck stages based on hold time.

    For each product ID, if the number of wafers in hold status at the same stage
    consecutively reaches the specified number of days, it is considered a bottleneck.

    Args:
        start_time: Start time of the analysis data time range
        end_time: End time of the analysis data time range
        product_ids: List of product IDs. If None, analyze all products in WIP within the time range
        min_wafer_count: Minimum number of wafers in hold status
        min_hold_days: Minimum number of consecutive hold days

    Returns:
        List of bottleneck information for each product_id, each element contains:
        - product_id: Product ID
        - primary_path: Primary process route
        - bottleneck_stages: List of bottleneck stages, each element contains:
            - stage: Stage name
            - reason: Bottleneck reason description (indicates wafer qty in hold status
              at the corresponding stage and consecutive hold days)
    """
    logger.info(
        f"Detecting bottlenecks by hold time: start_time={start_time}, end_time={end_time}, "
        f"product_ids={product_ids}, min_wafer_count={min_wafer_count}, min_hold_days={min_hold_days}"
    )

    # Query data within the time range
    query = render_sql(
        "get_lid_snapshots_ranged.sql.j2",
        ctx={
            "snapshot_start": start_time,
            "snapshot_cut": end_time,
        },
    )
    df = fetch_data(query)

    if df.is_empty():
        logger.warning("No data found in the specified time range")
        return []

    # Filter by product_ids if specified
    if product_ids is not None:
        df = df.filter(pl.col("product_id").is_in(product_ids))
    else:
        # Get all product_ids in WIP within the time range
        product_ids = df.select("product_id").unique()["product_id"].to_list()

    if df.is_empty():
        logger.warning("No data found for specified product_ids")
        return []

    # Analyze each product_id
    results = []

    for product_id in product_ids:
        product_df = df.filter(pl.col("product_id") == product_id)

        if product_df.is_empty():
            continue

        # Get primary process route
        try:
            primary_path = get_pid_primary_path(product_id, path_end_time=end_time)
        except Exception as e:
            logger.warning(f"Failed to get primary path for product {product_id}: {e}")
            primary_path = []

        # Group by lot_id and stage, calculate hold days for each lot at each stage
        # If adjacent records are both in hold status, consider the time between them as hold time
        lot_stage_all = product_df.select(
            [
                "lot_id",
                "stage",
                "lot_status",
                "current_wafer_qty",
                pl.col("created_at").alias("timestamp"),
            ]
        ).sort(["lot_id", "stage", "timestamp"])

        # Mark records that are in hold status
        lot_stage_with_hold_flag = lot_stage_all.with_columns(
            (pl.col("lot_status").str.to_lowercase() == "hold").alias("is_hold")
        )

        # Calculate consecutive hold days for each lot at each stage
        # For adjacent records that are both in hold, sum up the time intervals between them
        lot_stage_with_intervals = lot_stage_with_hold_flag.with_columns(
            [
                # Calculate time difference from previous record
                (pl.col("timestamp") - pl.col("timestamp").shift(1).over(["lot_id", "stage"]))
                .dt.total_seconds()
                .alias("time_diff_seconds"),
                # Check if previous record was also in hold
                pl.col("is_hold").shift(1).over(["lot_id", "stage"]).alias("prev_is_hold"),
            ]
        ).with_columns(
            [
                # Mark if this is the start of a hold segment
                # (no previous record, or previous record was not hold, or gap > threshold)
                ((pl.col("prev_is_hold").is_null()) | (~pl.col("prev_is_hold")))
                .cast(pl.Int32)
                .cum_sum()
                .over(["lot_id", "stage"])
                .alias("hold_segment"),
            ]
        )

        # Filter to only hold records and calculate total hold days for each segment
        # Sum up time intervals between adjacent hold records
        lot_stage_hold_segments = (
            lot_stage_with_intervals.filter(pl.col("is_hold"))
            .filter(
                # Only count intervals where previous record was also hold
                (pl.col("prev_is_hold")) & (pl.col("time_diff_seconds").is_not_null())
            )
            .group_by(["lot_id", "stage", "hold_segment"], maintain_order=False)
            .agg(
                [
                    pl.col("current_wafer_qty").first().alias("wafer_qty"),
                    # Sum of time intervals between adjacent hold records in this segment
                    (pl.col("time_diff_seconds").sum() / 86400.0).alias("hold_days"),
                ]
            )
            .filter(pl.col("hold_days") >= min_hold_days)
        )

        if lot_stage_hold_segments.is_empty():
            continue

        # For each lot-stage combination, keep only the longest consecutive hold segment
        lot_stage_hold_days = lot_stage_hold_segments.group_by(
            ["lot_id", "stage"], maintain_order=False
        ).agg(
            [
                pl.col("wafer_qty").first().alias("wafer_qty"),
                pl.col("hold_days").max().alias("hold_days"),
            ]
        )

        if lot_stage_hold_days.is_empty():
            continue

        # Group by stage, aggregate total wafer count in hold for each stage
        stage_hold_stats = (
            lot_stage_hold_days.group_by("stage", maintain_order=False)
            .agg(
                [
                    pl.sum("wafer_qty").alias("total_wafer_qty"),
                    pl.col("hold_days").max().alias("max_hold_days"),
                    pl.col("lot_id").count().alias("lot_count"),
                ]
            )
            .filter(pl.col("total_wafer_qty") >= min_wafer_count)
            .sort("total_wafer_qty", descending=True)
        )

        if stage_hold_stats.is_empty():
            continue

        # Build bottleneck stage information
        bottleneck_stages = []
        for row in stage_hold_stats.iter_rows(named=True):
            bottleneck_stages.append(
                {
                    "stage": row["stage"],
                    "reason": f"{row['total_wafer_qty']} wafers at stage {row['stage']} "
                    f"consecutively held for {row['max_hold_days']} days",
                }
            )

        if bottleneck_stages:
            results.append(
                {
                    "product_id": product_id,
                    "primary_path": primary_path,
                    "bottleneck_stages": bottleneck_stages,
                }
            )

    logger.info(f"Found bottlenecks for {len(results)} products")
    return results


def detect_bottleneck_by_daily_qty_variance(
    start_time: datetime,
    end_time: datetime,
    product_ids: Optional[list[str]] = None,
    processing_statuses: list[str] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Detect bottleneck stages based on daily processing quantity variance.

    For each product ID, analyze the daily processing quantity at each stage,
    calculate the variance of daily processing quantities, and consider the top K
    stages with the highest variance as bottleneck stages.

    Args:
        start_time: Start time of the analysis data time range
        end_time: End time of the analysis data time range
        product_ids: List of product IDs. If None, analyze all products in WIP within the time range
        processing_statuses: Set of statuses considered as "processing",
            can be ["Active", "Bank"] or just ["Active"]
        top_k: Parameter K, return the top K stages with highest variance

    Returns:
        List of bottleneck information for each product_id, each element contains:
        - product_id: Product ID
        - primary_path: Primary process route
        - bottleneck_stages: List of bottleneck stages, each element contains:
            - stage: Stage name
            - reason: Bottleneck reason description (indicates its ranking in top K)
    """
    processing_statuses = processing_statuses or ["Active"]

    logger.info(
        f"Detecting bottlenecks by variance: start_time={start_time}, end_time={end_time}, "
        f"product_ids={product_ids}, processing_statuses={processing_statuses}, top_k={top_k}"
    )

    # Query data within the time range
    query = render_sql(
        "get_lid_snapshots_ranged.sql.j2",
        ctx={
            "snapshot_start": start_time,
            "snapshot_cut": end_time,
            "include_parent_filter": True,
        },
    )
    df = fetch_data(query)

    if df.is_empty():
        logger.warning("No data found in the specified time range")
        return []

    # Filter by product_ids if specified
    if product_ids is not None:
        df = df.filter(pl.col("product_id").is_in(product_ids))
    else:
        # Get all product_ids in WIP within the time range
        product_ids = df.select("product_id").unique()["product_id"].to_list()

    if df.is_empty():
        logger.warning("No data found for specified product_ids")
        return []

    # Filter data with "processing" status
    processing_df = df.filter(
        pl.col("lot_status").str.to_lowercase().is_in([s.lower() for s in processing_statuses])
    )

    if processing_df.is_empty():
        logger.info("No processing status data found")
        return []

    # Analyze each product_id
    results = []

    for product_id in product_ids:
        product_df = processing_df.filter(pl.col("product_id") == product_id)

        if product_df.is_empty():
            continue

        # Get primary process route
        try:
            primary_path = get_pid_primary_path(product_id, path_end_time=end_time)
        except Exception as e:
            logger.warning(f"Failed to get primary path for product {product_id}: {e}")
            primary_path = []

        # Group by date and stage, aggregate daily processing quantity (wafer count)
        # First deduplicate by date, lot_id, and stage to ensure each lot is counted only once per day
        daily_stage_qty = (
            product_df.select(
                [
                    pl.col("created_at").cast(pl.Date).alias("date"),
                    "lot_id",
                    "stage",
                    "current_wafer_qty",
                ]
            )
            .unique(subset=["date", "lot_id", "stage"], keep="first")
            .group_by(["date", "stage"], maintain_order=False)
            .agg(pl.sum("current_wafer_qty").alias("daily_wafer_qty"))
        )

        if daily_stage_qty.is_empty():
            continue

        # Calculate variance of daily processing quantity for each stage
        stage_variance = (
            daily_stage_qty.group_by("stage", maintain_order=False)
            .agg(pl.col("daily_wafer_qty").var().alias("variance"))
            .filter(pl.col("variance").is_not_null())
            .sort("variance", descending=True)
            .head(top_k)
        )

        if stage_variance.is_empty():
            continue

        # Build bottleneck stage information
        bottleneck_stages = []
        for rank, row in enumerate(stage_variance.iter_rows(named=True), start=1):
            bottleneck_stages.append(
                {
                    "stage": row["stage"],
                    "reason": f"Ranked #{rank} in processing quantity variance "
                    f"(variance={row['variance']:.2f})",
                }
            )

        if bottleneck_stages:
            results.append(
                {
                    "product_id": product_id,
                    "primary_path": primary_path,
                    "bottleneck_stages": bottleneck_stages,
                }
            )

    logger.info(f"Found bottlenecks for {len(results)} products")
    return results


def detect_bottleneck_by_throughput_var_and_load_rate(
    start_time: datetime,
    end_time: datetime,
    product_ids: Optional[list[str]] = None,
    top_k: int = 5,
    min_daily_throughput: int = 1,
    min_days: int = 5,
    min_utilization: float = 0.5,
    full_load_threshold: float = 0.85,
    weight_variance: float = 0.5,
    weight_utilization: float = 0.25,
    weight_full_load: float = 0.25,
) -> list[dict[str, Any]]:
    """
    Detect bottleneck stages based on daily throughput variance and load rate.

    Throughput is defined at lot level: if the next snapshot of a lot is at a
    different stage than the current snapshot, the previous stage is considered
    to have produced the current wafer quantity at the time of the next snapshot.

    For each product ID, this function:
    1) Computes daily throughput per stage.
    2) Aggregates per-stage statistics: mean, max, variance of daily throughput.
    3) Derives utilization (mean / max) and full load rate
       (#days with throughput >= full_load_threshold * max / total days).
    4) Applies basic filters and computes a weighted score based on
       variance, utilization and full load rate, then selects top_k stages.

    Args:
        start_time: Start time of the analysis data time range.
        end_time: End time of the analysis data time range.
        product_ids: List of product IDs. If None, analyze all products in WIP within the time range.
        top_k: Number of bottleneck stages to return per product.
        min_daily_throughput: Minimum max daily throughput for a stage to be considered.
        min_days: Minimum number of days with throughput records for a stage.
        min_utilization: Minimum average utilization (mean / max) for a stage.
        full_load_threshold: Threshold (0-1) to judge a day as "full load" relative to max throughput.
        weight_variance: Weight of normalized variance in the final score.
        weight_utilization: Weight of utilization in the final score.
        weight_full_load: Weight of full load rate in the final score.

    Returns:
        List of bottleneck information for each product_id, each element contains:
        - product_id: Product ID
        - primary_path: Primary process route
        - bottleneck_stages: List of bottleneck stages, each element contains:
            - stage: Stage name
            - reason: Bottleneck reason description including key metrics
    """
    logger.info(
        "Detecting bottlenecks by throughput variance and load rate: "
        f"start_time={start_time}, end_time={end_time}, product_ids={product_ids}, "
        f"top_k={top_k}, min_daily_throughput={min_daily_throughput}, "
        f"min_days={min_days}, min_utilization={min_utilization}, "
        f"full_load_threshold={full_load_threshold}, "
        f"weights=({weight_variance}, {weight_utilization}, {weight_full_load})"
    )

    # Query data within the time range
    query = render_sql(
        "get_lid_snapshots_ranged.sql.j2",
        ctx={
            "snapshot_start": start_time,
            "snapshot_cut": end_time,
        },
    )
    df = fetch_data(query)

    if df.is_empty():
        logger.warning("No data found in the specified time range")
        return []

    # Filter by product_ids if specified
    if product_ids is not None:
        df = df.filter(pl.col("product_id").is_in(product_ids))
    else:
        # Get all product_ids in WIP within the time range
        product_ids = df.select("product_id").unique()["product_id"].to_list()

    if df.is_empty():
        logger.warning("No data found for specified product_ids")
        return []

    results: list[dict[str, Any]] = []

    for product_id in product_ids:
        product_df = df.filter(pl.col("product_id") == product_id)

        if product_df.is_empty():
            continue

        # Get primary process route
        try:
            primary_path = get_pid_primary_path(product_id, path_end_time=end_time)
        except Exception as e:
            logger.warning(f"Failed to get primary path for product {product_id}: {e}")
            primary_path = []

        # Build lot-level time series and derive throughput events.
        # Throughput event is triggered when a lot moves from one stage to another.
        lot_timeline = product_df.select(
            [
                "lot_id",
                "stage",
                "current_wafer_qty",
                pl.col("created_at").cast(pl.Date).alias("date"),
            ]
        ).sort(["lot_id", "date"])

        if lot_timeline.is_empty():
            continue

        lot_with_next = lot_timeline.with_columns(
            [
                pl.col("stage").shift(-1).over("lot_id").alias("next_stage"),
                pl.col("date").shift(-1).over("lot_id").alias("next_date"),
            ]
        )

        throughput_events = lot_with_next.filter(
            pl.col("next_date").is_not_null() & (pl.col("next_stage") != pl.col("stage"))
        ).select(
            [
                pl.col("next_date").alias("date"),
                pl.col("stage").alias("stage"),
                pl.col("current_wafer_qty").alias("throughput_qty"),
            ]
        )

        if throughput_events.is_empty():
            continue

        # Aggregate daily throughput per stage
        daily_stage_throughput = throughput_events.group_by(
            ["date", "stage"], maintain_order=False
        ).agg(pl.sum("throughput_qty").alias("daily_throughput"))

        if daily_stage_throughput.is_empty():
            continue

        # Per-stage statistics: mean, max, variance, days_count
        stage_stats = daily_stage_throughput.group_by("stage", maintain_order=False).agg(
            [
                pl.col("daily_throughput").mean().alias("mean_daily_throughput"),
                pl.col("daily_throughput").max().alias("max_daily_throughput"),
                pl.col("daily_throughput").var().alias("throughput_variance"),
                pl.count().alias("days_count"),
            ]
        )

        if stage_stats.is_empty():
            continue

        # Compute full load rate for each stage:
        # fraction of days where daily_throughput >= full_load_threshold * max_daily_throughput
        stage_max = stage_stats.select(["stage", "max_daily_throughput"])
        throughput_with_max = daily_stage_throughput.join(
            stage_max, on="stage", how="left"
        ).with_columns(
            (
                pl.col("daily_throughput") >= pl.col("max_daily_throughput") * full_load_threshold
            ).alias("is_full_load")
        )

        full_load_stats = throughput_with_max.group_by("stage", maintain_order=False).agg(
            [
                pl.col("is_full_load").cast(pl.Float64).mean().alias("full_load_rate"),
            ]
        )

        stage_stats = stage_stats.join(full_load_stats, on="stage", how="left")

        # Derive utilization and apply basic filters.
        stage_stats = stage_stats.with_columns(
            (
                pl.when(pl.col("max_daily_throughput") > 0)
                .then(pl.col("mean_daily_throughput") / pl.col("max_daily_throughput"))
                .otherwise(0.0)
            ).alias("utilization")
        ).filter(
            (pl.col("max_daily_throughput") >= min_daily_throughput)
            & (pl.col("days_count") >= min_days)
            & (pl.col("utilization") >= min_utilization)
            & pl.col("throughput_variance").is_not_null()
        )

        if stage_stats.is_empty():
            continue

        # Normalize variance across stages to [0, 1].
        var_min = stage_stats.select(pl.col("throughput_variance").min()).item()
        var_max = stage_stats.select(pl.col("throughput_variance").max()).item()

        if var_max is None or var_min is None:
            continue

        if var_max == var_min:
            stage_stats = stage_stats.with_columns(pl.lit(1.0).alias("variance_norm"))
        else:
            stage_stats = stage_stats.with_columns(
                ((pl.col("throughput_variance") - var_min) / (var_max - var_min + 1e-6)).alias(
                    "variance_norm"
                )
            )

        # Compute final score using weighted sum of variance_norm, utilization and full_load_rate.
        stage_stats = stage_stats.with_columns(
            (
                weight_variance * pl.col("variance_norm")
                + weight_utilization * pl.col("utilization")
                + weight_full_load * pl.col("full_load_rate")
            ).alias("bottleneck_score")
        )

        stage_stats = stage_stats.sort("bottleneck_score", descending=True).head(top_k)

        if stage_stats.is_empty():
            continue

        bottleneck_stages: list[dict[str, Any]] = []
        for rank, row in enumerate(stage_stats.iter_rows(named=True), start=1):
            bottleneck_stages.append(
                {
                    "stage": row["stage"],
                    "reason": (
                        f"Ranked #{rank} by throughput bottleneck score "
                        f"(score={row['bottleneck_score']:.3f}, "
                        f"mean_daily_throughput={row['mean_daily_throughput']:.1f}, "
                        f"max_daily_throughput={row['max_daily_throughput']:.1f}, "
                        f"variance={row['throughput_variance']:.2f}, "
                        f"utilization={row['utilization']:.2f}, "
                        f"full_load_rate={row['full_load_rate']:.2f})"
                    ),
                }
            )

        if bottleneck_stages:
            results.append(
                {
                    "product_id": product_id,
                    "primary_path": primary_path,
                    "bottleneck_stages": bottleneck_stages,
                }
            )

    logger.info(f"Found bottlenecks (throughput-based) for {len(results)} products")
    return results
