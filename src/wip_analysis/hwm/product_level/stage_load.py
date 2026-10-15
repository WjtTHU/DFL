"""
Calculate stage load ratios by normalizing rolling wafer sums with stage durations.

This module processes stage lot counts and actual wafer sums from
`get_pid_stage_stats`, calculates rolling sums over various time windows
(7d, 1m, 3m, 6m, 1y), then divides the rolling wafer sums by the average
stage duration to get a normalized wafer-load metric.
"""

from datetime import datetime
from typing import Any, Optional

import polars as pl
from loguru import logger

from wip_analysis.hwm.stage import get_stage_duration
from wip_analysis.hwm.stage.utils.lot_progress_readiness import ensure_lot_progress_populated

from .primary_path import get_pid_primary_path
from .stage_lot_cnt import get_pid_stage_stats


def get_pid_stage_load(
    product_id: Optional[str],
    snapshot_start: Optional[datetime] = None,
    snapshot_end: Optional[datetime] = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """
    Calculate rolling wafer-load ratios for each stage in a product's primary path.

    For each stage and snapshot_cut combination, this function:
    1. Uses stage lot counts together with actual stage wafer sums
    2. Calculates rolling sums over multiple time windows
    3. Divides by the average stage duration to normalize the load
    4. Returns results organized by snapshot_cut and stage

    Args:
        product_id: The product ID to analyze. If None, uses "ALL"
        snapshot_start: Optional lower bound for `snapshot_cut`
        snapshot_end: Optional upper bound for `snapshot_cut`
        raise_on_error: If True, raises exceptions instead of returning empty fallback output.

    Returns:
        Dictionary with structure:
        {
            "product_id": str,
            "stage_order": list[str],  # Primary path stage names
            "load_ratios": [
                {
                    "snapshot_cut": datetime,
                    "stage_metrics": {
                        "stage_name": {
                            "avg_duration_seconds": float,
                            "lot_counts": [int, ...],
                            "wafer_sums": [int, ...],
                            "rolling_7d": [float, ...],  # Wafer/day calendar time series
                            "rolling_1m": [float, ...],
                            "rolling_3m": [float, ...],
                            "rolling_6m": [float, ...],
                            "rolling_1y": [float, ...]
                        },
                        ...
                    },
                    "calendar_times": [datetime, ...]  # Aligned with rolling arrays
                }
            ]
        }
    """
    product_id = product_id or "ALL"
    duration_product_id = None if product_id == "ALL" else product_id
    logger.info(
        f"Calculating stage load ratios for product {product_id}, "
        f"snapshot_start={snapshot_start}, snapshot_end={snapshot_end}"
    )

    try:
        primary_path = get_pid_primary_path(
            product_id,
            path_start_time=snapshot_start,
            path_end_time=snapshot_end,
        )
        if not primary_path:
            logger.warning(f"No primary path found for product {product_id}")
            return {
                "product_id": product_id,
                "stage_order": [],
                "load_ratios": [],
            }

        stage_stats = get_pid_stage_stats(
            product_id,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
            raise_on_error=True,
        )

        if not stage_stats:
            logger.warning(f"No stage statistics found for product {product_id}")
            return {
                "product_id": product_id,
                "stage_order": primary_path,
                "load_ratios": [],
            }

        # 1. Get stage duration averages for all stages in the primary path
        stage_durations_data = []
        for stage_name in primary_path:
            try:
                dur_df = get_stage_duration(
                    stage_name=stage_name,
                    product_id=duration_product_id,
                    lot_id=None,
                    cut_time="3000-12-31",
                    scored=False,
                    snapshot_start=snapshot_start,
                    snapshot_end=snapshot_end,
                )
                if not dur_df.is_empty() and "duration_seconds" in dur_df.columns:
                    avg_duration = dur_df["duration_seconds"].mean()
                else:
                    avg_duration = None  # Handle missing data later
                stage_durations_data.append(
                    {"stage_name": stage_name, "avg_duration_seconds": avg_duration}
                )
            except Exception as e:
                logger.warning(f"Error getting duration for stage {stage_name}: {e}")
                stage_durations_data.append(
                    {"stage_name": stage_name, "avg_duration_seconds": None}
                )

        missing_duration_stages = [
            row["stage_name"]
            for row in stage_durations_data
            if row.get("avg_duration_seconds") is None
        ]

        if missing_duration_stages and duration_product_id is not None:
            ensure_lot_progress_populated(
                product_id=duration_product_id,
                stage_names=missing_duration_stages,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
                context="stage load",
            )

        df_durations = pl.DataFrame(stage_durations_data)
        if df_durations.is_empty():
            df_durations = pl.DataFrame(
                {
                    "stage_name": pl.Series([], dtype=pl.Utf8),
                    "avg_duration_seconds": pl.Series([], dtype=pl.Float64),
                }
            )
        else:
            df_durations = df_durations.with_columns(
                pl.col("stage_name").cast(pl.Utf8),
                pl.col("avg_duration_seconds").cast(pl.Float64),
            )

        # 2. Prepare the main DataFrame
        df = (
            pl.DataFrame(stage_stats)
            .with_columns(pl.lit(primary_path).alias("stage_name"))
            .explode(["stage_lot_cnt", "stage_wafer_sum", "stage_name"])
            .rename({"stage_lot_cnt": "lot_count", "stage_wafer_sum": "wafer_sum"})
            .with_columns(
                pl.col("lot_count").cast(pl.Int64),
                pl.col("wafer_sum").cast(pl.Int64),
            )
            .join(df_durations, on="stage_name", how="left")
            .sort(["snapshot_cut", "stage_name", "calendar_time"])
        )

        # 3. Calculate all rolling sums and normalize
        windows = {
            "rolling_7d": "7d",
            "rolling_1m": "1mo",
            "rolling_3m": "3mo",
            "rolling_6m": "6mo",
            "rolling_1y": "1y",
        }

        # Calculate rolling metrics for each window period
        df_list = []
        for name, period in windows.items():
            # Use DataFrame.rolling() method with group_by for each snapshot_cut and stage_name
            rolling_result = (
                df.rolling(
                    index_column="calendar_time",
                    period=period,
                    group_by=["snapshot_cut", "stage_name"],
                )
                .agg(
                    [
                        pl.col("wafer_sum").sum().alias(f"{name}_wafer_sum"),
                        pl.col("avg_duration_seconds").first(),
                    ]
                )
                .with_columns(
                    [
                        # Calculate the load ratio: wafer_count / (avg_duration_seconds / 86400)
                        pl.when(pl.col("avg_duration_seconds") > 0)
                        .then(
                            (
                                pl.col(f"{name}_wafer_sum")
                                / (pl.col("avg_duration_seconds") / 86400.0)
                            ).round(4)
                        )
                        .otherwise(None)
                        .alias(name)
                    ]
                )
                .select(["snapshot_cut", "stage_name", "calendar_time", name])
            )
            df_list.append(rolling_result)

        # Join all rolling results together
        df_ratios = df
        for rolling_df in df_list:
            df_ratios = df_ratios.join(
                rolling_df, on=["snapshot_cut", "stage_name", "calendar_time"], how="left"
            )

        # 4. Structure the output
        load_ratios = []
        for snapshot_cut, snapshot_df in df_ratios.group_by("snapshot_cut"):
            snapshot_cut_value = (
                snapshot_cut[0] if isinstance(snapshot_cut, tuple) else snapshot_cut
            )
            stage_metrics: dict[str, Any] = {}

            # Build a complete calendar axis per snapshot to avoid truncation by any one stage.
            calendar_times = (
                snapshot_df.select("calendar_time").unique().sort("calendar_time")["calendar_time"]
            ).to_list()
            axis_df = pl.DataFrame({"calendar_time": calendar_times})

            for stage_name in primary_path:
                stage_df = (
                    snapshot_df.filter(pl.col("stage_name") == stage_name)
                    .select(
                        "calendar_time",
                        "avg_duration_seconds",
                        "lot_count",
                        "wafer_sum",
                        *windows.keys(),
                    )
                    .sort("calendar_time")
                )

                aligned_df = (
                    axis_df.join(stage_df, on="calendar_time", how="left")
                    .with_columns(
                        pl.col("lot_count").fill_null(0),
                        pl.col("wafer_sum").fill_null(0),
                    )
                    .with_columns(
                        pl.col("lot_count").cast(pl.Int64),
                        pl.col("wafer_sum").cast(pl.Int64),
                    )
                )

                non_null_duration = aligned_df.select(pl.col("avg_duration_seconds").drop_nulls())
                avg_duration_seconds = (
                    non_null_duration.item(0, "avg_duration_seconds")
                    if not non_null_duration.is_empty()
                    else None
                )

                stage_metrics[stage_name] = {
                    "avg_duration_seconds": avg_duration_seconds,
                    "lot_counts": aligned_df["lot_count"].to_list(),
                    "wafer_sums": aligned_df["wafer_sum"].to_list(),
                    "rolling_7d": aligned_df["rolling_7d"].to_list(),
                    "rolling_1m": aligned_df["rolling_1m"].to_list(),
                    "rolling_3m": aligned_df["rolling_3m"].to_list(),
                    "rolling_6m": aligned_df["rolling_6m"].to_list(),
                    "rolling_1y": aligned_df["rolling_1y"].to_list(),
                }

            load_ratios.append(
                {
                    "snapshot_cut": snapshot_cut_value,
                    "stage_metrics": stage_metrics,
                    "calendar_times": calendar_times,
                }
            )

        load_ratios.sort(key=lambda row: row["snapshot_cut"])

        result = {
            "product_id": product_id,
            "stage_order": primary_path,
            "load_ratios": load_ratios,
        }

        logger.info(
            f"Calculated load ratios for {len(load_ratios)} snapshots, "
            f"{len(primary_path)} stages for product {product_id}"
        )
        return result

    except Exception as e:
        logger.exception(f"Error calculating stage load ratios for product {product_id}: {e}")
        if raise_on_error:
            raise
        return {
            "product_id": product_id,
            "stage_order": [],
            "load_ratios": [],
        }


if __name__ == "__main__":
    import time

    logger.add(f"logs/stage_load_ratio/{time.strftime('%Y%m%d_%H%M%S')}.log")
    result = get_pid_stage_load("0NQWrgOqH3MAfHOrBFLADg==")

    # Print sample result
    if result["load_ratios"]:
        first_snapshot = result["load_ratios"][0]
        logger.info(f"Sample snapshot_cut: {first_snapshot['snapshot_cut']}")
        logger.info(f"Calendar times count: {len(first_snapshot['calendar_times'])}")
        for stage in result["stage_order"][:3]:  # First 3 stages
            if stage in first_snapshot["stage_metrics"]:
                metrics = first_snapshot["stage_metrics"][stage]
                logger.info(f"Stage {stage}:")
                logger.info(f"  Avg duration: {metrics['avg_duration_seconds']:.2f}s")
                logger.info(f"  Sample 7d ratios: {metrics['rolling_7d'][:5]}")
