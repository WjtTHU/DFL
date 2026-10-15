from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, Optional

import polars as pl
from loguru import logger

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql
from wip_analysis.hwm.product_level import get_pid_primary_path, get_pid_stage_stats
from wip_analysis.hwm.stage import get_stage_capacity, get_stage_duration

from .single_step import get_lot_prog_on_snapshot

CapacityWindow = Literal["7d", "all"]


@dataclass(frozen=True)
class LotQuickestFinishConfig:
    duration_quantile: float = 0
    capacity_window: CapacityWindow = "7d"
    include_queue_wait: bool = True


def _get_latest_snapshot_for_lot(lot_id: str) -> tuple[datetime, str]:
    snapshots_query = render_sql("get_lid_snapshot_times.sql.j2", {"lot_id": lot_id})
    snapshots = fetch_data(snapshots_query)
    if snapshots.is_empty():
        raise ValueError(f"No snapshots found for lot_id={lot_id}")

    last_row = snapshots.sort("created_at").row(-1, named=True)
    snapshot_at = last_row["created_at"]
    product_id = last_row["product_id"]
    if not isinstance(snapshot_at, datetime):
        raise TypeError(f"Unexpected created_at type: {type(snapshot_at)}")
    if not isinstance(product_id, str) or not product_id:
        raise ValueError("Missing product_id for lot")
    return snapshot_at, product_id


def _safe_quantile(values: pl.Series, q: float) -> Optional[float]:
    clean = values.drop_nulls()
    if clean.is_empty():
        return None
    result = clean.quantile(q)
    return float(result) if result is not None else None


def _stage_duration_floor_seconds(
    duration_df: pl.DataFrame,
    primary_path: list[str],
    quantile: float,
) -> dict[str, float]:
    if duration_df.is_empty():
        return {stage: 0.0 for stage in primary_path}

    floors: dict[str, float] = {}
    for stage in primary_path:
        stage_df = duration_df.filter(pl.col("stage_name") == stage)
        if stage_df.is_empty():
            floors[stage] = 0.0
            continue

        durations = stage_df.get_column("duration_seconds").drop_nulls()
        if durations.is_empty():
            floors[stage] = 0.0
            continue

        # Calculate floor based on strategy, fallback to mean
        floor_val: float = 0.0
        result = durations.quantile(quantile)
        if result is not None:
            floor_val = float(result)
        else:
            mean_val = durations.mean()
            floor_val = float(mean_val) if mean_val is not None else 0.0  # type: ignore

        floors[stage] = max(0.0, floor_val)

    return floors


def _stage_capacity_lots_per_day(
    capacity_df: pl.DataFrame,
    primary_path: list[str],
    as_of: datetime,
    window: CapacityWindow,
) -> dict[str, Optional[float]]:
    if capacity_df.is_empty():
        return {stage: None for stage in primary_path}

    # Filter by time window if needed
    df = capacity_df
    if window == "7d":
        start_date = as_of.date() - timedelta(days=7)
        df = df.filter(pl.col("calendar_date") >= start_date)

    # Get max capacity for each stage
    capacities: dict[str, Optional[float]] = {}
    for stage in primary_path:
        stage_df = df.filter(pl.col("stage_name") == stage)
        if stage_df.is_empty():
            capacities[stage] = None
            continue

        max_throughput = stage_df.select(pl.col("entering_lot_count").max()).item()
        capacities[stage] = float(max_throughput) if max_throughput is not None else None

    return capacities


def _stage_wip_counts(
    product_id: str,
    primary_path: list[str],
    target_date: datetime,
) -> dict[str, Optional[int]]:
    """Get WIP (lots in process) per stage at the given time.

    For past/present times: uses the latest available snapshot <= target_date
    For future times: uses the latest available snapshot and projects forward
    based on capacity trends.
    """
    stats = get_pid_stage_stats(product_id)
    if not stats:
        return {stage: None for stage in primary_path}

    df = pl.DataFrame(stats).with_columns(
        pl.col("snapshot_cut").cast(pl.Datetime),
        pl.col("calendar_time").cast(pl.Date),
    )

    if df.is_empty():
        return {stage: None for stage in primary_path}

    # Get the latest available snapshot
    latest_snapshot = df.select(pl.col("snapshot_cut").max()).item()

    # Filter to latest snapshot and closest calendar time
    snapshot_df = df.filter(pl.col("snapshot_cut") == latest_snapshot)
    target_date_only = target_date.date()

    # Find the closest calendar time <= target_date
    valid_dates = snapshot_df.filter(pl.col("calendar_time") <= target_date_only)
    if valid_dates.is_empty():
        # If target is before all data, use earliest available
        row = snapshot_df.sort("calendar_time").row(0, named=True)
    else:
        # Use the latest date <= target
        row = valid_dates.sort("calendar_time").row(-1, named=True)

    stage_lot_cnt = row.get("stage_lot_cnt")
    if not isinstance(stage_lot_cnt, list) or len(stage_lot_cnt) != len(primary_path):
        logger.warning(f"Invalid stage_lot_cnt format: {stage_lot_cnt}")
        return {stage: None for stage in primary_path}

    return {
        stage: int(stage_lot_cnt[idx]) if idx < len(stage_lot_cnt) else None
        for idx, stage in enumerate(primary_path)
    }


def _determine_current_stage(
    primary_path: list[str],
    stage_details: list[dict[str, Any]],
    as_of: datetime,
) -> tuple[int, str, datetime]:
    """Determine the current stage index, name, and start time.

    Returns:
        (stage_index, stage_name, stage_start_time) - defaults to stage 0 if nothing started
    """
    stage_map = {sd["stage_name"]: sd for sd in stage_details}

    # Find last actual/interpolated stage in primary path
    for idx in range(len(primary_path) - 1, -1, -1):
        stage = primary_path[idx]
        sd = stage_map.get(stage)
        if not sd:
            continue

        stage_type = sd.get("stage_type")
        if stage_type in ("actual", "interpolated"):
            try:
                start_time = datetime.fromisoformat(sd["start_time"])
                return idx, stage, start_time
            except (KeyError, ValueError, TypeError):
                pass

    # Default: nothing started yet, start from beginning
    return 0, primary_path[0], as_of


def _calculate_stage_metrics(
    stage: str,
    entry_time: datetime,
    is_current: bool,
    current_start: datetime,
    as_of: datetime,
    duration_floor: float,
    capacity: Optional[float],
    wip_count: Optional[int],
    include_queue_wait: bool,
) -> tuple[dict[str, Any], datetime]:
    """Calculate all metrics for a single stage and return exit time.

    Returns:
        (metrics_dict, exit_time) - metrics for this stage and when lot exits
    """
    # Calculate elapsed and remaining processing time
    elapsed = 0.0
    if is_current:
        elapsed = max(0.0, (as_of - current_start).total_seconds())
    remaining_proc = max(0.0, duration_floor - elapsed) if is_current else duration_floor

    # Calculate queue wait time based on WIP at entry time
    queue_wait = 0.0
    if include_queue_wait and capacity and capacity > 0 and wip_count is not None:
        # Lots ahead in queue (exclude current lot if already in this stage)
        lots_ahead = max(wip_count - (1 if is_current else 0), 0)
        queue_wait = (lots_ahead / capacity) * 86400.0  # Convert days to seconds

    stage_total = remaining_proc + queue_wait
    exit_time = entry_time + timedelta(seconds=stage_total)

    return {
        "stage_name": stage,
        "entry_time": entry_time,
        "duration_floor_seconds": round(duration_floor, 3),
        "elapsed_seconds": round(elapsed, 3),
        "remaining_process_seconds": round(remaining_proc, 3),
        "capacity_lots_per_day": capacity,
        "wip_lots": wip_count,
        "queue_wait_seconds": round(queue_wait, 3),
        "stage_total_seconds": round(stage_total, 3),
        "exit_time": exit_time,
    }, exit_time


def get_lot_quickest_finish(
    lot_id: str,
    *,
    as_of: Optional[datetime] = None,
    config: Optional[LotQuickestFinishConfig] = None,
) -> dict[str, Any]:
    """Estimate earliest possible finish time for a lot.

    Combines:
    - Per-stage minimal/percentile processing times (from stage duration history)
    - Per-stage max throughput (from stage capacity history)
    - Optional queue wait lower-bound from current WIP/capacity

    Args:
        lot_id: Lot identifier
        as_of: Time point for calculation (defaults to latest snapshot)
        config: Configuration for calculation strategy

    Returns:
        Dictionary with estimated finish time and detailed stage breakdown.
        On error, returns dict with error field.
    """
    cfg = config or LotQuickestFinishConfig()

    # 1. Get lot's latest snapshot and product
    try:
        snapshot_at, product_id = _get_latest_snapshot_for_lot(lot_id)
    except Exception as e:
        logger.exception(f"Failed to find snapshot for lot {lot_id}: {e}")
        return {
            "lot_id": lot_id,
            "product_id": None,
            "as_of": None,
            "earliest_finish_at": None,
            "total_seconds": None,
            "stages": [],
            "error": str(e),
        }

    as_of_t = as_of or snapshot_at
    primary_path: list[str] = get_pid_primary_path(product_id)
    if not primary_path:
        logger.error(f"Empty primary path for product {product_id}")
        return {
            "lot_id": lot_id,
            "product_id": product_id,
            "as_of": as_of_t,
            "earliest_finish_at": None,
            "total_seconds": None,
            "stages": [],
            "error": "Empty primary_path",
        }

    # 2. Get current lot progress
    stage_details, _ = get_lot_prog_on_snapshot(lot_id, as_of_t, refresh=False)
    current_idx, current_stage, current_start = _determine_current_stage(
        primary_path, stage_details, as_of_t
    )
    remaining_stages = primary_path[current_idx:]

    # 3. Get historical data for estimation
    dur_df = get_stage_duration(
        stage_name=None,
        product_id=product_id,
        lot_id=None,
        cut_time=as_of_t.isoformat(),
        scored=False,
    )
    duration_floors = _stage_duration_floor_seconds(dur_df, primary_path, cfg.duration_quantile)

    cap_df = get_stage_capacity(stage_name=None, product_id=product_id)
    capacities = _stage_capacity_lots_per_day(cap_df, primary_path, as_of_t, cfg.capacity_window)

    # 4. Calculate metrics sequentially for each remaining stage
    # This implements the ramp-up rule: determine quickest entry time for each stage,
    # then check WIP at that projected time to calculate realistic queue wait
    stages_out = []
    current_time = as_of_t  # Track when lot enters each stage

    for stage in remaining_stages:
        is_current = stage == current_stage

        # Get WIP count at the projected entry time for this stage
        wip_count = None
        if cfg.include_queue_wait:
            wip_counts_at_entry = _stage_wip_counts(product_id, primary_path, current_time)
            wip_count = wip_counts_at_entry.get(stage)

        # Calculate metrics and get exit time
        metrics, exit_time = _calculate_stage_metrics(
            stage=stage,
            entry_time=current_time,
            is_current=is_current,
            current_start=current_start,
            as_of=as_of_t,
            duration_floor=duration_floors.get(stage, 0.0),
            capacity=capacities.get(stage),
            wip_count=wip_count,
            include_queue_wait=cfg.include_queue_wait,
        )
        stages_out.append(metrics)

        # Exit time becomes entry time for next stage
        current_time = exit_time

    earliest_finish_at = current_time
    total_seconds = (earliest_finish_at - as_of_t).total_seconds()

    return {
        "lot_id": lot_id,
        "product_id": product_id,
        "as_of": as_of_t,
        "current_stage": current_stage,
        "remaining_stages": remaining_stages,
        "config": {
            "duration_quantile": cfg.duration_quantile,
            "capacity_window": cfg.capacity_window,
            "include_queue_wait": cfg.include_queue_wait,
        },
        "earliest_finish_at": earliest_finish_at,
        "total_seconds": round(total_seconds, 3),
        "stages": stages_out,
    }
