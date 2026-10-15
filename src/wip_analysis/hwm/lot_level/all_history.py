from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date, datetime, time
from functools import lru_cache
from typing import Any, TypedDict

import polars as pl
from loguru import logger

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql
from wip_analysis.hwm.stage import get_stage_duration_stats
from wip_analysis.utils.env_num import env_num
from wip_analysis.utils.logging import format_polars_df
from wip_pred.inference.predict import prediction_extra_feat_cols

from .single_step import (
    _apply_known_facts,
    _levenshtein_distance,
    _stage_details_from_cached_df,
    get_lot_prog_on_snapshot,
)
from .utils.stage_details import prediction_requires_facility_context

_DEFAULT_LOT_HISTORY_WORKERS = 8
_LOT_PROGRESS_CACHE_SCHEMA_OVERRIDES = {
    "stage_name": pl.Utf8,
    "stage_time": pl.Datetime,
    "stage_type": pl.Utf8,
    "stage_end": pl.Datetime,
    "product_id": pl.Utf8,
    "snapshot_cut": pl.Datetime,
    "path_id": pl.Int64,
}


class _PathMetadata(TypedDict):
    path_id: int | None
    note: str | None


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"Expected datetime, date, or ISO datetime string; got {type(value).__name__}")


def _format_optional_datetime(value: object) -> str | None:
    if value is None:
        return None
    return _coerce_datetime(value).isoformat()


def _load_cached_stage_details_by_snapshot(
    lot_id: str,
    snapshot_start: datetime,
    snapshot_end: datetime,
) -> dict[datetime, list[dict[str, Any]]]:
    query = render_sql(
        "get_lid_progress_on_snapshot.sql.j2",
        {
            "lot_id": lot_id,
            "snapshot_start": snapshot_start,
            "snapshot_end": snapshot_end,
        },
    )
    cached_df = fetch_data(query, schema_overrides=_LOT_PROGRESS_CACHE_SCHEMA_OVERRIDES)
    if cached_df.is_empty() or "snapshot_cut" not in cached_df.columns:
        return {}

    cached_by_snapshot: dict[datetime, list[dict[str, Any]]] = {}
    snapshot_cuts = cached_df.select(pl.col("snapshot_cut").unique().sort())
    for row in snapshot_cuts.iter_rows(named=True):
        snapshot_cut = row["snapshot_cut"]
        if not isinstance(snapshot_cut, datetime):
            continue
        snapshot_df = cached_df.filter(pl.col("snapshot_cut") == snapshot_cut).sort("stage_time")
        cached_by_snapshot[snapshot_cut] = _stage_details_from_cached_df(snapshot_df)

    return cached_by_snapshot


def _extract_path_id(stage_details: list[dict[str, Any]]) -> int | None:
    path_ids: list[int] = []
    for stage_detail in stage_details:
        value = stage_detail.get("path_id")
        if value is None:
            continue
        try:
            path_ids.append(int(value))
        except (TypeError, ValueError):
            logger.warning("Ignoring non-integer path_id in lot progress row: {}", value)

    unique_path_ids = sorted(set(path_ids))
    if not unique_path_ids:
        return None
    if len(unique_path_ids) > 1:
        logger.warning("Multiple path_id values found for one snapshot: {}", unique_path_ids)
    return unique_path_ids[0]


def _extract_note(stage_details: list[dict[str, Any]]) -> str | None:
    """Row-level note for a snapshot (e.g. the insufficient-history early-out explanation).

    The note is attached per-stage by the producer but is a property of the whole snapshot row,
    so we surface the first non-empty one (mirrors how _extract_path_id collapses per-stage ids).
    """
    for stage_detail in stage_details:
        note = stage_detail.get("note")
        if note:
            return str(note)
    return None


def _fetch_full_lot_snapshots(lot_id: str, snapshot_end: datetime) -> pl.DataFrame:
    """Fetch this lot's entire snapshot history up to snapshot_end (no lower bound).

    Used to reconstruct every cut from in-memory slices instead of one windowed DB
    query per cut. Safe only for lotwise prediction, where a single lot's rows are all
    the reconstruction and the model need.
    """
    query = render_sql(
        "get_lid_snapshots_ranged.sql.j2",
        {
            "lot_id": lot_id,
            "snapshot_start": None,
            "snapshot_cut": snapshot_end,
            "extra_feat_cols": prediction_extra_feat_cols(),
        },
    )
    return fetch_data(query)


def get_lot_prog_over_snapshots(
    lot_id: str,
    refresh: bool = False,
    snapshot_start: datetime | None = None,
    snapshot_end: datetime | None = None,
) -> list[dict[str, Any]]:
    """Get lot history, using an in-process cache for bounded historical ranges."""
    if refresh:
        _get_lot_prog_over_snapshots_cached.cache_clear()
        return _get_lot_prog_over_snapshots_uncached(
            lot_id=lot_id,
            refresh=True,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )

    if snapshot_end is None:
        return _get_lot_prog_over_snapshots_uncached(
            lot_id=lot_id,
            refresh=False,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )

    return deepcopy(_get_lot_prog_over_snapshots_cached(lot_id, snapshot_start, snapshot_end))


def clear_lot_history_cache() -> None:
    """Clear in-process lot history result cache."""
    _get_lot_prog_over_snapshots_cached.cache_clear()


@lru_cache(maxsize=128)
def _get_lot_prog_over_snapshots_cached(
    lot_id: str,
    snapshot_start: datetime | None,
    snapshot_end: datetime,
) -> list[dict[str, Any]]:
    return _get_lot_prog_over_snapshots_uncached(
        lot_id=lot_id,
        refresh=False,
        snapshot_start=snapshot_start,
        snapshot_end=snapshot_end,
    )


def _get_lot_prog_over_snapshots_uncached(
    lot_id: str,
    refresh: bool = False,
    snapshot_start: datetime | None = None,
    snapshot_end: datetime | None = None,
) -> list[dict[str, Any]]:
    """
    Get all lot snapshot analysis rows for a specific lot ID.

    This function retrieves all snapshots for a lot and ensures each has
    corresponding analysis data. If any snapshot lacks analysis, it will
    calculate it from scratch using get_lot_prog_on_snapshot.

    This function aggregates the results from `get_lot_prog_on_snapshot`.
    So, of course, this function should work as follows: Address issue first: gather the snapshot at specific times for this lot. Use the time scope and the lot ID as the primary key to fetch data from the lot progress table. Then distribute it along the primary path.

    Args:
        lot_id (str): The unique identifier for the lot
        refresh (bool): If True, ignore cache and recalculate everything from scratch
        snapshot_start (datetime | None): Only consider snapshots taken on/after this timestamp
        snapshot_end (datetime | None): Only consider snapshots taken on/before this timestamp

    Returns:
        List[Dict]: List of dictionaries containing
          * snapshot_t: datetime when the snapshot was taken
          * stage_details: List of stage detail dictionaries with keys
            - stage_name: str (stage name)
            - start_time: str (ISO formatted datetime)
            - stage_type: str ("actual", "interpolated", or "predicted")
            - cumulate_h: float (cumulative hours till this stage since lot start)
            - duration_z: float (z-score of the duration for this stage)
            - avg_cum_{window}_h: float (average cumulative hours, on different windows)
            - dev_vs_{window}_h: float (deviation vs avg cumulative hours, on different windows)
            - *we have: "1d", "7d", "1m", "1q", "6m", "1y" as time windows.*
          * product_id: Product identifier
    """
    #################################################
    # Get all unique snapshot timestamps for this lot
    #################################################
    logger.debug("Fetching snapshot times for lot '{}'", lot_id)

    try:
        snapshots_query = render_sql(
            "get_lid_snapshot_times.sql.j2",
            {
                "lot_id": lot_id,
                "snapshot_start": snapshot_start,
                "snapshot_end": snapshot_end,
            },
        )
        snapshots_times_all = fetch_data(snapshots_query)
    except Exception as e:
        logger.exception(f"Failed to fetch snapshot times for lot {lot_id}: {e}")
        return []

    if snapshots_times_all.is_empty():
        logger.warning(f"No snapshots found for lot '{lot_id}'")
        return []

    # Determine lot start time (use full history to avoid range bias)
    wafer_start_val = None
    if "wafer_start_date" in snapshots_times_all.columns:
        try:
            wafer_start_val = snapshots_times_all.item(0, "wafer_start_date")
        except Exception:
            wafer_start_val = None

    if isinstance(wafer_start_val, datetime):
        lot_start_t = wafer_start_val
    elif isinstance(wafer_start_val, date):
        lot_start_t = datetime.combine(wafer_start_val, time.min)
    else:
        first_created_val = None
        if "first_created_at" in snapshots_times_all.columns:
            try:
                first_created_val = snapshots_times_all.item(0, "first_created_at")
            except Exception:
                first_created_val = None
        lot_start_t = (
            first_created_val
            if isinstance(first_created_val, datetime)
            else snapshots_times_all.item(0, "created_at")
        )

    logger.debug("Wafer start time: {}", lot_start_t.isoformat())

    snapshots_times = snapshots_times_all

    if snapshots_times.is_empty():
        logger.warning(
            f"No snapshots found for lot '{lot_id}' in range: {snapshot_start} -> {snapshot_end}"
        )
        return []

    logger.opt(lazy=True).debug(
        "Found {} snapshots at: {}",
        lambda: snapshots_times.height,
        lambda: [t["created_at"].isoformat() for t in snapshots_times.iter_rows(named=True)],
    )

    #############################################
    # Gather analysis data for each snapshot time
    #############################################
    all_analysis_data: list[dict[str, Any]] = []
    product_id = snapshots_times.item(0, "product_id")
    cached_stage_details_by_snapshot: dict[datetime, list[dict[str, Any]]] = {}
    path_metadata_by_snapshot: dict[datetime, _PathMetadata] = {}
    if not refresh:
        try:
            cached_stage_details_by_snapshot = _load_cached_stage_details_by_snapshot(
                lot_id=lot_id,
                snapshot_start=snapshots_times.select(pl.col("created_at").min()).item(),
                snapshot_end=snapshots_times.select(pl.col("created_at").max()).item(),
            )
            logger.debug(
                "Preloaded lot progress cache for lot {}: {}/{} snapshot cuts",
                lot_id,
                len(cached_stage_details_by_snapshot),
                snapshots_times.height,
            )
        except Exception as e:
            logger.exception(f"Failed to preload cached lot progress for {lot_id}: {e}")

    snapshot_time_list = [
        _coerce_datetime(row["created_at"]) for row in snapshots_times.iter_rows(named=True)
    ]

    # Fetch this lot's full snapshot history once so each cut is reconstructed from a cheap
    # in-memory slice rather than its own windowed DB query. Lotwise only — facility-context
    # models still need their per-cut windowed multi-lot fetch (handled inside the worker).
    full_lot_snapshots: pl.DataFrame | None = None
    if not prediction_requires_facility_context():
        try:
            full_lot_snapshots = _fetch_full_lot_snapshots(
                lot_id, snapshot_end=max(snapshot_time_list)
            )
        except Exception as e:
            logger.exception(f"Failed to prefetch full snapshot history for {lot_id}: {e}")
            full_lot_snapshots = None

    def _compute_stage_details(
        snapshot_time: datetime,
    ) -> tuple[datetime, list[dict[str, Any]] | None]:
        lot_slice: pl.DataFrame | None = None
        if (
            full_lot_snapshots is not None
            and not full_lot_snapshots.is_empty()
            and "created_at" in full_lot_snapshots.columns
        ):
            lot_slice = full_lot_snapshots.filter(pl.col("created_at") <= snapshot_time)
        try:
            stage_details, _ = get_lot_prog_on_snapshot(
                lot_id,
                snapshot_time,
                refresh=refresh,
                path_start_time=None,
                # Keep product/path lookup aligned to the exact snapshot timestamp.
                path_end_time=None,
                # Each cut is reconstructed independently; cross-snapshot fact stability is
                # enforced afterwards in one cheap chronological pass below.
                known_stage_details=None,
                cached_stage_details=cached_stage_details_by_snapshot.get(snapshot_time),
                lot_snapshots=lot_slice,
            )
            return snapshot_time, stage_details
        except Exception as e:
            logger.exception(f"Failed to analyze lot {lot_id} at {snapshot_time}: {e}")
            return snapshot_time, None

    # The heavy per-cut work (path candidate selection + model inference) is independent
    # across cuts, so fan it out. Reconstructing actuals from the lot's full history makes
    # each cut self-consistent, which is what lets us drop the old serial fact-threading.
    results_by_snapshot: dict[datetime, list[dict[str, Any]] | None] = {}
    worker_count = min(
        len(snapshot_time_list),
        max(1, env_num("WIP_ANALYSIS_LOT_HISTORY_WORKERS", _DEFAULT_LOT_HISTORY_WORKERS)),
    )
    if worker_count <= 1:
        for snapshot_time in snapshot_time_list:
            st, sd = _compute_stage_details(snapshot_time)
            results_by_snapshot[st] = sd
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for st, sd in executor.map(_compute_stage_details, snapshot_time_list):
                results_by_snapshot[st] = sd

    # Single ordered pass: freeze established "actual" facts forward (cheap defence that also
    # covers facility-context models, which still recompute on a bounded window), then flatten
    # to per-stage analysis rows. This is the only step that must stay sequential.
    prev_facts: list[dict[str, Any]] = []
    for snapshot_time in snapshot_time_list:
        stage_details = results_by_snapshot.get(snapshot_time)
        if not stage_details:
            logger.error(f"No stage details computed for lot {lot_id} at {snapshot_time}")
            continue

        stage_details = _apply_known_facts(stage_details, prev_facts)
        path_metadata_by_snapshot[snapshot_time] = {
            "path_id": _extract_path_id(stage_details),
            "note": _extract_note(stage_details),
        }
        prev_facts = [sd for sd in stage_details if sd.get("stage_type") == "actual"]

        # iterate through each stage on this snapshot time
        try:
            for stage_detail in stage_details:
                stage_start_t = _coerce_datetime(stage_detail["start_time"])
                cumulative_hours = (stage_start_t - lot_start_t).total_seconds() / 3600
                all_analysis_data.append(
                    {
                        "snapshot_t": snapshot_time,
                        "product_id": product_id,
                        "stage_name": stage_detail["stage_name"],
                        "start_time": stage_start_t.isoformat(),
                        "end_time": _format_optional_datetime(stage_detail.get("end_time")),
                        "stage_type": stage_detail["stage_type"],
                        "cumulate_h": cumulative_hours,
                    }
                )
                logger.opt(lazy=True).debug(
                    "New stage: {:<13} Type: {:<12} Time: {}",
                    lambda sd=stage_detail: sd["stage_name"],
                    lambda sd=stage_detail: sd["stage_type"],
                    lambda sd=stage_detail: sd["start_time"],
                )
        except Exception as e:
            logger.exception(
                f"Failed to process stage details for lot {lot_id} at {snapshot_time}: {e}"
            )
            continue

    if not all_analysis_data:
        logger.warning(f"No valid analysis data found for lot {lot_id}")
        return []

    logger.debug(
        "Processing {} total stage records across all snapshots",
        len(all_analysis_data),
    )

    ####################################################################
    # Calculate rolling averages and deviations
    # TODO: Confirm the rolling logic. if data doesn't support window, will return a value now (should it be NULL?)
    logger.debug("Calculating rolling averages and deviations for Lot '{}'...", lot_id)
    df = pl.DataFrame(
        all_analysis_data,
        schema_overrides={
            "snapshot_t": pl.Datetime,
            "product_id": pl.Utf8,
            "stage_name": pl.Utf8,
            "start_time": pl.Utf8,
            "end_time": pl.Utf8,
            "stage_type": pl.Utf8,
            "cumulate_h": pl.Float64,
        },
        infer_schema_length=None,
    ).sort("snapshot_t", "stage_name")
    time_windows = {"1d": "1d", "7d": "7d", "1m": "1mo", "1q": "3mo", "6m": "6mo", "1y": "1y"}

    rolling_exprs = [
        pl.col("cumulate_h")
        .rolling_mean_by(by="snapshot_t", window_size=window)
        .over("stage_name")
        .alias(f"avg_cum_{name}_h")
        for name, window in time_windows.items()
    ]
    df_with_roll = df.with_columns(rolling_exprs)

    dev_exprs = [
        (pl.col("cumulate_h") - pl.col(f"avg_cum_{name}_h")).alias(f"dev_vs_{name}_h")
        for name in time_windows.keys()
    ]
    df_with_dev = df_with_roll.with_columns(dev_exprs).sort("snapshot_t", "cumulate_h")
    logger.opt(lazy=True).trace(
        "Lot '{}' data with Rolling AVG & Dev: \n{}",
        lambda: lot_id,
        lambda: format_polars_df(df_with_dev),
    )

    ###################################
    # Calculate stage duration z-scores
    ###################################
    try:
        stage_avg_std = get_stage_duration_stats(
            stage_name=None,
            product_id=product_id,
            lot_id=None,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )
        stage_avg_std = stage_avg_std.with_columns(pl.col("stage_name").cast(pl.Utf8))
    except Exception:
        logger.exception(f"Skipping duration z-score for lot '{lot_id}'")
        stage_avg_std = pl.DataFrame(
            {
                "stage_name": pl.Series([], dtype=pl.Utf8),
                "avg_duration_seconds": pl.Series([], dtype=pl.Float64),
                "std_duration_seconds": pl.Series([], dtype=pl.Float64),
            }
        )
    df_with_z = (
        df_with_dev.sort("snapshot_t", "cumulate_h")
        .with_columns(
            pl.coalesce(
                pl.col("end_time"),
                pl.col("start_time").shift(-1).over("snapshot_t"),
            ).alias("end_time")
        )
        .join(stage_avg_std, on="stage_name", how="left")
        .with_columns(
            (
                (
                    (
                        pl.col("end_time").str.to_datetime()
                        - pl.col("start_time").str.to_datetime()
                    ).dt.total_seconds()
                    - pl.col("avg_duration_seconds")
                )
                / pl.col("std_duration_seconds")
            ).alias("duration_z")
        )
    )
    # Build final results
    try:
        results = (
            df_with_z.group_by("snapshot_t", "product_id")
            .agg(
                pl.struct(
                    pl.col("stage_name"),
                    pl.col("start_time"),
                    pl.col("stage_type"),
                    pl.col("cumulate_h"),
                    pl.col("duration_z"),
                    pl.col("end_time"),
                    pl.col("avg_cum_1d_h"),
                    pl.col("avg_cum_7d_h"),
                    pl.col("avg_cum_1m_h"),
                    pl.col("avg_cum_1q_h"),
                    pl.col("avg_cum_6m_h"),
                    pl.col("avg_cum_1y_h"),
                    pl.col("dev_vs_1d_h"),
                    pl.col("dev_vs_7d_h"),
                    pl.col("dev_vs_1m_h"),
                    pl.col("dev_vs_1q_h"),
                    pl.col("dev_vs_6m_h"),
                    pl.col("dev_vs_1y_h"),
                )
                .sort_by("cumulate_h")
                .alias("stage_details")
            )
            .sort("snapshot_t")
            .to_dicts()
        )

        prev_path_order: list[str] | None = None
        for row in results:
            path_order = [sd["stage_name"] for sd in row.get("stage_details", [])]
            row["path_order"] = path_order
            path_metadata = path_metadata_by_snapshot.get(row["snapshot_t"], {})
            row["path_id"] = path_metadata.get("path_id")
            row["note"] = path_metadata.get("note")
            row["path_edit_distance"] = (
                None
                if prev_path_order is None
                else _levenshtein_distance(prev_path_order, path_order)
            )
            prev_path_order = path_order

        logger.debug(
            "Prepared {} snapshot-stage rows for lot {}",
            len(all_analysis_data),
            lot_id,
        )
        logger.opt(lazy=True).trace(
            "Final results for lot '{}': {}", lambda: lot_id, lambda: results
        )
        return results

    except Exception as e:
        logger.exception(f"Failed to build final results for lot {lot_id}: {e}")
        return []
