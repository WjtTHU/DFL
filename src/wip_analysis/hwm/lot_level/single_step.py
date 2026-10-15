from datetime import date, datetime, time, timedelta
from typing import Any, Literal

import polars as pl
from loguru import logger
from rapidfuzz.distance import Levenshtein

from wip_analysis.data_loader.postgres_loader import execute_batch_async, fetch_data
from wip_analysis.data_loader.sql_template import render_sql
from wip_analysis.hwm.product_level import (
    get_pid_path_candidates,
    get_pid_primary_durs,
    get_pid_primary_path,
)
from wip_analysis.hwm.stage import get_stage_duration
from wip_analysis.hwm.stage.utils.lot_completion_predictions import (
    persist_lot_completion_prediction_rows,
)
from wip_analysis.utils.env_num import env_num
from wip_pred.inference.predict import prediction_extra_feat_cols

from .utils.stage_details import (
    LotCompletionPrediction,
    build_stage_details,
    prediction_requires_facility_context,
)


def _levenshtein_distance(left: list[str], right: list[str]) -> int:
    return int(Levenshtein.distance(left, right))


def _jaccard_distance(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return 1.0 - (len(left_set & right_set) / len(union))


def _get_ground_truth_stage_names(
    lot_id: str,
    snapshot_cut: datetime,
    lot_snapshots: pl.DataFrame,
) -> list[str]:
    try:
        stage_history = get_stage_duration(
            stage_name=None,
            product_id=None,
            lot_id=lot_id,
            cut_time=snapshot_cut.isoformat(),
            scored=False,
        )
        actual_history = stage_history.filter(pl.col("stage_type") == "actual").sort("start_time")
        if not actual_history.is_empty():
            names = actual_history.get_column("stage_name").to_list()
            logger.debug(
                "Ground-truth stages for lot {} @ {} (source=stage_duration, n={}): {}",
                lot_id,
                snapshot_cut,
                len(names),
                names,
            )
            return names
    except Exception as e:
        logger.debug(
            "Failed to load actual stage history for {} at {}: {}",
            lot_id,
            snapshot_cut,
            e,
        )

    if lot_snapshots.is_empty() or "stage" not in lot_snapshots.columns:
        logger.warning(
            "Ground-truth stages for lot {} @ {}: empty (no stage_duration row, no snapshot fallback)",
            lot_id,
            snapshot_cut,
        )
        return []

    names = (
        lot_snapshots.filter(pl.col("lot_id") == lot_id)
        .sort("created_at")
        .get_column("stage")
        .to_list()
    )
    logger.debug(
        "Ground-truth stages for lot {} @ {} (source=snapshot_fallback, n={}): {}",
        lot_id,
        snapshot_cut,
        len(names),
        names,
    )
    return names


def _load_path_candidates(
    product_id: str,
    snapshot_cut: datetime,
    path_end_time: datetime | date | None,
) -> list[dict[str, object]]:
    path_end_date = path_end_time.date() if isinstance(path_end_time, datetime) else path_end_time
    return get_pid_path_candidates(
        pid=product_id,
        snapshot_end=snapshot_cut,
        path_end_date=path_end_date,
        rules={"clustering_method": "hdbscan"},
    )


# Surfaced (in the API response note) when a product has too little snapshot history up to the
# cut to compute a meaningful primary path / stage durations -- see the early-out in
# get_lot_prog_on_snapshot. Disable the early-out with WIP_ANALYSIS_MIN_PRODUCT_SNAPSHOTS<=1.
_INSUFFICIENT_HISTORY_NOTE = (
    "Insufficient product history (fewer than {need} distinct snapshot(s) up to this cut): "
    "showing observed stages only, no projected path/durations."
)


def _product_has_enough_snapshots(
    product_id: str,
    snapshot_end: datetime | date | None,
    min_snapshots: int,
) -> bool:
    """Whether the product has at least ``min_snapshots`` distinct snapshot observations up to the cut.

    Counts distinct ``created_at`` TIMESTAMPS (not calendar days): a product observed at fewer than
    2 distinct timestamps has no stage-to-stage transition over time, so its primary-path durations
    are unsolvable and any computed path window collapses to a single instant (the "No valid
    equations" / zero-span products-row defect). This gates the early-out that avoids producing such
    a degenerate profile. (Two observations on the SAME calendar day still count as two and can be
    enough -- that is why we count timestamps, not dates.)

    Cheap by design: the query stops after finding ``min_snapshots`` distinct timestamps (LIMIT), so
    the hot path (products with ample history) pays only an early-terminating indexed scan. On any
    error it returns True (fail open -> normal path, no early-out).
    """
    if min_snapshots <= 1:
        return True
    try:
        query = render_sql(
            "get_pid_snapshot_count.sql.j2",
            {
                "product_id": product_id,
                "snapshot_end": snapshot_end,
                "max_distinct": int(min_snapshots),
            },
        )
        df = fetch_data(query)
        if df.is_empty():
            return True
        value = df.item(0, "snapshot_count")
        return value is None or int(value) >= min_snapshots
    except Exception as e:
        logger.debug("Snapshot count failed for product {}; skipping early-out: {}", product_id, e)
        return True


def _observed_stage_details(
    lot_id: str,
    lot_snapshots: pl.DataFrame,
    note: str,
) -> list[dict[str, Any]]:
    """Reconstruct the lot's observed stage history from raw snapshots (fallback for the
    insufficient-history early-out when no stage-duration rows exist yet).

    Consecutive identical stages are collapsed to the first time the stage was seen. ``note`` is
    attached to each stage so the caller can surface one row-level explanation.
    """
    if lot_snapshots.is_empty() or "stage" not in lot_snapshots.columns:
        return []
    lot_rows = (
        lot_snapshots.filter(pl.col("lot_id") == lot_id)
        .sort("created_at")
        .select("stage", "created_at")
    )
    if lot_rows.is_empty():
        return []
    blocks = (
        lot_rows.with_columns(
            (pl.col("stage") != pl.col("stage").shift(1)).fill_null(True).alias("_change")
        )
        .with_columns(pl.col("_change").cum_sum().alias("_block"))
        .group_by("_block")
        .agg(
            pl.col("stage").first().alias("stage"),
            pl.col("created_at").min().alias("created_at"),
        )
        .sort("created_at")
    )
    details: list[dict[str, Any]] = []
    for row in blocks.iter_rows(named=True):
        stage = row["stage"]
        start_time = row["created_at"]
        if stage is None or not isinstance(start_time, datetime):
            continue
        details.append(
            {
                "stage_name": str(stage),
                "start_time": start_time.isoformat(),
                "stage_type": "actual",
                "note": note,
            }
        )
    return details


def _pick_best_path_candidate(
    candidate_paths: list[dict[str, object]],
    actual_stage_names: list[str],
    distance_priority: Literal["levenshtein", "jaccard"] = "levenshtein",
    coverage_threshold: float = 1.0,
) -> dict[str, object] | None:
    if not candidate_paths or not actual_stage_names:
        return None

    def _path_id_key(value: object) -> tuple[int, int | str]:
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, str(value))

    actual_unique = set(actual_stage_names)
    actual_n_unique = len(actual_unique)

    parsed: list[dict[str, Any]] = []
    for row in candidate_paths:
        row_path = row.get("path")
        path = [str(stage_name) for stage_name in row_path] if isinstance(row_path, list) else []
        if not path:
            continue
        path_set = set(path)
        covered = len(actual_unique & path_set)
        coverage = covered / actual_n_unique if actual_n_unique else 1.0
        parsed.append(
            {
                "row": row,
                "path_id": row.get("path_id"),
                "path": path,
                "lev": float(_levenshtein_distance(actual_stage_names, path)),
                "jac": _jaccard_distance(actual_stage_names, path),
                "coverage": coverage,
                "is_primary_path": bool(row.get("is_primary_path")),
                "lot_count": int(row.get("lot_count") or 0),
            }
        )

    if not parsed:
        return None

    logger.debug(
        "Path candidate selection: actual_stages (n={}, unique={}) threshold={:.2f}: {}",
        len(actual_stage_names),
        actual_n_unique,
        coverage_threshold,
        actual_stage_names,
    )
    for p in parsed:
        logger.debug(
            "  candidate path_id={} len={} coverage={:.2f} lev={:.0f} jac={:.3f} path={}",
            p["path_id"],
            len(p["path"]),
            p["coverage"],
            p["lev"],
            p["jac"],
            p["path"],
        )

    # Soft coverage filter: keep candidates whose path covers at least `coverage_threshold`
    # of the lot's distinct actual stages. Raw edit distance is structurally biased toward
    # short paths, so requiring some minimum coverage prevents a degenerate single-stage
    # path from winning over a long one purely on length difference. threshold=1.0 is the
    # strict "must contain every actual stage" rule; lower it to tolerate noise.
    eligible = [p for p in parsed if p["coverage"] >= coverage_threshold]
    if not eligible:
        logger.warning(
            "No path candidate meets coverage_threshold={:.2f} for actual stages {}; "
            "returning None so caller falls back to product primary path",
            coverage_threshold,
            sorted(actual_unique),
        )
        return None

    best_primary: float | None = None
    best_secondary: float | None = None
    best: list[dict[str, Any]] = []
    for p in eligible:
        if distance_priority == "levenshtein":
            primary, secondary = p["lev"], p["jac"]
        else:
            primary, secondary = p["jac"], p["lev"]

        if (
            best_primary is None
            or primary < best_primary
            or (
                primary == best_primary
                and best_secondary is not None
                and secondary < best_secondary
            )
        ):
            best_primary = primary
            best_secondary = secondary
            best = [p]
        elif primary == best_primary and secondary == best_secondary:
            best.append(p)

    if not best:
        return None

    chosen = min(
        best,
        key=lambda p: (
            not bool(p["is_primary_path"]),
            -int(p["lot_count"]),
            _path_id_key(p["path_id"]),
            len(p["path"]),
            tuple(p["path"]),
        ),
    )
    logger.debug(
        "Selected path candidate path_id={} len={} (lev={:.0f}, jac={:.3f}, ties={} deterministic)",
        chosen["path_id"],
        len(chosen["path"]),
        chosen["lev"],
        chosen["jac"],
        len(best),
    )
    return chosen["row"]


def _filter_reference_durations(
    reference_path: list[str],
    reference_durations: list[float],
    selected_path: list[str],
) -> list[float]:
    duration_map = {
        stage_name: duration
        for stage_name, duration in zip(reference_path, reference_durations, strict=True)
    }
    filtered: list[float] = []
    for stage_name in selected_path:
        duration = duration_map.get(stage_name)
        if duration is None:
            return []
        filtered.append(float(duration))
    return filtered


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _coerce_path_id(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Ignoring non-integer path_id for lot_progress persistence: {}", value)
        return None


def _build_persist_rows(
    stage_details: list[dict[str, Any]],
    path_id: int | None = None,
) -> list[dict[str, object]]:
    """Build persistable rows with stage_end/duration_seconds from stage start/end times."""
    parsed_rows: list[tuple[str, datetime, str, datetime | None]] = []

    for sd in stage_details:
        stage_name = sd.get("stage_name")
        start_time_raw = sd.get("start_time")
        if not stage_name or not start_time_raw:
            continue

        start_time = _parse_iso_datetime(start_time_raw)
        if start_time is None:
            logger.warning(
                f"Invalid stage start_time for persistence: stage={stage_name}, value={start_time_raw}"
            )
            continue

        explicit_end = _parse_iso_datetime(sd.get("end_time"))
        if sd.get("end_time") and explicit_end is None:
            logger.warning(
                f"Invalid stage end_time for persistence: stage={stage_name}, value={sd.get('end_time')}"
            )

        parsed_rows.append(
            (stage_name, start_time, sd.get("stage_type", "predicted"), explicit_end)
        )

    parsed_rows.sort(key=lambda row: row[1])

    persist_rows: list[dict[str, object]] = []
    for idx, row in enumerate(parsed_rows):
        stage_name, stage_time, stage_type, explicit_end = row
        next_stage_start = parsed_rows[idx + 1][1] if idx + 1 < len(parsed_rows) else None
        stage_end = explicit_end or next_stage_start
        if stage_end is not None and stage_end <= stage_time:
            logger.warning(
                "Ignoring non-positive stage duration for persistence: stage={}, start={}, end={}",
                stage_name,
                stage_time,
                stage_end,
            )
            stage_end = (
                next_stage_start if next_stage_start and next_stage_start > stage_time else None
            )
        duration_seconds = float((stage_end - stage_time).total_seconds()) if stage_end else None
        persist_rows.append(
            {
                "stage_name": stage_name,
                "stage_time": stage_time,
                "stage_type": stage_type,
                "stage_end": stage_end,
                "duration_seconds": duration_seconds,
                "path_id": path_id,
            }
        )

    return persist_rows


def _apply_known_facts(
    current: list[dict[str, Any]],
    known: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    curr_map = {sd["stage_name"]: sd for sd in current if sd.get("stage_name")}
    known_map = {sd["stage_name"]: sd for sd in known or [] if sd.get("stage_name")}
    for stage_name, known_sd in known_map.items():
        curr_sd = curr_map.get(stage_name)
        curr_type = curr_sd.get("stage_type") if curr_sd else None
        # A stage that was once observed ("actual") is an established fact: its start_time
        # must stay frozen across later snapshots. The fresh per-snapshot pass only looks at
        # a bounded recompute window, so once the raw observation scrolls out of that window
        # it would otherwise downgrade the stage back to "interpolated" (or shift the actual
        # time while the lot lingers), making a settled fact drift. Always let a known actual
        # win. Other known facts only backfill stages the current pass hasn't pinned down.
        if known_sd.get("stage_type") == "actual" or curr_sd is None or curr_type == "predicted":
            merged = dict(curr_sd or {})
            for key, value in known_sd.items():
                if key in ("path_id", "note"):
                    continue
                if value is not None:
                    merged[key] = value
            curr_map[stage_name] = merged

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sd in current:
        stage_name = sd.get("stage_name")
        if stage_name and stage_name in curr_map and stage_name not in seen:
            ordered.append(curr_map[stage_name])
            seen.add(stage_name)
    for stage_name, sd in curr_map.items():
        if stage_name not in seen:
            ordered.append(sd)
    return ordered


def _stage_details_from_cached_df(cached_df: pl.DataFrame) -> list[dict[str, Any]]:
    exprs = [
        pl.col("stage_name"),
        pl.col("stage_time").dt.strftime("%Y-%m-%dT%H:%M:%S%.f").alias("start_time"),
        pl.col("stage_type"),
    ]
    if "stage_end" in cached_df.columns:
        exprs.append(pl.col("stage_end").dt.strftime("%Y-%m-%dT%H:%M:%S%.f").alias("end_time"))
    if "path_id" in cached_df.columns:
        exprs.append(pl.col("path_id"))
    return cached_df.select(*exprs).to_dicts()


def get_lot_prog_on_snapshot(
    lot_id: str,
    snapshot_cut: datetime,
    refresh: bool = False,
    path_start_time: datetime | date | None = None,
    path_end_time: datetime | date | None = None,
    known_stage_details: list[dict[str, Any]] | None = None,
    persist: bool = True,
    cached_stage_details: list[dict[str, Any]] | None = None,
    lot_snapshots: pl.DataFrame | None = None,
    distance_priority: Literal["levenshtein", "jaccard"] = "levenshtein",
    coverage_threshold: float = env_num("WIP_ANALYSIS_LOT_PATH_COVERAGE_THRESHOLD", 0.9),
) -> tuple[list[dict[str, Any]], datetime]:
    """
    Retrieve or generate snapshot analysis for a specific lot up till given timespot.

    This function first attempts to fetch existing snapshot analysis from the database.
    If no existing analysis is found, it calculates the analysis from scratch using
    lot snapshots and saves the results to the database for future use.

    Args:
        lot_id (str): The unique identifier for the lot to analyze.
        snapshot_cut (datetime): Would look at snapshots <= this time to build analysis.
        refresh (bool): Whether to calculate the analysis from scratch.
        path_start_time: Optional lower timestamp bound for product path lookup.
        path_end_time: Optional upper timestamp bound for product path lookup.
        known_stage_details (list[dict[str, str]] | None): Known actual/interpolated stage details
            from the previous snapshot (used to keep historical facts stable).
        persist (bool): Whether to write computed results into the cache table.
        cached_stage_details (list[dict[str, str]] | None): Preloaded exact-cache rows for
            snapshot_cut, used by callers that batch-load lot_progress rows.
        lot_snapshots (pl.DataFrame | None): Pre-fetched raw snapshot rows, already filtered
            to created_at <= snapshot_cut and to the context this call needs (just this lot
            for lotwise models). When provided, the per-snapshot DB fetch is skipped so a
            caller computing many cuts can fetch the lot's history once and slice it.
        distance_priority ("levenshtein" | "jaccard"): Which distance metric to use as the
            primary criterion when picking the best path candidate. The other metric is used
            only as a tiebreaker. The two metrics are not mathematically comparable, so the
            caller must choose which one carries more weight.
        coverage_threshold (float): Minimum fraction of the lot's distinct actual stages
            that a path candidate must cover to be considered. 1.0 means every actual stage
            must appear in the candidate path; lower values tolerate noise. Candidates
            below the threshold are excluded; if none survive the function returns None and
            the caller falls back to the product primary path.

    Returns:
        out (tuple[list[dict[str, str]], datetime]): tuple containing

          list[Dict[str, str]]: List of stage detail dictionaries containing snapshot analysis data, keys
            - stage_name: str (stage name)
            - start_time: str (ISO formatted datetime)
            - stage_type: str ("actual", "interpolated", or "predicted")

          datetime: The timestamp of the latest snapshot used for analysis (datetime.min if no data found)

    Note:
        - If existing analysis is not found, the function will build stage details from
          lot snapshots using the product's primary path and durations
        - Results are cached in the hb_lot_progress table for performance
        - Returns empty list and datetime.min if no valid snapshots are found
    """
    if not refresh and cached_stage_details is not None:
        stage_details = _apply_known_facts(cached_stage_details, known_stage_details)
        logger.opt(lazy=True).debug(
            "Batched cache hit for lot {} (snapshot_cut={}, stages={})",
            lambda: lot_id,
            lambda: snapshot_cut,
            lambda: len(stage_details),
        )
        return stage_details, snapshot_cut

    cached_df = pl.DataFrame()
    if not refresh:
        try:
            fetch_cached_q = render_sql(
                "get_lid_progress_on_snapshot.sql.j2",
                {"lot_id": lot_id, "snapshot_cut": snapshot_cut},
            )
            cached_df = fetch_data(fetch_cached_q)
        except Exception as e:
            logger.exception(f"Failed to fetch cached lot progress for {lot_id}: {e}")

    # Case 1: Successfully loaded result from DB
    if not refresh and not cached_df.is_empty():
        try:
            stage_details = _stage_details_from_cached_df(cached_df)
            if "snapshot_cut" in cached_df.columns:
                snapshot_at = cached_df.select(pl.col("snapshot_cut").max()).item()
            else:
                snapshot_at = cached_df.select(pl.col("stage_time").max()).item()
            logger.opt(lazy=True).debug(
                "Cache hit for lot {} (snapshot_cut={}, stages={})",
                lambda: lot_id,
                lambda: snapshot_cut,
                lambda: len(stage_details),
            )
            stage_details = _apply_known_facts(stage_details, known_stage_details)
            return stage_details, snapshot_at
        except Exception as e:
            logger.exception(f"Corrupted cached stage_details for {lot_id}: {e}; rebuilding...")

    # Case 2: Build from raw snapshots (cache miss or corruption)
    seed_details = [
        sd
        for sd in (known_stage_details or [])
        if sd.get("stage_type") in {"actual", "interpolated"}
    ]
    if seed_details:
        stage_details = seed_details
        try:
            last_time = max(datetime.fromisoformat(sd["start_time"]) for sd in seed_details)
        except Exception:
            last_time = snapshot_cut
    else:
        old_stage_durs = get_stage_duration(
            stage_name=None, product_id=None, lot_id=lot_id, cut_time=snapshot_cut.isoformat()
        )

        if old_stage_durs.is_empty() or "start_time" not in old_stage_durs.columns:
            last_time = snapshot_cut
            stage_details = []
        else:
            max_end_time = (
                old_stage_durs["end_time"].max() if "end_time" in old_stage_durs.columns else None
            )
            if isinstance(max_end_time, datetime):
                last_time = max_end_time
            elif isinstance(max_end_time, date):
                last_time = datetime.combine(max_end_time, time.min)
            else:
                last_time = snapshot_cut

            start_time_col = old_stage_durs["start_time"]
            if start_time_col.dtype == pl.Null:
                stage_details = []
            else:
                exprs = [
                    pl.col("stage_name"),
                    pl.col("start_time").dt.strftime("%Y-%m-%dT%H:%M:%S%.f").alias("start_time"),
                    pl.col("stage_type"),
                ]
                if "end_time" in old_stage_durs.columns:
                    exprs.append(
                        pl.col("end_time").dt.strftime("%Y-%m-%dT%H:%M:%S%.f").alias("end_time")
                    )
                stage_details = old_stage_durs.select(*exprs).to_dicts()

    # Reconstruct the lot's observed stage history. Lotwise prediction only needs this
    # lot's rows, and the inference service re-windows internally, so we feed the lot's
    # FULL history (no 30-day cap): an "actual" stage stays observed regardless of how
    # long ago it was last seen or how sparse the snapshots are. Facility-context models
    # still need the windowed multi-lot set for their set-level features.
    if lot_snapshots is not None:
        lot_df = lot_snapshots
    else:
        if prediction_requires_facility_context():
            snapshot_start = snapshot_cut - timedelta(
                days=env_num("WIP_ANALYSIS_FACILITY_SNAPSHOT_WINDOW_DAYS", 30)
            )
            query_context_lot_id = None
        else:
            snapshot_start = None
            query_context_lot_id = lot_id
        latest_snap_q = render_sql(
            "get_lid_snapshots_ranged.sql.j2",
            {
                "lot_id": query_context_lot_id,
                "snapshot_start": snapshot_start,
                "snapshot_cut": snapshot_cut,
                "extra_feat_cols": prediction_extra_feat_cols(),
            },
        )
        lot_df = fetch_data(latest_snap_q)
    if lot_df.is_empty():
        logger.warning(f"Lot {lot_id} has no snapshots up to {snapshot_cut}, skipped.")
        return stage_details, last_time

    product_id = lot_df.filter(pl.col("lot_id") == lot_id).item(0, "product_id")
    effective_path_end_time = path_end_time or snapshot_cut

    # Insufficient-history early-out. A product observed at too few distinct snapshot timestamps
    # up to this cut has no stage-to-stage transition over time, so its primary-path durations are
    # unsolvable and the computed path window would collapse to a single instant (the
    # "No valid equations" / zero-span products-row defect). Rather than fabricate a degenerate
    # uniform-duration profile, skip the path/candidate/duration work and return the lot's
    # observed (actual + interpolated) stages only, annotated with an explanatory note.
    # Configurable: WIP_ANALYSIS_MIN_PRODUCT_SNAPSHOTS (default 2); set 0 or 1 to disable.
    min_snapshots = env_num("WIP_ANALYSIS_MIN_PRODUCT_SNAPSHOTS", 2, min_value=0)
    if min_snapshots >= 2 and not _product_has_enough_snapshots(
        product_id, effective_path_end_time, min_snapshots
    ):
        note = _INSUFFICIENT_HISTORY_NOTE.format(need=min_snapshots)
        logger.info(
            "Lot {} product {} has < {} distinct snapshot(s) up to {}; returning observed "
            "stages only (insufficient history for a product path/duration profile).",
            lot_id,
            product_id,
            min_snapshots,
            effective_path_end_time,
        )
        # Prefer the already-known actual/interpolated history; fall back to a raw-snapshot
        # reconstruction only if no stage-duration rows exist yet. Predicted/future stages are
        # intentionally omitted -- that is exactly the projection we cannot compute here.
        base_details = stage_details or _observed_stage_details(lot_id, lot_df, note=note)
        observed = _apply_known_facts(
            [{**sd, "note": note} for sd in base_details], known_stage_details
        )
        max_created_at = lot_df["created_at"].max()
        snapshot_at = max_created_at if isinstance(max_created_at, datetime) else snapshot_cut
        return observed, snapshot_at

    candidate_paths = _load_path_candidates(
        product_id=product_id,
        snapshot_cut=snapshot_cut,
        path_end_time=path_end_time,
    )
    actual_stage_names = _get_ground_truth_stage_names(lot_id, snapshot_cut, lot_df)
    selected_candidate = _pick_best_path_candidate(
        candidate_paths,
        actual_stage_names,
        distance_priority=distance_priority,
        coverage_threshold=coverage_threshold,
    )

    reference_path: list[str] | None = None
    reference_durations: list[float] | None = None
    selected_path_id: int | None = None
    if selected_candidate is not None:
        candidate_path = selected_candidate.get("path")
        primary_path = (
            [str(stage_name) for stage_name in candidate_path]
            if isinstance(candidate_path, list)
            else []
        )
        if not primary_path:
            logger.warning(
                f"Selected path candidate {selected_candidate.get('path_id')} for lot {lot_id} has no usable path; falling back"
            )
            primary_path = get_pid_primary_path(
                product_id,
                path_start_time=None,
                path_end_time=effective_path_end_time,
            )
            reference_path = primary_path
        logger.debug(
            f"Selected path candidate {selected_candidate.get('path_id')} for lot {lot_id} "
            f"(actual_stages={len(actual_stage_names)}, path_len={len(primary_path)})"
        )
        if primary_path:
            candidate_path_id = selected_candidate.get("path_id")
            selected_path_id = _coerce_path_id(candidate_path_id)
    else:
        logger.debug(
            f"No matching path candidate found for lot {lot_id}; falling back to product primary path"
        )
        primary_path = get_pid_primary_path(
            product_id,
            path_start_time=None,
            path_end_time=effective_path_end_time,
        )
        reference_path = primary_path

    if reference_path is None:
        reference_path = get_pid_primary_path(
            product_id,
            path_start_time=None,
            path_end_time=effective_path_end_time,
        )
    reference_durations = get_pid_primary_durs(
        product_id,
        path_start_time=None,
        path_end_time=effective_path_end_time,
    )

    if len(reference_path) != len(reference_durations):
        logger.warning(
            "Primary path/duration mismatch for product {} before alignment: "
            "path_len={} durations_len={}; forcing recompute",
            product_id,
            len(reference_path),
            len(reference_durations),
        )
        get_pid_primary_path.cache_clear()
        get_pid_primary_durs.cache_clear()
        reference_path = get_pid_primary_path(
            product_id,
            path_start_time=None,
            path_end_time=effective_path_end_time,
            force_refresh=True,
        )
        reference_durations = get_pid_primary_durs(
            product_id,
            path_start_time=None,
            path_end_time=effective_path_end_time,
            force_refresh=True,
        )

    durations = _filter_reference_durations(reference_path, reference_durations, primary_path)
    if not durations:
        logger.warning(
            f"Could not align product durations with selected path for lot {lot_id}: "
            f"reference_path_len={len(reference_path or [])}, reference_durations_len={len(reference_durations or [])}, "
            f"selected_path_len={len(primary_path)}"
        )
        durations = [1.0] * len(primary_path)

    logger.debug(
        "Product '{}', selected path len={}, durations len={}, path={}, durations={}",
        product_id,
        len(primary_path),
        len(durations),
        primary_path,
        durations,
    )

    logger.debug(
        f"Building stage details for lot {lot_id} (snapshots={lot_df.height}, path_len={len(primary_path)})"
    )
    completion_predictions: list[LotCompletionPrediction] = []
    try:
        new_stage_details = build_stage_details(
            target_lot_id=lot_id,
            lot_snapshots=lot_df,
            primary_path=primary_path,
            durations=durations,
            completion_prediction_sink=completion_predictions,
        )
        stage_details.extend(new_stage_details)
    except Exception as e:
        logger.exception(f"Failed to build stage details for lot {lot_id}: {e}")
        return [], datetime.min

    stage_details = _apply_known_facts(stage_details, known_stage_details)
    if selected_path_id is not None:
        for stage_detail in stage_details:
            stage_detail["path_id"] = selected_path_id

    max_created_at = lot_df["created_at"].max()
    snapshot_at: datetime = max_created_at if isinstance(max_created_at, datetime) else datetime.min

    # Write compute result to db
    if persist:
        try:
            insert_q = render_sql("upsert_lid_progress_up_to_cut.sql.j2")
            persist_rows = _build_persist_rows(stage_details, path_id=selected_path_id)
            params_list = [
                (
                    lot_id,
                    snapshot_at,
                    product_id,
                    row["stage_name"],
                    row["stage_time"],
                    row["stage_type"],
                    row["stage_end"],
                    row["duration_seconds"],
                    row["path_id"],
                )
                for row in persist_rows
            ]
            if params_list:
                execute_batch_async(insert_q, params_list)

            completion_prediction_rows = [
                (
                    lot_id,
                    prediction.split_time,
                    product_id,
                    prediction.split_time,
                    prediction.remaining_days,
                    prediction.predicted_completion_time,
                    prediction.model_dir,
                )
                for prediction in completion_predictions
            ]
            persisted_prediction_count = persist_lot_completion_prediction_rows(
                completion_prediction_rows,
                async_write=True,
            )
            if persisted_prediction_count:
                logger.debug(
                    "Cached lot completion prediction for {} at {} (rows={})",
                    lot_id,
                    completion_predictions[-1].split_time,
                    persisted_prediction_count,
                )

            logger.debug(
                f"Cached lot progress for {lot_id} at {snapshot_at} (stages={len(stage_details)})"
            )
        except Exception as e:
            logger.error(f"Failed to persist lot progress cache for {lot_id}: {e}")

    return stage_details, snapshot_at
