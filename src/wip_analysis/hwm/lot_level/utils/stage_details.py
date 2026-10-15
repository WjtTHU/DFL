from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import polars as pl
from loguru import logger

from wip_pred.inference.predict import (
    get_inference_service,
)
from wip_pred.inference.predict import (
    prediction_requires_facility_context as _prediction_requires_facility_context,
)

from .interpolate import interpolate_between


@dataclass(frozen=True)
class LotCompletionPrediction:
    split_time: datetime
    remaining_days: float
    predicted_completion_time: datetime
    model_dir: str


def prediction_requires_facility_context(model_dir: Path = Path("checkpoints/_release")) -> bool:
    return _prediction_requires_facility_context(model_dir)


def build_stage_details(
    target_lot_id: str,
    lot_snapshots: pl.DataFrame,
    primary_path: list[str],
    durations: list[float],
    model_dir: Path = Path("checkpoints/_release"),
    completion_prediction_sink: list[LotCompletionPrediction] | None = None,
) -> list[dict[str, str]]:
    """
    Builds a list of stage starting times for each snapshot in a lot's history,
    combining **actual**, **interpolated**, and **predicted** stage start times.

    this function processes each snapshot to determine the **start times** of each stage.
    - uses actual data where available,
    - interpolates missing stages between known points,
    - predicts future stages based on provided durations and constraints.

    Args:
        target_lot_id (str): The unique identifier for the lot being analyzed.
        lot_snapshots (pl.DataFrame): DataFrame containing snapshots of lots **up to a certain
          point in time when a snapshot is taken**, expected columns
          1. 'created_at': timestamp of the snapshot
          2. 'stage': name of the stage this lot is on
          3. 'lot_id': identifier for the lot
          4. 'lot_status': current status of the lot
          5. 'product_id': identifier for the product
          6. 'current_wafer_qty': current quantity of wafers in the lot
          7. 'parent_lot_id': identifier for the parent lot
        primary_path (List[str]): Ordered list of stage names representing the primary process path.
        durations (List[float]): Estimated **relative** duration for each stage in the primary path.
        model_dir (Path): Path to the directory containing the trained model & related files for predictions.
        completion_prediction_sink: Optional list receiving the model-based completion
          prediction metadata. Fallback estimates are intentionally not recorded here.

    Returns:
        results (List[Dict[str, str]]): A list where each element corresponds to a stage.
        Each element is a dictionary containing
          - "stage": Stage name.
          - "time": ISO-formatted start time.
          - "type": literal "actual", "interpolated", or "predicted"
    """
    if lot_snapshots.is_empty():
        logger.warning("Empty lot snapshots provided")
        return []

    path_index: dict[str, int] = {s: i for i, s in enumerate(primary_path)}
    last_stage_idx = len(primary_path) - 1
    if len(durations) != len(primary_path):
        logger.warning("Durations length mismatch, using uniform weights")
        durations = [1.0] * len(primary_path)

    def _predict_remaining_days_fallback() -> float:
        # Estimate remaining time by scaling path weights with observed lot pace.
        if prev_stg_idx < 0:
            remaining_weight = float(sum(durations))
            return max(remaining_weight, 1.0)

        remaining_weight = float(sum(durations[prev_stg_idx + 1 :]))
        if remaining_weight <= 0:
            return 1.0

        observed_start = min(stg_timestamp.values()) if stg_timestamp else prev_time
        elapsed_hours = max((prev_time - observed_start).total_seconds() / 3600.0, 1.0)
        observed_weight = float(sum(durations[: prev_stg_idx + 1]))
        if observed_weight <= 0:
            hours_per_weight = 24.0
        else:
            hours_per_weight = elapsed_hours / observed_weight
        return max((remaining_weight * hours_per_weight) / 24.0, 1.0)

    stg_timestamp: dict[int, datetime] = {}
    stg_t_type: dict[int, str] = {}
    stg_end_timestamp: dict[int, datetime] = {}
    predicted_completion_time: datetime | None = None

    ##############
    # History Part
    ##############
    sorted_snapshots = lot_snapshots.filter(pl.col("lot_id") == target_lot_id).sort("created_at")

    wafer_start_time: datetime | None = None
    if "wafer_start_date" in sorted_snapshots.columns:
        try:
            non_null = sorted_snapshots.select(pl.col("wafer_start_date").drop_nulls())
            if not non_null.is_empty():
                v = non_null.item(0, 0)
                if isinstance(v, datetime):
                    wafer_start_time = v
                elif isinstance(v, date):
                    wafer_start_time = datetime.combine(v, time.min)
        except Exception as e:
            logger.debug("Failed to read wafer_start_date for lot {}: {}", target_lot_id, e)

    prev_stg_idx: int = -1
    first_snapshot_time = sorted_snapshots.item(0, "created_at")
    prev_time: datetime = (
        first_snapshot_time if isinstance(first_snapshot_time, datetime) else datetime.now()
    )

    for row in sorted_snapshots.iter_rows(named=True):
        row_time = row["created_at"]
        row_stage = row["stage"]

        stage_idx = path_index.get(row_stage, -1)
        if stage_idx == -1:
            logger.debug("Stage not found in primary path used: {}", row_stage)
            continue

        if stage_idx < prev_stg_idx:
            logger.debug(
                "Suspected re-work lot '{}': {}->{}",
                row["lot_id"],
                prev_stg_idx,
                stage_idx,
            )
            continue

        if prev_stg_idx == -1 and stage_idx > 0:
            if wafer_start_time is not None and wafer_start_time < row_time:
                stg_timestamp[0] = wafer_start_time
                stg_t_type[0] = "interpolated"
                if stage_idx > 1:
                    interpolated = interpolate_between(
                        start_time=wafer_start_time,
                        end_time=row_time,
                        start_idx=0,
                        end_idx=stage_idx,
                        weights=durations,
                    )
                    stg_timestamp.update(interpolated)
                    stg_t_type.update({k: "interpolated" for k in interpolated.keys()})
            else:
                for i in range(stage_idx):
                    stg_timestamp[i] = row_time
                    stg_t_type[i] = "interpolated"

        elif stage_idx > prev_stg_idx + 1 and row_time > prev_time:
            interpolated = interpolate_between(
                start_time=prev_time,
                end_time=row_time,
                start_idx=prev_stg_idx,
                end_idx=stage_idx,
                weights=durations,
            )
            stg_timestamp.update(interpolated)
            stg_t_type.update({k: "interpolated" for k in interpolated.keys()})

        # Record the FIRST snapshot that caught the lot at this stage as its start time.
        # If consecutive snapshots keep observing the same stage (the lot lingers), keep
        # the earliest observation rather than overwriting with the latest — the first
        # sighting is the best estimate of when the lot entered the stage, and it must
        # stay stable no matter the snapshot frequency. prev_time still advances to the
        # latest observation below, so gap interpolation and the prediction split anchor
        # correctly off the moment the lot was last seen at the previous stage.
        if stg_t_type.get(stage_idx) != "actual":
            stg_timestamp[stage_idx] = row_time
            stg_t_type[stage_idx] = "actual"

        prev_stg_idx = stage_idx
        prev_time = row_time

    #################
    # Prediction Part
    #################
    if prev_stg_idx < 0:
        raise ValueError(f"No observed stages for lot {target_lot_id} matched primary path")

    if last_stage_idx >= 0 and not sorted_snapshots.is_empty() and prev_stg_idx <= last_stage_idx:
        try:
            service = get_inference_service(model_dir)
            split_time = pd.Timestamp(prev_time)
            pred_res = service.predict_remaining_days(lot_snapshots, target_lot_id, split_time)
            if target_lot_id not in pred_res.index:
                raise KeyError(f"Target lot {target_lot_id} not present in prediction result")
            rem_time = float(pred_res.loc[target_lot_id])
            logger.debug(
                "Predicted remaining time for lot {} from stage '{}' ({}) is {:.2f} days.",
                target_lot_id,
                primary_path[prev_stg_idx],
                prev_stg_idx,
                rem_time,
            )

            # Interpolate predicted stages
            predicted_completion_time = prev_time + timedelta(days=rem_time)
            if completion_prediction_sink is not None:
                completion_prediction_sink.append(
                    LotCompletionPrediction(
                        split_time=prev_time,
                        remaining_days=rem_time,
                        predicted_completion_time=predicted_completion_time,
                        model_dir=Path(model_dir).as_posix(),
                    )
                )
            if prev_stg_idx < last_stage_idx:
                predicted = interpolate_between(
                    start_time=prev_time,
                    end_time=predicted_completion_time,
                    start_idx=prev_stg_idx,
                    end_idx=last_stage_idx + 1,
                    weights=durations,
                )
                stg_timestamp.update(predicted)
                stg_t_type.update({k: "predicted" for k in predicted.keys()})
        except Exception as e:
            logger.exception("Failed to predict remaining time for lot {}: {}", target_lot_id, e)
            rem_time = _predict_remaining_days_fallback()
            logger.warning(
                f"Using fallback remaining time for lot {target_lot_id}: {rem_time:.2f} days"
            )
            predicted_completion_time = prev_time + timedelta(days=rem_time)
            if prev_stg_idx < last_stage_idx:
                predicted = interpolate_between(
                    start_time=prev_time,
                    end_time=predicted_completion_time,
                    start_idx=prev_stg_idx,
                    end_idx=last_stage_idx + 1,
                    weights=durations,
                )
                stg_timestamp.update(predicted)
                stg_t_type.update({k: "predicted" for k in predicted.keys()})

    ordered_indices = sorted(stg_timestamp)
    for idx, next_idx in zip(ordered_indices, ordered_indices[1:], strict=False):
        stg_end_timestamp[idx] = stg_timestamp[next_idx]
    if predicted_completion_time is not None and last_stage_idx in stg_timestamp:
        if predicted_completion_time > stg_timestamp[last_stage_idx]:
            stg_end_timestamp[last_stage_idx] = predicted_completion_time
        else:
            logger.warning(
                "Predicted completion time {} is not after final stage start {} for lot {}",
                predicted_completion_time,
                stg_timestamp[last_stage_idx],
                target_lot_id,
            )

    # Build results
    results: list[dict[str, str]] = []
    for stage_idx, stage_name in enumerate(primary_path):
        if stage_idx in stg_timestamp:
            row = {
                "stage_name": stage_name,
                "start_time": stg_timestamp[stage_idx].isoformat(),
                "stage_type": stg_t_type.get(stage_idx, "predicted"),
            }
            if stage_idx in stg_end_timestamp:
                row["end_time"] = stg_end_timestamp[stage_idx].isoformat()
            results.append(row)

    return results
