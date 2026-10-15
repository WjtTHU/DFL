from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, TypedDict

import pandas as pd
from loguru import logger

from wip_analysis.data_loader.postgres_loader import execute_batch, execute_batch_async, fetch_data
from wip_analysis.data_loader.sql_template import render_sql
from wip_pred.inference.predict import DEFAULT_MODEL_DIR, get_inference_service


class LatestSnapshotCut(TypedDict):
    lot_id: str
    snapshot_cut: datetime


def _group_prediction_cuts(
    prediction_cuts: list[LatestSnapshotCut],
) -> dict[datetime, list[str]]:
    grouped: dict[datetime, list[str]] = defaultdict(list)
    for row in prediction_cuts:
        lot_id = str(row["lot_id"])
        snapshot_cut = row["snapshot_cut"]
        if lot_id not in grouped[snapshot_cut]:
            grouped[snapshot_cut].append(lot_id)
    return dict(grouped)


def persist_lot_completion_prediction_rows(
    rows_to_upsert: Sequence[Sequence[Any]],
    *,
    async_write: bool = False,
) -> int:
    """Persist already-computed lot completion prediction rows."""
    if not rows_to_upsert:
        return 0

    upsert_query = render_sql("upsert_lot_completion_predictions.sql.j2")
    if async_write:
        execute_batch_async(upsert_query, rows_to_upsert)
    else:
        execute_batch(upsert_query, rows_to_upsert)
    return len(rows_to_upsert)


def compute_lot_completion_predictions(
    product_id: str,
    prediction_cuts: list[LatestSnapshotCut],
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> int:
    """Compute and persist completion predictions for missing lot+snapshot cuts."""
    if not prediction_cuts:
        return 0

    service = get_inference_service(model_dir)
    model_dir_text = Path(model_dir).as_posix()
    rows_to_upsert: list[tuple[Any, ...]] = []

    for snapshot_cut, lot_ids in _group_prediction_cuts(prediction_cuts).items():
        split_time = pd.Timestamp(snapshot_cut)
        snapshot_start = split_time - pd.Timedelta(days=service.window_days)
        query = render_sql(
            "get_lid_snapshots_ranged.sql.j2",
            {
                "lot_id": None,
                "product_ids": [product_id],
                "snapshot_start": snapshot_start.to_pydatetime(),
                "snapshot_cut": snapshot_cut,
                "extra_feat_cols": service.extra_feat_cols,
            },
        )
        lot_snapshots = fetch_data(query)
        if lot_snapshots.is_empty():
            logger.warning(
                "No snapshots found for completion prediction product_id={} snapshot_cut={}",
                product_id,
                snapshot_cut,
            )
            continue

        predictions = service.predict_remaining_days(lot_snapshots, lot_ids, split_time)
        if predictions.empty:
            logger.warning(
                "Model returned no completion predictions product_id={} snapshot_cut={} lot_count={}",
                product_id,
                snapshot_cut,
                len(lot_ids),
            )
            continue

        for lot_id, remaining_days in predictions.items():
            if pd.isna(remaining_days):
                continue
            remaining_days_float = float(remaining_days)
            predicted_completion_time = (
                split_time + pd.Timedelta(days=remaining_days_float)
            ).to_pydatetime()
            rows_to_upsert.append(
                (
                    str(lot_id),
                    snapshot_cut,
                    product_id,
                    snapshot_cut,
                    remaining_days_float,
                    predicted_completion_time,
                    model_dir_text,
                )
            )

    if not rows_to_upsert:
        return 0

    return persist_lot_completion_prediction_rows(rows_to_upsert)
