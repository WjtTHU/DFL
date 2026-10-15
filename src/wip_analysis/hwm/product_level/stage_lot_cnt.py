from datetime import datetime
from typing import Any, Optional

from loguru import logger

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql
from wip_analysis.hwm.stage.utils.lot_progress_readiness import ensure_lot_progress_populated

from .primary_path import get_pid_primary_path


def get_pid_stage_stats(
    product_id: Optional[str],
    snapshot_start: Optional[datetime] = None,
    snapshot_end: Optional[datetime] = None,
    raise_on_error: bool = False,
) -> list[dict]:
    """
    Aggregate lot counts and wafer sums by stage_time at day level for a given product.
    Returns all stage_time estimates (including multiple values for same day due to snapshot updates),
    sorted first by stage_time, then by snapshot_cut.

    Args:
        product_id: The product ID to analyze
        snapshot_start: Optional lower bound for analysis snapshot filtering
        snapshot_end: Optional upper bound for analysis snapshot filtering

    Returns:
        List of dictionaries containing calendar_time, snapshot_cut, stage_lot_cnt,
        and stage_wafer_sum.
    """
    product_id = product_id or "ALL"
    query_product_id = None if product_id == "ALL" else product_id
    try:
        if query_product_id is not None:
            ensure_lot_progress_populated(
                product_id=query_product_id,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
                context="stage statistics",
            )

        primary_path = get_pid_primary_path(
            product_id,
            path_start_time=snapshot_start,
            path_end_time=snapshot_end,
        )
        stage_to_index = {stage: idx for idx, stage in enumerate(primary_path)}

        # Query to aggregate lot counts by day-level stage_time and snapshot_cut
        query = render_sql(
            "get_pid_stage_stats.sql.j2",
            {
                "product_id": query_product_id,
                "snapshot_start": snapshot_start.isoformat() if snapshot_start else None,
                "snapshot_end": snapshot_end.isoformat() if snapshot_end else None,
            },
        )
        df = fetch_data(query)

        if df.is_empty():
            logger.warning(f"No stage statistics found for product {product_id}")
            return []

        result_by_key: dict[tuple[Any, Any], dict[str, Any]] = {}

        for row in df.iter_rows(named=True):
            stage_idx = stage_to_index.get(row["stage_name"])
            if stage_idx is None:
                continue

            key = (row["calendar_time"], row["snapshot_cut"])
            result_row = result_by_key.get(key)
            if result_row is None:
                result_row = {
                    "calendar_time": key[0],
                    "snapshot_cut": key[1],
                    "stage_lot_cnt": [0] * len(primary_path),
                    "stage_wafer_sum": [0] * len(primary_path),
                }
                result_by_key[key] = result_row

            result_row["stage_lot_cnt"][stage_idx] = int(row["lot_count"] or 0)
            result_row["stage_wafer_sum"][stage_idx] = int(row.get("wafer_count") or 0)

        result_data = [
            result_by_key[key] for key in sorted(result_by_key, key=lambda item: (item[0], item[1]))
        ]

        for row in result_data:
            logger.opt(lazy=True).trace(
                "{} | {} | {}",
                lambda r=row: r["calendar_time"],
                lambda r=row: r["snapshot_cut"],
                lambda r=row: {
                    "stage_lot_cnt": r["stage_lot_cnt"],
                    "stage_wafer_sum": r["stage_wafer_sum"],
                },
            )
        logger.info(
            f"Retrieved {len(result_data)} stage statistics records for product {product_id}"
        )
        return result_data

    except Exception as e:
        logger.exception(f"Error retrieving stage statistics for product {product_id}: {e}")
        if raise_on_error:
            raise
        return []


if __name__ == "__main__":
    import time

    logger.add(f"logs/primary_path_intervals/{time.strftime('%Y%m%d_%H%M%S')}.log")
    # get_primary_path_durations("MiwdxnTnc/kk6vii7HWoSw==")
    get_pid_stage_stats("0NQWrgOqH3MAfHOrBFLADg==")
