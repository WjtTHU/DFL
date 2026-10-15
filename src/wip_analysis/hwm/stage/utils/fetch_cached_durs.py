from typing import Optional

import polars as pl
from loguru import logger

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql


def fetch_cached_stg_dur(
    stage_name: Optional[str],
    product_id: Optional[str],
    lot_id: Optional[str],
    cut_time: Optional[str],
    snapshot_start=None,
    snapshot_end=None,
) -> pl.DataFrame:
    """
    Fetch stage duration data from lot_progress_table.

    Args:
        stage_name (str, optional): Name of the stage to query. If None, returns all stages.
        product_id (str, optional): Product ID to filter by. If None, returns all products.
        lot_id (str, optional): Lot ID to filter by. If None, returns all lots.
        cut_time (str, optional): Cut-off time to filter data. If None, no cut-off is applied.

    Returns:
        out (pl.DataFrame): A DataFrame containing stage duration data from lot_progress_table. cols
          - product_id: identifier of the product
          - lot_id: identifier of the lot
          - stage_name: name of the stage
          - start_time: start time of the stage (stage_time)
          - end_time: end time of the stage (stage_end)
          - duration_seconds: duration of the stage in seconds
    """
    query = render_sql(
        "get_stg_durs.sql.j2",
        {
            "stage_name": stage_name,
            "product_id": product_id,
            "lot_id": lot_id,
            "cut_time": cut_time,
            "snapshot_start": snapshot_start,
            "snapshot_end": snapshot_end,
        },
    )

    try:
        df = fetch_data(query)
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch from lot_progress_table: {e}")
        raise


def fetch_cached_stg_dur_stats(
    stage_name: Optional[str],
    product_id: Optional[str],
    lot_id: Optional[str],
    cut_time: Optional[str],
    snapshot_start=None,
    snapshot_end=None,
) -> pl.DataFrame:
    """Fetch per-stage duration averages/stddevs directly from lot_progress_table."""
    query = render_sql(
        "get_stg_dur_stats.sql.j2",
        {
            "stage_name": stage_name,
            "product_id": product_id,
            "lot_id": lot_id,
            "cut_time": cut_time,
            "snapshot_start": snapshot_start,
            "snapshot_end": snapshot_end,
        },
    )

    try:
        return fetch_data(query)
    except Exception as e:
        logger.warning(f"Failed to fetch duration stats from lot_progress_table: {e}")
        raise
