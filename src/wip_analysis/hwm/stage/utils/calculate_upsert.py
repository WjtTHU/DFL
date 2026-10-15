from typing import Optional

import polars as pl
from loguru import logger

from wip_analysis.data_loader.postgres_loader import execute_query_async, fetch_data
from wip_analysis.data_loader.sql_template import render_sql


def calc_stage_dur(
    stage_name: Optional[str], product_id: Optional[str], lot_id: Optional[str], cut_time: str
) -> pl.DataFrame:
    """
    Calculate stage duration data directly from lot_progress_table.
    Note that the durations are from lot-progress table which INCLUDES both actual and
    interpolated times when stages start, but predicted stage_times are EXCLUDED.

    Args:
        stage_name (str, optional): Name of the stage to query.
        product_id (str, optional): Product ID to filter by.
        lot_id (str, optional): Lot ID to filter by.
        cut_time (str, optional): Cut-off time to upon when real-history data is considered.

    Returns:
        A DataFrame containing calculated stage duration data, cols
          - product_id: identifier of the product
          - lot_id: identifier of the lot
          - stage_name: name of the stage
          - stage_type: type of the stage
          - start_time: start time of the stage
          - end_time: end time of the stage
          - duration_seconds: duration of the stage in seconds
    """
    query = render_sql(
        "get_pid_stages_durations.sql.j2",
        {
            "stage_name": stage_name,
            "product_id": product_id,
            "lot_id": lot_id,
            "cut_time": cut_time,
        },
    )
    return fetch_data(query)


def upsert_to_cache(df: pl.DataFrame) -> None:
    """
    Upsert calculated stage duration data directly to lot_progress_table.
    Updates the stage_end and duration_seconds columns for existing records.

    Args:
        df: DataFrame containing stage duration data, schema matches that returned by calc_stage_dur().
    """
    if df.is_empty():
        logger.info("No data to upsert to lot_progress_table")
        return

    try:
        rows = df.to_dicts()

        # Render the upsert query using the template
        upsert_query = render_sql("upsert_stg_durs.sql.j2", {"rows": rows})
        execute_query_async(upsert_query)
        logger.info(f"Successfully upserted {len(df)} rows to lot_progress_table")
    except Exception as e:
        logger.exception(f"Failed to upsert to lot_progress_table: {e}")
