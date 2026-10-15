from typing import Optional

import polars as pl
from loguru import logger

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql


def snapshot_to_seq(
    sql: Optional[str] = None,
    sql_template: Optional[str] = "get_full_data.sql.j2",
    raw_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Get the recipe sequences from the database.

    Args:
        sql (str, optional): SQL query to execute. Defaults to None.
        sql_template (str, optional): SQL query template to use. Defaults to "get_full_data.sql.j2".
        raw_df (pl.DataFrame, optional): If provided, use this DataFrame instead of executing a SQL query.
        **At least one of the three arguments must be provided.**

        DataFrame should have the following columns
          - `lot_id`: Identifier for the lot.
          - `product_id`: Identifier for the product.
          - `stage`: The stage in the recipe sequence.

    Returns:
        out (pl.DataFrame): A DataFrame with columns `lot_id`, `product_id`, `recipe_seq`.
    """
    if raw_df is not None:
        df = raw_df
    else:
        if not sql and not sql_template:
            raise ValueError("A SQL query is required. Either 'sql' or 'sql_template'.")
        query = sql or render_sql(sql_template)  # type: ignore[arg-type]
        df = fetch_data(query)

    logger.debug("Transforming snapshot data to recipe sequences")
    res = (
        df.sort("lot_id", "created_at")
        .filter(
            (pl.col("stage") != pl.col("stage").shift(1).over("lot_id"))
            | (pl.col("stage").shift(1).over("lot_id").is_null())
        )
        .group_by("lot_id", "product_id", maintain_order=True)
        .agg(pl.col("stage").alias("recipe_seq"))
    )
    logger.debug("Transformed snapshot data into {} recipe sequences", res.height)

    return res
