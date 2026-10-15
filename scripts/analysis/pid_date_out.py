#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import date

import polars as pl
from loguru import logger

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql


def _query_finish_lots(
    product_id: str,
    target_day: date,
) -> pl.DataFrame:
    query = render_sql(
        "get_pid_finish_lots_on_date.sql.j2",
        {"product_id": product_id, "target_date": target_day},
    )
    return fetch_data(query)


def _query_latest_wafers(
    product_id: str,
) -> pl.DataFrame:
    query = render_sql(
        "get_pid_latest_wafers.sql.j2",
        {"product_id": product_id},
    )
    return fetch_data(query)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", required=True, help="Product ID (e.g. HB123)")
    parser.add_argument("--date", required=True, dest="day", help="Target date (YYYY-MM-DD)")

    args = parser.parse_args()
    product_id: str = args.pid
    target_day = date.fromisoformat(args.day)

    finish_df = _query_finish_lots(
        product_id,
        target_day,
    )
    if finish_df.is_empty():
        logger.warning(f"No lots found for pid={product_id} with predicted finish date={args.day}")
        return 0

    wafers_df = _query_latest_wafers(product_id)
    if wafers_df.is_empty():
        logger.warning(f"No snapshot wafer data found for product_id={product_id}")
        return 2

    joined = (
        finish_df.join(wafers_df, on="lot_id", how="left")
        .with_columns(pl.col("current_wafer_qty").fill_null(0))
        .with_columns(pl.col("current_wafer_qty").cast(pl.Int64))
    )

    total_wafers = int(joined.select(pl.col("current_wafer_qty").sum()).item())
    lot_count = int(joined.height)

    logger.info(
        f"pid={product_id} date={target_day.isoformat()} lots={lot_count} wafers={total_wafers}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
