#!/usr/bin/env python
"""
Simple utility to load CSV files into PostgreSQL tables with configurable column mapping.

Usage:
    uv run csv_to_db.py <csv_file> [--batch-size 10000]
"""

import argparse
from pathlib import Path
from typing import Optional

import polars as pl
from loguru import logger

from wip_analysis.data_loader.postgres_loader import execute_batch
from wip_analysis.data_loader.sql_template import render_sql


def load_csv_to_db(
    csv_path: str | Path,
    column_mapping: Optional[dict[str, str]] = None,
    batch_size: int = 1000,
) -> None:
    """
    Load a CSV file into a PostgreSQL table with optional column name mapping.

    Args:
        csv_path: Path to the CSV file to load
        column_mapping: Optional dict mapping CSV column names to DB column names
                       If None, assumes CSV columns match DB columns exactly
                       Example: {"csv_col_name": "db_col_name"}
        if_exists: What to do if table has data:
                  - "append": Insert new rows (default)
                  - "replace": Truncate table first, then insert
                  - "fail": Raise error if table exists
        batch_size: Number of rows to insert per batch

    Example:
        # Load CSV with column mapping
        load_csv_to_db(
            csv_path="data.csv",
            column_mapping={
                "Product ID": "product_id",
                "Lot Number": "lot_id",
                "Timestamp": "created_at"
            }
        )
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info(f"Loading CSV from {csv_path}")
    df = pl.read_csv(csv_path, ignore_errors=True)
    logger.info(f"Loaded {df.height} rows from CSV")

    # Apply column mapping if provided
    if column_mapping:
        logger.info(f"Applying column mapping: {column_mapping}")
        df = df.rename(column_mapping)

    # Validate required columns exist after mapping
    db_columns = df.columns
    logger.info(f"Database columns after mapping: {db_columns}")

    # Prepare insert statement using Jinja2 template
    insert_query = render_sql(
        "insert_snapshot_csv.sql.j2",
        ctx={"table_cols": db_columns},
    )

    # Insert data in batches
    total_rows = len(df)
    rows_data = df.rows()

    for i in range(0, total_rows, batch_size):
        batch = rows_data[i : i + batch_size]
        batch_end = min(i + batch_size, total_rows)

        try:
            execute_batch(
                insert_query,
                params_list=batch,
            )
            logger.info(f"Inserted rows {i + 1} to {batch_end} / {total_rows}")

        except Exception as e:
            logger.error(f"Failed to insert batch {i + 1} to {batch_end}: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Load CSV files into PostgreSQL with configurable column mapping"
    )
    parser.add_argument("csv_file", help="Path to the CSV file to load")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for inserts")

    args = parser.parse_args()

    # Execute the load
    load_csv_to_db(
        csv_path=args.csv_file,
        column_mapping=None,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
