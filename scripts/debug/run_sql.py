#!/usr/bin/env python
"""Run ad-hoc SQL queries with default project configuration for debugging.

Examples:
    # 1) Run a template with default config
    uv run scripts/debug/run_sql.py --template get_pid_primary_path.sql.j2 \
      --ctx-json '{"product_id": "xxx"}'

        # 1.1) Override config files explicitly
        uv run scripts/debug/run_sql.py --template get_pid_primary_path.sql.j2 \
            --sql-config config/sql_base.yaml --db-config config/postgres.yaml \
            --ctx-json '{"product_id": "xxx"}'

    # 2) Run a raw SQL file
    uv run scripts/debug/run_sql.py --sql-file /tmp/debug.sql

    # 3) Run inline SQL
    uv run scripts/debug/run_sql.py --sql "SELECT now() AS ts"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl
import psycopg
import yaml
from loguru import logger

from wip_analysis.data_loader.sql_template import render_sql


def _load_sql(args: argparse.Namespace) -> str:
    if args.template:
        ctx: dict[str, Any] = {}
        if args.ctx_json:
            try:
                parsed = json.loads(args.ctx_json)
                if not isinstance(parsed, dict):
                    raise ValueError("--ctx-json must be a JSON object")
                ctx = parsed
            except Exception as exc:
                raise ValueError(f"Invalid --ctx-json: {exc}") from exc
        return render_sql(args.template, ctx=ctx, config_path=args.sql_config)

    if args.sql_file:
        sql_path = Path(args.sql_file)
        if not sql_path.exists():
            raise FileNotFoundError(f"SQL file not found: {args.sql_file}")
        return sql_path.read_text(encoding="utf-8").strip()

    if args.sql:
        return args.sql.strip()

    raise ValueError("One of --template, --sql-file, or --sql is required")


def _fetch_direct(query: str, config_path: str = "config/postgres.yaml") -> pl.DataFrame:
    """Single-connection fetch without connection pool, suitable for one-shot scripts."""
    with open(config_path, "r") as f:
        db_cfg = yaml.safe_load(f)["database"]
    conninfo = (
        f"postgresql://{db_cfg['user']}:{db_cfg['password']}"
        f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['dbname']}"
    )
    with psycopg.connect(conninfo) as conn:
        return pl.read_database(query=query, connection=conn)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SQL for debugging with project default config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sql-config",
        default="config/sql_base.yaml",
        help="Path to SQL template config YAML (default: config/sql_base.yaml)",
    )
    parser.add_argument(
        "--db-config",
        default="config/postgres.yaml",
        help="Path to DB connection config YAML (default: config/postgres.yaml)",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--template", help="Template name under sql/templates, e.g. get_pids_all.sql.j2"
    )
    group.add_argument("--sql-file", help="Path to a raw .sql file")
    group.add_argument("--sql", help="Inline SQL text")

    parser.add_argument(
        "--ctx-json",
        help='JSON object for template context, e.g. \'{"product_id": "xxx"}\'',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="How many rows to print (default: 50, use -1 for all)",
    )
    parser.add_argument(
        "--show-sql",
        action="store_true",
        help="Print rendered SQL before executing",
    )

    args = parser.parse_args()

    try:
        if not Path(args.sql_config).exists():
            logger.error(f"SQL config file not found: {args.sql_config}")
            return 1

        if not Path(args.db_config).exists():
            logger.error(f"DB config file not found: {args.db_config}")
            return 1

        sql_text = _load_sql(args)
        if not sql_text:
            logger.error("SQL is empty")
            return 1

        if args.show_sql:
            print("===== SQL =====")
            print(sql_text)
            print("===============")

        df = _fetch_direct(sql_text, config_path=args.db_config)

        print(f"Rows: {df.height}, Cols: {df.width}")
        if df.is_empty():
            print("Result is empty")
            return 0

        with pl.Config(tbl_rows=args.limit, tbl_cols=-1, tbl_width_chars=-1, fmt_str_lengths=200):
            print(df)
        return 0

    except Exception as exc:
        logger.exception(f"Failed to execute SQL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
