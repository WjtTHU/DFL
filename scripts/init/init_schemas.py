#!/usr/bin/env python
"""
Initialize database schemas for a specific environment configuration.

Usage:
    uv run scripts/populate/init_schemas.py --config config/environments/custom.yaml
"""

import argparse
from pathlib import Path

from loguru import logger

from wip_analysis.data_loader.postgres_loader import execute_query


def render_schema(schema_name: str, config_path: str) -> str:
    """
    Render a SQL schema template using the specified configuration.

    Args:
        schema_name: Name of the schema template file (e.g., "product.sql.j2")
        config_path: Path to the configuration YAML file

    Returns:
        Rendered SQL string
    """
    # Schema templates are in sql/schemas/ instead of sql/templates/
    # We need to temporarily override the templates directory
    from jinja2 import Environment, FileSystemLoader

    from wip_analysis.data_loader.sql_template import _load_sql_conf

    conf = _load_sql_conf(config_path)
    db = conf.get("database", {})

    tables = dict(db.get("tables", {}))
    columns = dict(db.get("columns", {}))

    # Use sql/schemas as template directory for schemas
    schema_dir = "sql/schemas"
    env = Environment(
        loader=FileSystemLoader(schema_dir),
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
        keep_trailing_newline=False,
    )

    template = env.get_template(schema_name)
    sql_text = template.render(tables=tables, columns=columns).strip()

    logger.opt(lazy=True).trace("Rendered schema '{}':\n{}", lambda: schema_name, lambda: sql_text)
    return sql_text


def init_schema(schema_name: str, config_path: str, dry_run: bool = False):
    """
    Initialize a single schema in the database.

    Args:
        schema_name: Name of the schema template (e.g., "product.sql.j2")
        config_path: Path to the configuration YAML file
        dry_run: If True, only print the SQL without executing
    """
    logger.info(f"Initializing schema: {schema_name}")

    sql = render_schema(schema_name, config_path)

    if dry_run:
        logger.info(f"[DRY RUN] Would execute:\n{sql}")
        return

    try:
        execute_query(sql)
        logger.success(f"✓ Schema '{schema_name}' initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize schema '{schema_name}': {e}")
        raise


def init_all_schemas(config_path: str, dry_run: bool = False):
    """
    Initialize all schemas in sql/schemas/ directory.

    Args:
        config_path: Path to the configuration YAML file
        dry_run: If True, only print the SQL without executing
    """
    schema_dir = Path("sql/schemas")
    schema_files = sorted(schema_dir.glob("*.sql.j2"))

    if not schema_files:
        logger.warning(f"No schema templates found in {schema_dir}")
        return

    logger.info(f"Found {len(schema_files)} schema templates")
    logger.info(f"Using configuration: {config_path}")

    for schema_file in schema_files:
        init_schema(schema_file.name, config_path, dry_run=dry_run)

    logger.success(f"✓ All schemas initialized using config: {config_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialize database schemas for a specific environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

# Use custom config file
uv run scripts/populate/init_schemas.py --config config/environments/custom.yaml

# Dry run to see what would be executed
uv run scripts/populate/init_schemas.py --config config/environments/custom.yaml --dry-run
        """,
    )

    parser.add_argument("--config", type=str, help="Path to custom configuration YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    parser.add_argument(
        "--schema",
        type=str,
        help="Initialize only a specific schema (e.g., 'product.sql.j2'). If not specified, all schemas will be initialized.",
    )

    args = parser.parse_args()

    config_path = args.config

    # Verify config file exists
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1

    logger.info(f"Initializing schemas with configuration: {config_path}")

    try:
        if args.schema:  # Initialize single schema
            init_schema(args.schema, config_path, dry_run=args.dry_run)
        else:  # Initialize all schemas
            init_all_schemas(config_path, dry_run=args.dry_run)
        return 0
    except Exception as e:
        logger.exception(f"Schema initialization failed: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
