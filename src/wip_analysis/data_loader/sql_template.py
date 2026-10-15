from datetime import date, datetime, time
from functools import lru_cache
from typing import Any, Mapping

import yaml
from jinja2 import Environment, FileSystemLoader
from loguru import logger


def _sql_literal(val: Any) -> str:
    """
    Render a Python value as a SQL literal (PostgreSQL compatible).

    - None -> NULL
    - str -> single quoted with embedded single quotes escaped as ''
    - date/datetime/time -> ISO string single quoted
    - bool -> TRUE/FALSE
    - numbers -> as-is
    - list/tuple/set -> parenthesized comma-separated list of literals
    - otherwise -> str(val) single quoted
    """
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, (datetime, date, time)):
        return f"'{val.isoformat()}'"
    if isinstance(val, (list, tuple, set)):
        return "(" + ", ".join(_sql_literal(v) for v in val) + ")"
    # default to string-like
    s = str(val)
    s = s.replace("'", "''")
    return f"'{s}'"


@lru_cache(maxsize=1)
def _load_sql_conf(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        conf = yaml.safe_load(f)
    return conf


@lru_cache(maxsize=1)
def _get_env_and_context(config_path: str) -> tuple[Environment, dict[str, dict]]:
    conf = _load_sql_conf(config_path)
    db = conf.get("database", {})

    tables = dict(db.get("tables", {}))
    columns = dict(db.get("columns", {}))

    templates_dir = db.get("templates", {}).get("base_dir", "sql/templates")
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
        keep_trailing_newline=False,
    )
    env.filters["sql"] = _sql_literal

    base_ctx = {"tables": tables, "columns": columns}
    return env, base_ctx


def render_sql(
    template_name: str,
    ctx: Mapping[str, Any] | None = None,
    *,
    config_path: str = "config/sql_base.yaml",
) -> str:
    """
    Render a SQL template with provided context.

    Context variables are provided at the top-level (e.g., product_id, lot_id, snapshot_cut).
    Table and column mappings are injected automatically under `tables` and `columns`.

    Args:
        template_name: Name of the SQL template file (e.g., "get_product_stage_stats.sql.j2")
        ctx: Optional dict of context variables for rendering the template. would override default `tables`/`columns` ctx upon conflict
        config_path: Path to the SQL configuration YAML file
    Returns:
        Rendered SQL string
    """
    env, base_ctx = _get_env_and_context(config_path)
    try:
        tmpl = env.get_template(template_name)
    except Exception as e:
        logger.exception(f"Failed to load SQL template '{template_name}': {e}")
        raise

    render_ctx: dict[str, Any] = dict(base_ctx)
    if ctx:
        render_ctx.update(ctx)

    sql_text = tmpl.render(**render_ctx).strip()
    logger.opt(lazy=True).debug("Rendered SQL from '{}'", lambda: template_name)
    logger.opt(lazy=True).trace("Query: \n{}", lambda: sql_text)
    return sql_text
