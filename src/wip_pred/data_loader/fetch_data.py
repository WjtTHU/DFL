from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql

_BASE_COLS = {
    "product_id",
    "lot_id",
    "parent_lot_id",
    "lot_status",
    "stage",
    "current_wafer_qty",
    "created_at",
}


def _extra_feat_cols_from_embed(embed_config: dict | None) -> list[str]:
    """Extract feature columns from embed config that are not already in the base SELECT."""
    if not embed_config:
        return []
    cols: list[str] = []
    for col in embed_config.get("numeric_feat_cols", []):
        if col not in _BASE_COLS and col not in cols:
            cols.append(col)
    for cat in embed_config.get("categorical_feat_cols", []):
        name = cat["name"] if isinstance(cat, dict) else cat
        if name not in _BASE_COLS and name not in cols:
            cols.append(name)
    return cols


def fetch_raw_data(
    sql_template: str = "get_full_data.sql.j2",
    config_path: str = "config/postgres.yaml",
    embed_config: dict | None = None,
    snapshot_start=None,
    snapshot_end=None,
):
    extra_cols = _extra_feat_cols_from_embed(embed_config)
    ctx = {
        "extra_feat_cols": extra_cols,
        "snapshot_start": snapshot_start,
        "snapshot_end": snapshot_end,
    }
    df = fetch_data(render_sql(sql_template, ctx=ctx), config_path).to_pandas()
    return df
