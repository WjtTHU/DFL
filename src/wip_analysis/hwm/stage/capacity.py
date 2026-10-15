from datetime import datetime
from typing import Optional

import polars as pl

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql
from wip_analysis.hwm.stage.utils.lot_progress_readiness import ensure_lot_progress_populated


def _normalize_text_filter(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_stage_filter(
    stage_name: Optional[str | list[str]],
) -> Optional[str | list[str]]:
    if stage_name is None:
        return None
    if isinstance(stage_name, str):
        return _normalize_text_filter(stage_name)

    normalized = [name.strip() for name in stage_name if name is not None and name.strip()]
    return normalized or None


def get_stage_capacity(
    stage_name: Optional[str | list[str]] = None,
    product_id: Optional[str] = None,
    snapshot_start: Optional[datetime] = None,
    snapshot_end: Optional[datetime] = None,
    raise_on_error: bool = False,
) -> pl.DataFrame:
    """
    Get the capacity of a specific stage for a given product.

    Args:
        stage_name: Stage name or stage-name list to query. If None, retrieves data for all
            stages.
        product_id: Product ID to filter by. If None, retrieves data for all products.
        snapshot_start: Optional start time (inclusive) for stage entry time filter.
        snapshot_end: Optional end time (inclusive) for stage entry time filter.

    Returns:
        The capacity of the specified stage for the given product.
    """
    stage_name = _normalize_stage_filter(stage_name)
    product_id = _normalize_text_filter(product_id)

    if product_id is not None:
        try:
            ensure_lot_progress_populated(
                product_id=product_id,
                stage_names=stage_name,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
                context="stage capacity",
            )
        except ValueError:
            if raise_on_error:
                raise
            return pl.DataFrame()

    query = render_sql(
        "get_pid_stages_transitions.sql.j2",
        {
            "stage_name": stage_name,
            "product_id": product_id,
            "snapshot_start": snapshot_start,
            "snapshot_end": snapshot_end,
        },
    )
    return fetch_data(query)
