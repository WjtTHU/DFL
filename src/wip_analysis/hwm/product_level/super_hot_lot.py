from datetime import datetime
from datetime import time as datetime_time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql
from wip_analysis.hwm.product_level.primary_path import get_pid_primary_path
from wip_analysis.hwm.stage.utils.lot_progress_readiness import ensure_lot_progress_populated


class SuperHotLotNoDataError(ValueError):
    """Raised when no lot progress rows are available for plotting."""


def build_super_hot_lot_plot(pid: str, snapshot_cut: datetime) -> go.Figure:
    ensure_lot_progress_populated(
        product_id=pid,
        context="super hot lot plotting",
        snapshot_start=datetime.combine(snapshot_cut.date(), datetime_time.min),
        snapshot_end=datetime.combine(snapshot_cut.date(), datetime_time.max),
    )

    query = render_sql(
        "get_pid_lid_progress.sql.j2", ctx={"pid": pid, "snapshot_cut": snapshot_cut}
    )
    df = fetch_data(query).to_pandas()
    if df.empty:
        raise SuperHotLotNoDataError(
            f"No lot progress data found for product {pid} at snapshot_cut {snapshot_cut}"
        )

    primary_path = get_pid_primary_path(pid, path_end_time=snapshot_cut)

    df["accum_time"] = (
        df["enter_time"] - df.groupby("lot_id")["enter_time"].transform("min")
    ).dt.total_seconds() / (24 * 3600)
    df["stage_name"] = pd.Categorical(df["stage_name"], categories=primary_path, ordered=True)
    df = df.sort_values("stage_name")

    fig = px.line(
        df,
        x="accum_time",
        y="stage_name",
        color="lot_id",
        markers=False,
        category_orders={"stage_name": primary_path[::-1]},
    )

    fig_markers = px.scatter(
        df,
        x="accum_time",
        y="stage_name",
        color="lot_id",
        symbol="stage_type",
        category_orders={"stage_name": primary_path[::-1]},
    )

    fig.add_traces(fig_markers.data)

    fig.update_yaxes(title="Stage")
    fig.update_xaxes(title="Cumulative Days")

    return fig


def plot_accum_time_per_stage(pid: str, snapshot_cut: datetime):
    fig = build_super_hot_lot_plot(pid, snapshot_cut)

    output_dir = Path("outputs/super_hot_lot")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_dir / f"{pid.replace('/', '_')}.html")


if __name__ == "__main__":
    import time

    from loguru import logger

    logger.add(f"logs/super_hot_lot/{time.strftime('%Y%m%d_%H%M%S')}.log")
    plot_accum_time_per_stage(
        "i2M8chm9LxtBBE58a2JJJg==",
        snapshot_cut=datetime.strptime("2024-10-05 08:34:15", "%Y-%m-%d %H:%M:%S"),
    )
