import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql


def get_wafer_sum_df(df):
    wafer_sum_df = df.groupby(["stage", "created_at"])["current_wafer_qty"].sum().reset_index()
    return wafer_sum_df


def plot_bars(pid: str, stage_counts: pd.DataFrame, max_active_wafer_sum: pd.DataFrame):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0, row_heights=[0.4, 0.6]
    )
    bar = px.bar(stage_counts, y="count", x="stage")
    for trace in bar.data:
        fig.add_trace(trace, row=1, col=1)

    fig.add_hline(
        y=0.6 * stage_counts["count"].max(),
        line=dict(color="red", width=1, dash="dash"),
        annotation_text="60%",
        annotation_position="right",
        row=1,
        col=1,
    )

    fig.add_hline(
        y=0.7 * stage_counts["count"].max(),
        line=dict(color="red", width=1, dash="dash"),
        annotation_text="70%",
        annotation_position="right",
        row=1,
        col=1,
    )

    fig.add_hline(
        y=0.8 * stage_counts["count"].max(),
        line=dict(color="red", width=1, dash="dash"),
        annotation_text="80%",
        annotation_position="right",
        row=1,
        col=1,
    )

    bar = px.bar(max_active_wafer_sum, x="stage", y="current_wafer_qty")
    for trace in bar.data:
        fig.add_trace(trace, row=2, col=1)

    fig.add_hline(
        y=200,
        line=dict(color="red", width=1, dash="dash"),
        row=2,
        col=1,
    )

    fig.update_layout(title_text="Frequency & Active wafer count", showlegend=False)

    fig.update_xaxes(title_text="Stage", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Active wafer count", row=2, col=1)

    fig.write_html(f"outputs/bottleneck/wafer_qty_by_day/{pid.replace('/', '_')}.html")

    return fig


def get_bottleneck_candidates(
    pid: str = "ALL",
    status: str = "ALL",
    ratio_to_max: float = 0.6,
    top_k: int = 10,
    plot: bool = False,
):
    """
    Get candidate bottleneck stages based on stage frequency and WIP level.

    Args:
        pid (str): product ID filter; use "ALL" to include all products
        status (str): lot status filter; use "ALL" to include all statuses
        ratio_to_max (float): minimum frequency ratio relative to the most frequent stage
        top_k (int): select top_k stages with frequency_ratio higher than ratio_to_max
        plot (bool): whether to plot stage count and WIP distribution

    Returns:
        List[str]: list of candidate bottleneck stages
    """
    df = fetch_data(
        render_sql("get_full_data.sql.j2"), config_path="config/postgres.yaml"
    ).to_pandas()
    if pid != "ALL":
        df = df[df["product_id"] == pid]

    if status != "ALL":
        df = df[df["lot_status"] == status]

    df = get_wafer_sum_df(df)
    stage_counts = df["stage"].value_counts().reset_index()
    stage_counts.columns = ["stage", "count"]

    stages_with_high_freq = stage_counts[
        stage_counts["count"] >= ratio_to_max * stage_counts["count"].max()
    ]["stage"].tolist()
    max_active_wafer_sum = df.groupby("stage")["current_wafer_qty"].max().reset_index()
    bottleneck_cand = (
        max_active_wafer_sum[max_active_wafer_sum["stage"].isin(stages_with_high_freq)]
        .sort_values("current_wafer_qty")["stage"]
        .head(top_k)
        .tolist()
    )

    if plot:
        plot_bars(pid, stage_counts, max_active_wafer_sum)

    return bottleneck_cand


if __name__ == "__main__":
    bottlenecks = get_bottleneck_candidates(status="Active", plot=True)
    print(bottlenecks)
