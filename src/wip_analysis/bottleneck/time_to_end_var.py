import plotly.graph_objects as go
from plotly.subplots import make_subplots

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql
from wip_analysis.hwm.product_level.primary_path import get_pid_primary_path


def get_time_to_end(df):
    # Sort data to ensure chronological processing per Lot
    df_sorted = df.sort_values(["lot_id", "created_at"])

    # Identify rows where the Stage ends (next stage differs) or the Lot ends (next Lot differs)
    stage_ends = df_sorted["stage"] != df_sorted["stage"].shift(-1)
    lot_ends = df_sorted["lot_id"] != df_sorted["lot_id"].shift(-1)

    # Filter to keep only the end-of-stage timestamps
    cols = ["product_id", "lot_id", "stage", "created_at"]
    result_df = df_sorted.loc[stage_ends | lot_ends, cols].copy()

    # Determine the final completion time (max reportTime) for each Lot
    completion_time = result_df.groupby("lot_id")["created_at"].transform("max")

    # Calculate remaining time from current stage end to full Lot completion in days
    result_df["time_to_end"] = (completion_time - result_df["created_at"]).dt.days

    return result_df


def plot_stage_stats(pid: str = ""):
    df = fetch_data(
        render_sql("get_full_data.sql.j2"), config_path="config/postgres.yaml"
    ).to_pandas()
    df = df[df["product_id"] == pid]
    x_axis_order = get_pid_primary_path(pid=pid)
    df = df[df["stage"].isin(x_axis_order)]

    df_processed = get_time_to_end(df)

    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.3, 0.7]
    )

    # 1. Top Subplot: Bar chart for Stage Frequency
    stage_counts = df_processed["stage"].value_counts().reset_index()
    stage_counts.columns = ["stage", "count"]

    # Sort or reindex based on provided order
    if x_axis_order:
        stage_counts = stage_counts.set_index("stage").reindex(x_axis_order).reset_index()
    else:
        stage_counts = stage_counts.sort_values("stage")

    fig.add_trace(
        go.Bar(x=stage_counts["stage"], y=stage_counts["count"], name="Frequency"), row=1, col=1
    )

    # 2. Bottom Subplot: Bar chart for Standard Deviation of time_to_end
    stds = df_processed.groupby("stage")["time_to_end"].std().reset_index()
    stds.columns = ["stage", "std"]

    # Apply the same ordering to the std data
    if x_axis_order:
        stds = stds.set_index("stage").reindex(x_axis_order).reset_index()

    fig.add_trace(go.Bar(x=stds["stage"], y=stds["std"], name="Std Dev"), row=2, col=1)

    # Update layout and axis labels
    fig.update_layout(
        title_text=f"Frequency & Std of Time to End <Product ID = {pid}>", showlegend=False
    )
    fig.update_xaxes(title_text="Stage", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Std Dev", row=2, col=1)

    fig.write_html(f"outputs/bottleneck/time_to_end_var/{pid.replace('/', '_')}.html")

    return fig


if __name__ == "__main__":
    plot_stage_stats("HBZ7NXHE58AvrwCktXDK4g==")
