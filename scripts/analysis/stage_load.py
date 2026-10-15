#!/usr/bin/env python
from pathlib import Path

import plotly.graph_objects as go
from loguru import logger

from wip_analysis.hwm.product_level.stage_load import get_pid_stage_load


def create_visualization(product_id: str, output_dir: Path):
    """
    Create interactive visualization for stage load ratios.
    Single bar chart with snapshot selector and time slider (K.I.S.S.).

    Args:
        product_id: Product ID to analyze
        output_dir: Directory to save HTML output
    """
    logger.info(f"Fetching stage load data for product: {product_id}")
    data = get_pid_stage_load(product_id)

    if not data["load_ratios"]:
        logger.info(f"No load ratio data found for product {product_id}")
        return

    stage_order = data["stage_order"]
    logger.info(f"Found {len(data['load_ratios'])} snapshots with {len(stage_order)} stages")

    # Rolling window options
    rolling_windows = ["rolling_7d", "rolling_1m", "rolling_3m", "rolling_6m", "rolling_1y"]
    window_labels = ["7 Days", "1 Month", "3 Months", "6 Months", "1 Year"]

    # Color palette for stages
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ] * 10  # Repeat if more than 10 stages

    # Create figure with single bar chart
    fig = go.Figure()

    # Start with first snapshot, first window, first time point
    snapshot = data["load_ratios"][0]
    stage_metrics = snapshot["stage_metrics"]
    window_key = rolling_windows[0]

    # Build initial bar chart (first time point)
    bar_x = []
    bar_y = []
    bar_colors = []
    for stage_name in stage_order:
        if stage_name not in stage_metrics:
            continue
        load_values = stage_metrics[stage_name].get(window_key, [])
        if load_values:
            bar_x.append(stage_name)
            bar_y.append(load_values[0])
            bar_colors.append(colors[stage_order.index(stage_name)])

    fig.add_trace(
        go.Bar(
            x=bar_x,
            y=bar_y,
            marker=dict(color=bar_colors),
            hovertemplate="<b>%{x}</b><br>Load: %{y:.2f}<extra></extra>",
        )
    )

    # Helper function to build bar data for any snapshot/window/time combination
    def build_bar_data(snap_idx: int, window_idx: int, time_idx: int):
        """Build bar chart data for specific snapshot, window, and time point."""
        snapshot = data["load_ratios"][snap_idx]
        calendar_times_list = snapshot["calendar_times"]
        stage_metrics = snapshot["stage_metrics"]
        window_key = rolling_windows[window_idx]

        bar_x = []
        bar_y = []
        bar_colors = []
        for stage_name in stage_order:
            if stage_name not in stage_metrics:
                continue
            load_values = stage_metrics[stage_name].get(window_key, [])
            if time_idx < len(load_values):
                bar_x.append(stage_name)
                bar_y.append(load_values[time_idx])
                bar_colors.append(colors[stage_order.index(stage_name)])

        calendar_time = (
            calendar_times_list[time_idx] if time_idx < len(calendar_times_list) else None
        )
        time_str = (
            calendar_time.strftime("%Y-%m-%d %H:%M")
            if calendar_time and hasattr(calendar_time, "strftime")
            else str(calendar_time)
            if calendar_time
            else "N/A"
        )

        return bar_x, bar_y, bar_colors, time_str, len(calendar_times_list)

    # Create dropdown for snapshot selection
    snapshot_buttons = []
    for snap_idx, snapshot in enumerate(data["load_ratios"]):
        snapshot_cut = snapshot["snapshot_cut"]
        snapshot_label = (
            snapshot_cut.strftime("%Y-%m-%d %H:%M")
            if snapshot_cut and hasattr(snapshot_cut, "strftime")
            else str(snapshot_cut)
            if snapshot_cut
            else "N/A"
        )

        # Reset to first window and first time point when changing snapshot
        bar_x, bar_y, bar_colors, time_str, _ = build_bar_data(snap_idx, 0, 0)

        snapshot_buttons.append(
            dict(
                label=snapshot_label,
                method="update",
                args=[
                    {"x": [bar_x], "y": [bar_y], "marker.color": [bar_colors]},
                    {
                        "title.text": f"Stage Load Ratios<br><sub>Snapshot: {snapshot_label} | Window: {window_labels[0]} | Time: {time_str}</sub>"
                    },
                ],
            )
        )

    # Create dropdown for rolling window selection
    window_buttons = []
    for window_idx, window_label in enumerate(window_labels):
        # Reset to first time point when changing window (keep current snapshot as 0)
        bar_x, bar_y, bar_colors, time_str, _ = build_bar_data(0, window_idx, 0)

        snapshot_label = (
            data["load_ratios"][0]["snapshot_cut"].strftime("%Y-%m-%d %H:%M")
            if hasattr(data["load_ratios"][0]["snapshot_cut"], "strftime")
            else str(data["load_ratios"][0]["snapshot_cut"])
        )

        window_buttons.append(
            dict(
                label=window_label,
                method="update",
                args=[
                    {"x": [bar_x], "y": [bar_y], "marker.color": [bar_colors]},
                    {
                        "title.text": f"Stage Load Ratios<br><sub>Snapshot: {snapshot_label} | Window: {window_label} | Time: {time_str}</sub>"
                    },
                ],
            )
        )

    # Create slider for calendar time selection (progress bar)
    first_snapshot = data["load_ratios"][0]
    n_times = len(first_snapshot["calendar_times"])
    slider_steps = []

    snapshot_label = (
        first_snapshot["snapshot_cut"].strftime("%Y-%m-%d %H:%M")
        if hasattr(first_snapshot["snapshot_cut"], "strftime")
        else str(first_snapshot["snapshot_cut"])
    )

    for time_idx in range(n_times):
        bar_x, bar_y, bar_colors, time_str, _ = build_bar_data(0, 0, time_idx)

        slider_steps.append(
            dict(
                method="update",
                args=[
                    {"x": [bar_x], "y": [bar_y], "marker.color": [bar_colors]},
                    {
                        "title.text": f"Stage Load Ratios<br><sub>Snapshot: {snapshot_label} | Window: {window_labels[0]} | Time: {time_str}</sub>"
                    },
                ],
                label=time_str,
            )
        )

    # Update layout
    initial_time_str = (
        first_snapshot["calendar_times"][0].strftime("%Y-%m-%d %H:%M")
        if hasattr(first_snapshot["calendar_times"][0], "strftime")
        else str(first_snapshot["calendar_times"][0])
    )

    fig.update_layout(
        title=dict(
            text=f"Stage Load Ratios<br><sub>Snapshot: {snapshot_label} | Window: {window_labels[0]} | Time: {initial_time_str}</sub>",
            y=0.98,  # Position title higher to avoid overlap
            yanchor="top",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Stage",
        yaxis_title="Load Ratio",
        height=800,  # Increased height to accommodate dropdowns and slider
        showlegend=False,
        hovermode="closest",
        xaxis=dict(tickangle=-45),
        margin=dict(t=120, b=150),  # Add top and bottom margins
        updatemenus=[
            # Snapshot selector
            dict(
                buttons=snapshot_buttons,
                direction="down",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=1.08,  # Moved down to avoid overlap
                yanchor="top",
                bgcolor="lightgray",
                bordercolor="gray",
                borderwidth=1,
            ),
            # Rolling window selector
            dict(
                buttons=window_buttons,
                direction="down",
                showactive=True,
                x=0.25,
                xanchor="left",
                y=1.08,  # Moved down to avoid overlap
                yanchor="top",
                bgcolor="lightblue",
                bordercolor="blue",
                borderwidth=1,
            ),
        ],
        sliders=[
            dict(
                active=0,
                steps=slider_steps,
                x=0.1,
                len=0.8,
                xanchor="left",
                y=0,
                yanchor="top",
                currentvalue=dict(
                    prefix="<b>Calendar Time: </b>",
                    visible=True,
                    xanchor="center",
                    font=dict(size=14),
                ),
                pad=dict(t=50, b=10),
            )
        ],
    )

    # Add annotations for dropdowns
    fig.add_annotation(
        text="<b>Snapshot:</b>",
        x=0.02,
        xref="paper",
        xanchor="left",
        y=1.11,  # Adjusted to match dropdown position
        yref="paper",
        yanchor="bottom",
        showarrow=False,
        font=dict(size=12),
    )
    fig.add_annotation(
        text="<b>Window:</b>",
        x=0.25,
        xref="paper",
        xanchor="left",
        y=1.11,  # Adjusted to match dropdown position
        yref="paper",
        yanchor="bottom",
        showarrow=False,
        font=dict(size=12),
    )

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_pid = product_id.replace("/", "_").replace("\\", "_")
    output_file = output_dir / f"{safe_pid}.html"
    fig.write_html(str(output_file))
    logger.info(f"Visualization saved to: {output_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize stage load ratios")
    parser.add_argument("product_id", type=str, help="Product ID to analyze")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "pid_stage_load",
        help="Output directory for HTML files",
    )

    args = parser.parse_args()
    create_visualization(args.product_id, args.output_dir)


if __name__ == "__main__":
    main()
