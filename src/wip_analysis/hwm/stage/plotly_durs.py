from pathlib import Path
from typing import Optional

import plotly.express as px
import polars as pl
from loguru import logger

from .duration import get_stage_duration


def plot_stage_duration(
    stage_name: Optional[str] = None,
    product_id: Optional[str] = None,
    output_dir: Path = Path("./outputs/stage_dur_viz"),
) -> tuple[dict, str]:
    """
    Visualize stage duration data with statistics and scatter plot.

    Creates a scatter plot where:
    - X-axis: calendar time (entry date)
    - Y-axis: duration in hours
    - Colors: different products (if multiple products are present)

    Args:
        stage_name: Name of the stage to analyze. If None, analyzes all stages.
        product_id: Product ID to filter by. If None, includes all products.
        output_dir: Directory to save output files.

    Returns:
        Tuple of (statistics dict, path to saved HTML plot).
    """
    logger.info(f"Fetching stage duration data for stage={stage_name}, product={product_id}")
    df = get_stage_duration(
        stage_name=stage_name, product_id=product_id, lot_id=None, cut_time="3000-12-31"
    )

    if df.is_empty():
        logger.warning("No data found for the given parameters")
        return {}, ""

    df = df.with_columns((pl.col("duration_seconds") / 3600).alias("duration_hours"))

    stats = {
        "count": df.height,
        "avg_hours": df["duration_hours"].mean(),
        "std_hours": df["duration_hours"].std(),
        "min_hours": df["duration_hours"].min(),
        "max_hours": df["duration_hours"].max(),
        "p25_hours": df["duration_hours"].quantile(0.25),
        "p50_hours": df["duration_hours"].median(),
        "p75_hours": df["duration_hours"].quantile(0.75),
        "p95_hours": df["duration_hours"].quantile(0.95),
    }
    logger.info(f"Statistics calculated: {stats}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    stage_str = stage_name or "all"
    product_str = product_id or "all"
    filename = f"{stage_str}_{product_str}.html"
    output_file = output_dir / filename

    # Create scatter plot
    n_products = df["product_id"].n_unique()
    if n_products > 1:
        # Multiple products - use color to distinguish
        fig = px.scatter(
            df.to_pandas(),
            x="start_time",
            y="duration_hours",
            color="product_id",
            hover_data=["lot_id", "stage_name", "entry_date", "duration_hours"],
            title=f"Stage Duration Over Time - {stage_str}",
            labels={
                "start_time": "Calendar Time",
                "duration_hours": "Duration (hours)",
                "product_id": "Product ID",
            },
        )
    else:
        # Single product - use single color
        fig = px.scatter(
            df.to_pandas(),
            x="start_time",
            y="duration_hours",
            hover_data=["lot_id", "stage_name", "entry_date", "duration_hours"],
            title=f"Stage Duration Over Time - {stage_str}",
            labels={
                "start_time": "Calendar Time",
                "duration_hours": "Duration (hours)",
            },
        )

    fig.write_html(str(output_file))
    logger.info(f"Plot saved to {output_file}")

    return stats, str(output_file)
