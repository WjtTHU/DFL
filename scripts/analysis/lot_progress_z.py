#!/usr/bin/env python
import sys
import time
from pathlib import Path

from loguru import logger

from wip_analysis.hwm.lot_level.stage_dur_z import (
    LotProgressZNoDataError,
    build_lot_progress_z_plot,
)


def plot_lot_progress_z_heatmap(lot_id: str, output_dir: str = "./outputs/lot_progress_z"):
    logger.info(f"Fetching data for lot {lot_id}")
    try:
        plot = build_lot_progress_z_plot(lot_id, refresh=False)
    except LotProgressZNoDataError as e:
        logger.error(str(e))
        return

    logger.info(
        "Processing {} snapshots, {} stages, {} drawable z values",
        plot.snapshot_count,
        plot.stage_count,
        plot.z_value_count,
    )
    for warning in plot.warnings:
        logger.warning(warning)

    fig = plot.fig

    # Save to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Sanitize lot_id for filename (replace slashes and other problematic chars)
    safe_lot_id = lot_id.replace("/", "_").replace("\\", "_")
    output_file = output_path / f"{safe_lot_id}.html"

    fig.write_html(str(output_file))
    logger.info(f"Saved heatmap to {output_file}")

    return fig


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lot_progress_z_heatmap.py <lot_id>")
        sys.exit(1)

    logger.add(f"logs/lot_progress_z/{time.strftime('%Y%m%d_%H%M%S')}.log")
    lot_id = sys.argv[1]
    plot_lot_progress_z_heatmap(lot_id)
