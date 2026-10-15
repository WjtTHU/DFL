#!/usr/bin/env python
"""
Example script to visualize stage duration data.

Usage:
    python scripts/analysis/visualize_stage_duration.py --stage "STAGE_NAME" --product "PRODUCT_ID"
    python scripts/analysis/visualize_stage_duration.py --stage "STAGE_NAME"
    python scripts/analysis/visualize_stage_duration.py --product "PRODUCT_ID"
    python scripts/analysis/visualize_stage_duration.py  # All stages and products
"""

import argparse
from pathlib import Path

from loguru import logger

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.hwm.stage import plot_stage_duration


def main():
    parser = argparse.ArgumentParser(description="Visualize stage duration statistics")
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Stage name to analyze (e.g., 'ETCH'). If not provided, analyzes all stages.",
    )
    parser.add_argument(
        "--product",
        type=str,
        default=None,
        help="Product ID to filter by. If not provided, includes all products.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./outputs/stage_dur_viz",
        help="Directory to save output files.",
    )

    args = parser.parse_args()

    q = "SELECT DISTINCT stage from hb_parent_drop_tail"
    stages = fetch_data(q)

    for s in stages["stage"]:
        logger.info("Starting stage duration analysis...")
        logger.info(f"  Stage: {s}")
        logger.info(f"  Product: {args.product or 'ALL'}")
        logger.info(f"  Output directory: {args.output_dir}")

        try:
            plot_stage_duration(stage_name=s, product_id=args.product, output_dir=args.output_dir)

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise


if __name__ == "__main__":
    logger.add("logs/stage_dur_viz.log", rotation="1000 MB")
    main()
