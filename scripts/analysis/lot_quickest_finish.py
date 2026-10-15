#!/usr/bin/env python
import argparse
import json

from loguru import logger

from wip_analysis.hwm.lot_level import LotQuickestFinishConfig, get_lot_quickest_finish


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Estimate the earliest possible finish time for a lot by combining stage-duration "
            "floors (min/quantile) with observed stage capacity (max lots/day), optionally "
            "adding a coarse queue-wait bound from current WIP."
        )
    )
    p.add_argument("lot_id", help="Target lot_id")
    p.add_argument(
        "--duration-quantile",
        type=float,
        default=0.0,
        help="Quantile in [0,1]",
    )
    p.add_argument(
        "--capacity-window",
        choices=["7d", "all"],
        default="7d",
        help="Window for taking the max observed lots/day capacity",
    )
    p.add_argument(
        "--no-queue",
        action="store_true",
        help="Disable queue-wait lower bound from current WIP/capacity",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = LotQuickestFinishConfig(
        duration_quantile=args.duration_quantile,
        capacity_window=args.capacity_window,
        include_queue_wait=not args.no_queue,
    )

    logger.info(
        "Estimating quickest finish: lot_id={}, q={}, capacity_window={}, queue_wait={}",
        args.lot_id,
        cfg.duration_quantile,
        cfg.capacity_window,
        cfg.include_queue_wait,
    )

    result = get_lot_quickest_finish(args.lot_id, config=cfg)
    print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    main()
