from loguru import logger

from wip_analysis.hwm.lot_level import clear_lot_history_cache
from wip_analysis.hwm.product_level.aggregate_paths_stats import clear_paths_stats_caches
from wip_analysis.hwm.product_level.path_candidates import clear_pid_path_candidate_caches
from wip_analysis.hwm.product_level.primary_path import clear_pid_primary_path_cache
from wip_analysis.hwm.product_level.primary_path_dur import clear_pid_primary_durs_cache
from wip_analysis.hwm.stage.duration import clear_stage_duration_caches
from wip_analysis.utils.lot_id_check import clear_lot_id_legality_cache


def clear_refresh_lru_caches() -> None:
    """Clear in-process LRU caches that can affect forced refresh API results."""
    clearers = (
        ("lot_id_legality", clear_lot_id_legality_cache),
        ("lot_history", clear_lot_history_cache),
        ("primary_path", clear_pid_primary_path_cache),
        ("primary_path_durs", clear_pid_primary_durs_cache),
        ("path_candidates", clear_pid_path_candidate_caches),
        ("paths_stats", clear_paths_stats_caches),
        ("stage_duration", clear_stage_duration_caches),
    )
    for _name, clear in clearers:
        clear()

    logger.info(
        "Cleared refresh LRU caches: {}",
        ", ".join(name for name, _clear in clearers),
    )
