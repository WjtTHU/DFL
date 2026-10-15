from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from loguru import logger

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql
from wip_analysis.hwm.product_level.aggregate_paths_stats import (
    get_paths_with_stats_for_products,
)
from wip_analysis.hwm.product_level.plot_paths import (
    PlotPathsNoDataError,
    build_paths_plot_html,
)
from wip_api.models.paths_stats import (
    ClusterInfo,
    PathsPlotRequest,
    PathsStatsRequest,
    PathsStatsResponse,
    ProductPathsStats,
    TimeStats,
)
from wip_api.utils.api_errors import raise_api_error
from wip_api.utils.refresh_cache import clear_refresh_lru_caches

router = APIRouter()


def _query_snapshot_times_in_range(
    snapshot_start: datetime,
    snapshot_end: datetime,
) -> list[datetime]:
    """Return distinct snapshot times in [snapshot_start, snapshot_end]."""
    query = render_sql(
        "get_scope_snapshot_times.sql.j2",
        {
            "product_id": None,
            "lot_id": None,
            "snapshot_start": snapshot_start,
            "snapshot_end": snapshot_end,
        },
    )
    snapshot_df = fetch_data(query)
    if snapshot_df.is_empty():
        return []
    return [row["snapshot_time"] for row in snapshot_df.iter_rows(named=True)]


def _get_paths_with_stats_for_snapshot_range(
    snapshot_start: datetime,
    snapshot_end: datetime,
    product_ids: list[str] | None = None,
    rules: dict[str, Any] | None = None,
) -> list[tuple[datetime, dict[str, list[dict[str, Any]]]]]:
    """Compute path stats for each snapshot cut in the requested window."""
    snapshot_times = _query_snapshot_times_in_range(
        snapshot_start=snapshot_start,
        snapshot_end=snapshot_end,
    )
    if not snapshot_times:
        return []

    cuts: list[tuple[datetime, dict[str, list[dict[str, Any]]]]] = []
    for snapshot_time in snapshot_times:
        logger.debug(
            "Computing paths with stats for snapshot_end=%s (snapshot_start=None)",
            snapshot_time,
        )
        cut_result = get_paths_with_stats_for_products(
            snapshot_start=None,
            snapshot_end=snapshot_time,
            product_ids=product_ids,
            rules=rules,
        )
        cuts.append((snapshot_time, cut_result))
    return cuts


def _product_paths_stats_from_result(
    snapshot_time: datetime,
    result: dict[str, list[dict[str, Any]]],
) -> list[ProductPathsStats]:
    products: list[ProductPathsStats] = []
    for product_id, clusters_data in result.items():
        clusters: list[ClusterInfo] = []
        for cluster_data in clusters_data:
            time_stats_dict = cluster_data["time_stats"]
            time_stats = TimeStats(
                earliest_start=time_stats_dict.get("earliest_start"),
                latest_start=time_stats_dict.get("latest_start"),
                mean_start=time_stats_dict.get("mean_start"),
                median_start=time_stats_dict.get("median_start"),
            )
            clusters.append(
                ClusterInfo(
                    cluster_id=cluster_data["cluster_id"],
                    lot_count=cluster_data["lot_count"],
                    wafer_qty=cluster_data["wafer_qty"],
                    path=cluster_data["path"],
                    time_stats=time_stats,
                    type=cluster_data.get("type", "Secondary"),
                )
            )
        products.append(
            ProductPathsStats(
                product_id=product_id,
                snapshot_end=snapshot_time,
                clusters=clusters,
            )
        )
    return products


def _paths_stats_result_to_response(
    cuts: list[tuple[datetime, dict[str, list[dict[str, Any]]]]],
) -> PathsStatsResponse:
    products: list[ProductPathsStats] = []
    for snapshot_time, cut_result in cuts:
        products.extend(_product_paths_stats_from_result(snapshot_time, cut_result))

    response_reason: str | None = None
    if not products:
        response_reason = "No paths statistics found for the given snapshot window and filters"

    return PathsStatsResponse(products=products, reason=response_reason)


@router.post("/paths_stats", response_model=PathsStatsResponse)
async def get_paths_with_stats(payload: PathsStatsRequest) -> PathsStatsResponse:
    """
    Get primary/secondary paths (by clusters) with per-cluster stats for products.

    This endpoint:
    - Builds recipe sequences from snapshots
    - Clusters lot recipe sequences per product
    - Solves a primary path for each cluster

    Notes:
        - Cluster id 0 is treated as outliers and excluded
        - The "primary" route for a product is typically the cluster with the largest lot_count
        - Other clusters can be considered "secondary" routes
        - Returns one product entry per snapshot cut in [snapshot_start, snapshot_end]
    """
    if payload.snapshot_start >= payload.snapshot_end:
        raise_api_error(
            status_code=400,
            reason="snapshot_start must be before snapshot_end",
            snapshot_start=payload.snapshot_start,
            snapshot_end=payload.snapshot_end,
            product_ids=payload.product_ids,
        )

    logger.info(
        "Fetching paths with stats | start=%s | end=%s | product_ids=%s | rules=%s",
        payload.snapshot_start,
        payload.snapshot_end,
        payload.product_ids,
        payload.rules,
    )

    try:
        cuts = await run_in_threadpool(
            _get_paths_with_stats_for_snapshot_range,
            payload.snapshot_start,
            payload.snapshot_end,
            payload.product_ids,
            payload.rules,
        )

        response = _paths_stats_result_to_response(cuts)
        logger.info(
            "Successfully retrieved paths with stats for %d product entries",
            len(response.products),
        )
        return response

    except ValueError as e:
        raise_api_error(
            status_code=400,
            reason=str(e),
            snapshot_start=payload.snapshot_start,
            snapshot_end=payload.snapshot_end,
            product_ids=payload.product_ids,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error fetching paths with stats: {e}")
        raise_api_error(
            status_code=500,
            reason="Failed to fetch paths with stats",
            snapshot_start=payload.snapshot_start,
            snapshot_end=payload.snapshot_end,
            product_ids=payload.product_ids,
        )


async def _paths_plot_html_response(
    product_id: str,
    payload: PathsPlotRequest,
) -> HTMLResponse:
    if payload.snapshot_start >= payload.snapshot_end:
        raise_api_error(
            status_code=400,
            reason="snapshot_start must be before snapshot_end",
            product_id=product_id,
            snapshot_start=payload.snapshot_start,
            snapshot_end=payload.snapshot_end,
        )

    logger.info(
        "Building paths plot for {} | start=%s | end=%s | align=%s | refresh=%s | rules=%s",
        product_id,
        payload.snapshot_start,
        payload.snapshot_end,
        payload.align,
        payload.refresh,
        payload.rules,
    )

    try:
        if payload.refresh:
            clear_refresh_lru_caches()

        html = await run_in_threadpool(
            build_paths_plot_html,
            product_id,
            payload.snapshot_start,
            payload.snapshot_end,
            align=payload.align,
            rules=payload.rules,
        )
        return HTMLResponse(html)
    except HTTPException:
        raise
    except ValueError as e:
        raise_api_error(
            status_code=400,
            reason=str(e),
            product_id=product_id,
            snapshot_start=payload.snapshot_start,
            snapshot_end=payload.snapshot_end,
        )
    except PlotPathsNoDataError as e:
        raise_api_error(
            status_code=404,
            reason=str(e),
            product_id=product_id,
            snapshot_start=payload.snapshot_start,
            snapshot_end=payload.snapshot_end,
        )
    except Exception as e:
        logger.exception(
            "Error building paths plot for {} between {} and {}: {}",
            product_id,
            payload.snapshot_start,
            payload.snapshot_end,
            e,
        )
        raise_api_error(
            status_code=500,
            reason="Failed to build paths plot",
            product_id=product_id,
            snapshot_start=payload.snapshot_start,
            snapshot_end=payload.snapshot_end,
        )


@router.post("/products/{product_id:path}/paths.html", response_class=HTMLResponse)
async def get_paths_plot_html(
    product_id: str,
    request: PathsPlotRequest,
) -> HTMLResponse:
    """
    Build an interactive network graph of primary/secondary paths for a product.

    Uses the same clustering and path statistics pipeline as POST /paths_stats.
    """
    return await _paths_plot_html_response(product_id, request)
