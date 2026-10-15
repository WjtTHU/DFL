from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from wip_analysis.hwm.product_level import (
    # get_pid_primary_durs,
    get_pid_primary_path,
    # get_pid_quickest_cycle,
    get_pid_stage_stats,
)
from wip_api.models.hwm_prod import (
    ProdStageStatsRequest,
    ProdStageStatsResponse,
    ProdStageStatsRow,
)
from wip_api.utils.api_errors import raise_api_error

router = APIRouter()


@router.post("/products/{product_id:path}", response_model=ProdStageStatsResponse)
async def get_prod_hwm(
    product_id: str,
    payload: ProdStageStatsRequest | None = None,
) -> ProdStageStatsResponse:
    """
    Get product-level HWM stage lot-count statistics across snapshots.

    Args:
        product_id: Product identifier to analyze.

    Returns:
        ProdStageStatsResponse with stage order and per-snapshot lot counts.

    Raises:
        HTTPException: 500 if data retrieval fails.
    """
    payload = payload or ProdStageStatsRequest()
    requested_product_id = payload.product_id
    snapshot_start = payload.snapshot_start
    snapshot_end = payload.snapshot_end

    if snapshot_start and snapshot_end and snapshot_start >= snapshot_end:
        raise_api_error(
            status_code=400,
            reason="snapshot_start must be before snapshot_end",
            product_id=product_id,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )

    if requested_product_id and requested_product_id != product_id:
        raise_api_error(
            status_code=400,
            reason="product_id in request body does not match path parameter",
            path_product_id=product_id,
            requested_product_id=requested_product_id,
        )

    logger.info(
        "Fetching product HWM data for {}, snapshot_start={}, snapshot_end={}",
        product_id,
        snapshot_start,
        snapshot_end,
    )

    try:
        stage_stats_data = await run_in_threadpool(
            get_pid_stage_stats,
            product_id,
            snapshot_start,
            snapshot_end,
            True,
        )
        stage_order = await run_in_threadpool(
            get_pid_primary_path,
            product_id,
            path_start_time=snapshot_start,
            path_end_time=snapshot_end,
        )
        # _ = await run_in_threadpool(get_pid_primary_durs, product_id)
        # _ = await run_in_threadpool(get_pid_quickest_cycle, product_id)

        if not stage_stats_data:
            logger.warning(f"No stage statistics found for product {product_id}")
            return ProdStageStatsResponse(
                product_id=product_id,
                stage_order=[],
                wip_stats=[],
                reason="No stage statistics found for the given product, should POST /api/v1/lots/{lot_id} for all the lots under this product first to populate the data",
            )

        # Convert the data to ProdStageStatsRow objects
        wip_stats: list[ProdStageStatsRow] = []
        for data_row in stage_stats_data:
            stats_row = ProdStageStatsRow(
                calendar_time=data_row["calendar_time"],
                snapshot_cut=data_row["snapshot_cut"],
                stage_lot_cnt=data_row["stage_lot_cnt"],
            )
            wip_stats.append(stats_row)

        logger.info(f"Successfully retrieved {len(wip_stats)} HWM records for product {product_id}")

        return ProdStageStatsResponse(
            product_id=product_id, stage_order=stage_order, wip_stats=wip_stats
        )

    except ValueError as e:
        raise_api_error(
            status_code=409,
            reason=str(e),
            product_id=product_id,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )
    except TimeoutError as e:
        raise_api_error(
            status_code=504,
            reason=str(e),
            product_id=product_id,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )
    except Exception as e:
        logger.error(f"Error fetching HWM data for product {product_id}: {e}")
        raise_api_error(
            status_code=500,
            reason="Failed to fetch HWM data",
            product_id=product_id,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )
