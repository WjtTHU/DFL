from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from wip_analysis.hwm.product_level import get_pid_stage_load
from wip_api.models.stage_load import StageLoadRatioRequest, StageLoadRatioResponse
from wip_api.utils.api_errors import raise_api_error

router = APIRouter()


@router.post("/products/{product_id:path}/load_ratio", response_model=StageLoadRatioResponse)
async def get_product_stage_load_ratio(
    product_id: str,
    payload: StageLoadRatioRequest | None = None,
) -> StageLoadRatioResponse:
    """
    Calculate rolling wafer-based stage load ratios for a product.

    This endpoint computes normalized load ratios for each stage in a product's
    primary path by:
    1. Using actual wafer sums derived from each lot's first snapshot occurrence
    2. Calculating rolling sums over multiple time windows (7d, 1m, 3m, 6m, 1y)
    3. Dividing by the average stage duration to get a normalized load metric
    4. Returning time-series data for each snapshot and stage, including raw lot counts
         and wafer sums aligned with the calendar axis

    The returned load values represent throughput capacity utilization in wafers/day,
    where higher values indicate more wafers passing through relative to the stage's
    typical duration.

    Args:
        product_id: The product ID to analyze (URL path parameter)
        payload: Optional request body with snapshot window filters.

    Returns:
        StageLoadRatioResponse containing:
        - product_id: The analyzed product
        - stage_order: Ordered list of stage names in primary path
        - load_ratios: Array of snapshot-specific load ratio data

    Raises:
        HTTPException: 500 if calculation fails
    """
    payload = payload or StageLoadRatioRequest()
    requested_product_id = payload.product_id
    snapshot_start = payload.snapshot_start
    snapshot_end = payload.snapshot_end

    if snapshot_start and snapshot_end and snapshot_start >= snapshot_end:
        raise_api_error(
            status_code=400,
            reason="snapshot_start must be before snapshot_end",
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
            product_id=product_id,
        )

    if requested_product_id and requested_product_id != product_id:
        raise_api_error(
            status_code=400,
            reason="product_id in request body does not match path parameter",
            path_product_id=product_id,
            requested_product_id=requested_product_id,
        )

    logger.info(
        "Fetching stage load ratios for product {}, snapshot_start={}, snapshot_end={}",
        product_id,
        snapshot_start,
        snapshot_end,
    )

    try:
        result = await run_in_threadpool(
            get_pid_stage_load,
            product_id,
            snapshot_start,
            snapshot_end,
        )

        if not result["load_ratios"]:
            logger.warning(f"No load ratio data found for product {product_id}")
            return StageLoadRatioResponse(
                **result,
                reason=("No stage load ratio data found for the given product and snapshot window"),
            )

        logger.info(
            f"Successfully calculated load ratios for {len(result['load_ratios'])} "
            f"snapshots and {len(result['stage_order'])} stages for product {product_id}"
        )

        return StageLoadRatioResponse(**result)

    except Exception as e:
        if isinstance(e, TimeoutError):
            raise_api_error(
                status_code=504,
                reason=str(e),
                product_id=product_id,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
            )
        if isinstance(e, ValueError):
            raise_api_error(
                status_code=409,
                reason=str(e),
                product_id=product_id,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
            )
        logger.exception(f"Error calculating load ratios for product {product_id}: {e}")
        raise_api_error(
            status_code=500,
            reason="Failed to calculate load ratios",
            product_id=product_id,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )


@router.post("/stage_load_ratio", response_model=StageLoadRatioResponse)
async def get_stage_load_ratio(
    payload: StageLoadRatioRequest | None = None,
) -> StageLoadRatioResponse:
    """
    Calculate rolling wafer-based stage load ratios with optional product filter.

    This is an alternative endpoint that accepts product_id in the request body
    instead of a path parameter. If no product_id is provided, analyzes all products.
    It also supports optional snapshot window filtering. Returned load values are
    based on actual wafer sums from the snapshot table.

    Args:
        payload: Optional request body containing product_id and snapshot window.

    Returns:
        StageLoadRatioResponse with load ratio calculations

    Raises:
        HTTPException: 500 if calculation fails
    """
    payload = payload or StageLoadRatioRequest()
    product_id = payload.product_id
    snapshot_start = payload.snapshot_start
    snapshot_end = payload.snapshot_end

    if snapshot_start and snapshot_end and snapshot_start >= snapshot_end:
        raise_api_error(
            status_code=400,
            reason="snapshot_start must be before snapshot_end",
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
            product_id=product_id,
        )

    logger.info(
        "Fetching stage load ratios for product_id={}, snapshot_start={}, snapshot_end={}",
        product_id,
        snapshot_start,
        snapshot_end,
    )

    try:
        result = await run_in_threadpool(
            get_pid_stage_load,
            product_id,
            snapshot_start,
            snapshot_end,
        )

        if not result["load_ratios"]:
            logger.warning(f"No load ratio data found for product_id={product_id}")
            return StageLoadRatioResponse(
                **result,
                reason="No stage load ratio data found for the given filters",
            )

        logger.info(
            f"Successfully calculated load ratios for {len(result['load_ratios'])} "
            f"snapshots for product_id={product_id or 'ALL'}"
        )

        return StageLoadRatioResponse(**result)

    except Exception as e:
        if isinstance(e, TimeoutError):
            raise_api_error(
                status_code=504,
                reason=str(e),
                product_id=product_id,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
            )
        if isinstance(e, ValueError):
            raise_api_error(
                status_code=409,
                reason=str(e),
                product_id=product_id,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
            )
        logger.exception(
            "Error calculating load ratios for product_id={}, snapshot_start={}, "
            "snapshot_end={}: {}",
            product_id,
            snapshot_start,
            snapshot_end,
            e,
        )
        raise_api_error(
            status_code=500,
            reason="Failed to calculate load ratios",
            product_id=product_id,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )
