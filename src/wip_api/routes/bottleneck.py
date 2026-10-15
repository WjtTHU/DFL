from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from wip_analysis.bottleneck.bottle_neck_detection import (
    detect_bottleneck_by_daily_qty_variance,
    detect_bottleneck_by_hold_time,
    detect_bottleneck_by_throughput_var_and_load_rate,
)
from wip_api.models.bottleneck import (
    BottleneckDetectionRequest,
    BottleneckMethod,
    BottleneckResult,
    BottleneckRules,
)
from wip_api.utils.api_errors import raise_api_error

router = APIRouter()


@router.post("/bottleneck_detection", response_model=list[BottleneckResult])
async def run_bottleneck_detection(
    payload: BottleneckDetectionRequest,
) -> list[BottleneckResult]:
    """
    Run bottleneck stage detection using the specified method.

    Supported methods:
    - hold_time: detect stages where lots stay in hold state consecutively.
    - daily_qty_variance: detect stages with highest variance in daily processing quantity.
    - throughput_var_and_load_rate: detect stages based on daily throughput variance and load rate.
    """
    if payload.start_time >= payload.end_time:
        raise_api_error(
            status_code=400,
            reason="start_time must be before end_time",
            start_time=payload.start_time,
            end_time=payload.end_time,
            method=payload.method,
            product_ids=payload.product_ids,
        )

    rules: BottleneckRules = payload.rules or BottleneckRules()
    logger.info(
        "Starting bottleneck detection | method=%s | start=%s | end=%s | product_ids=%s | rules=%s",
        payload.method,
        payload.start_time,
        payload.end_time,
        payload.product_ids,
        rules.model_dump(exclude_none=True),
    )

    try:
        if payload.method == BottleneckMethod.HOLD_TIME:
            min_wafer_count = rules.min_wafer_count if rules.min_wafer_count is not None else 10
            min_hold_days = rules.min_hold_days if rules.min_hold_days is not None else 3
            results = await run_in_threadpool(
                detect_bottleneck_by_hold_time,
                start_time=payload.start_time,
                end_time=payload.end_time,
                product_ids=payload.product_ids,
                min_wafer_count=min_wafer_count,
                min_hold_days=min_hold_days,
            )
        elif payload.method == BottleneckMethod.DAILY_QTY_VARIANCE:
            processing_statuses = (
                rules.processing_statuses if rules.processing_statuses is not None else ["Active"]
            )
            top_k = rules.top_k if rules.top_k is not None else 5
            results = await run_in_threadpool(
                detect_bottleneck_by_daily_qty_variance,
                start_time=payload.start_time,
                end_time=payload.end_time,
                product_ids=payload.product_ids,
                processing_statuses=processing_statuses,
                top_k=top_k,
            )
        elif payload.method == BottleneckMethod.THROUGHPUT_VAR_AND_LOAD_RATE:
            top_k = rules.top_k if rules.top_k is not None else 5
            min_daily_throughput = (
                rules.min_daily_throughput if rules.min_daily_throughput is not None else 1
            )
            min_days = rules.min_days if rules.min_days is not None else 5
            min_utilization = rules.min_utilization if rules.min_utilization is not None else 0.5
            full_load_threshold = (
                rules.full_load_threshold if rules.full_load_threshold is not None else 0.85
            )
            weight_variance = rules.weight_variance if rules.weight_variance is not None else 0.5
            weight_utilization = (
                rules.weight_utilization if rules.weight_utilization is not None else 0.25
            )
            weight_full_load = (
                rules.weight_full_load if rules.weight_full_load is not None else 0.25
            )
            results = await run_in_threadpool(
                detect_bottleneck_by_throughput_var_and_load_rate,
                start_time=payload.start_time,
                end_time=payload.end_time,
                product_ids=payload.product_ids,
                top_k=top_k,
                min_daily_throughput=min_daily_throughput,
                min_days=min_days,
                min_utilization=min_utilization,
                full_load_threshold=full_load_threshold,
                weight_variance=weight_variance,
                weight_utilization=weight_utilization,
                weight_full_load=weight_full_load,
            )
        else:
            raise_api_error(
                status_code=400,
                reason="Unsupported detection method",
                method=payload.method,
            )

        if not results:
            raise_api_error(
                status_code=404,
                reason="No bottleneck results found for the given filters and method",
                start_time=payload.start_time,
                end_time=payload.end_time,
                method=payload.method,
                product_ids=payload.product_ids,
            )

        typed_results = [BottleneckResult(**result) for result in results]

        logger.info(
            "Bottleneck detection finished | method=%s | products_found=%d",
            payload.method,
            len(typed_results),
        )
        return typed_results

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error during bottleneck detection: {e}")
        raise_api_error(
            status_code=500,
            reason="Bottleneck detection failed",
            start_time=payload.start_time,
            end_time=payload.end_time,
            method=payload.method,
            product_ids=payload.product_ids,
        )
