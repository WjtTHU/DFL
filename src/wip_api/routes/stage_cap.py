from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from wip_analysis.hwm.stage import get_stage_capacity
from wip_api.models.stage_cap import (
    StageCapacityRequest,
    StageCapacityResponse,
    StageCapacityRow,
    StageCapacitySeries,
)
from wip_api.utils.api_errors import raise_api_error

router = APIRouter()


@router.post("/stage_cap", response_model=StageCapacityResponse)
async def get_stage_capacity_data(
    payload: StageCapacityRequest | None = None,
) -> StageCapacityResponse:
    """
    Get stage capacity statistics showing lot and wafer volume entering stages over time.

    This endpoint retrieves manufacturing stage capacity data, which includes:
    - The number of lots entering each stage per day
    - The summed wafer quantity for those lots at stage entry time
    - Optional filtering by a stage name, ordered stage-name list, and/or product ID
    - Per-stage response groups that follow the input stage order

    Args:
        payload: Optional request body with stage/product filters and snapshot window.

    Returns:
        StageCapacityResponse containing lot counts and actual wafer sums.

    Raises:
        HTTPException: 400 for invalid snapshot window, 500 for retrieval failures.
    """
    payload = payload or StageCapacityRequest()
    stage_name = payload.stage_name
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
            stage_name=stage_name,
        )

    if isinstance(stage_name, list) and not stage_name:
        logger.info("Received empty stage_name list for stage capacity query; using all stages")
        stage_name = None

    query_stage_name = stage_name
    if isinstance(stage_name, list):
        query_stage_name = list(dict.fromkeys(stage_name))

    logger.info(
        "Fetching stage capacity data for stage_name=%s, product_id=%s, "
        "snapshot_start=%s, snapshot_end=%s",
        stage_name,
        product_id,
        snapshot_start,
        snapshot_end,
    )

    try:
        capacity_df = await run_in_threadpool(
            get_stage_capacity,
            stage_name=query_stage_name,
            product_id=product_id,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
            raise_on_error=True,
        )

        if capacity_df.is_empty():
            logger.warning(
                "No stage capacity data found for stage_name=%s, product_id=%s, "
                "snapshot_start=%s, snapshot_end=%s",
                stage_name,
                product_id,
                snapshot_start,
                snapshot_end,
            )
            return StageCapacityResponse(
                product_id=product_id,
                stage_name=stage_name,
                capacity_data=[],
                reason="No stage capacity data found for the given filters",
            )

        rows_by_stage: dict[str, list[StageCapacityRow]] = {}
        for row in capacity_df.iter_rows(named=True):
            stage = row["stage_name"]
            capacity_row = StageCapacityRow(
                calendar_date=row["calendar_date"],
                stage_name=stage,
                entering_lot_count=row["entering_lot_count"],
                entering_wafer_count=row["entering_wafer_count"],
            )
            rows_by_stage.setdefault(stage, []).append(capacity_row)

        if stage_name is None:
            ordered_stage_names = sorted(rows_by_stage)
        elif isinstance(stage_name, str):
            ordered_stage_names = [stage_name]
        else:
            ordered_stage_names = stage_name

        capacity_data = [
            StageCapacitySeries(stage_name=stage, wafer_data=rows_by_stage.get(stage, []))
            for stage in ordered_stage_names
        ]

        total_wafer_rows = sum(len(series.wafer_data) for series in capacity_data)
        logger.info(
            f"Successfully retrieved {len(capacity_data)} stage series and "
            f"{total_wafer_rows} wafer-capacity rows"
        )

        return StageCapacityResponse(
            product_id=product_id, stage_name=stage_name, capacity_data=capacity_data
        )

    except ValueError as e:
        raise_api_error(
            status_code=409,
            reason=str(e),
            product_id=product_id,
            stage_name=stage_name,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )
    except TimeoutError as e:
        raise_api_error(
            status_code=504,
            reason=str(e),
            product_id=product_id,
            stage_name=stage_name,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )
    except Exception as e:
        logger.error(
            f"Error fetching stage capacity data for stage_name={stage_name}, "
            f"product_id={product_id}: {e}"
        )
        raise_api_error(
            status_code=500,
            reason="Failed to fetch stage capacity data",
            product_id=product_id,
            stage_name=stage_name,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )
