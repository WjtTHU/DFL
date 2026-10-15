from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql
from wip_analysis.hwm.stage import get_stage_duration
from wip_api.models.stage_dur import (
    StageDurationRequest,
    StageDurationResponse,
    StageDurationRow,
)
from wip_api.utils.api_errors import raise_api_error

router = APIRouter()


def _query_initial_wafers(product_id: str | None, lot_ids: list[str]) -> dict[tuple[str, str], int]:
    if not lot_ids:
        return {}

    query = render_sql(
        "get_lot_initial_wafers.sql.j2",
        {
            "product_id": product_id,
            "lot_ids": lot_ids,
        },
    )
    wafer_df = fetch_data(query)
    if wafer_df.is_empty():
        return {}

    wafer_map: dict[tuple[str, str], int] = {}
    for row in wafer_df.iter_rows(named=True):
        wafer_count = row.get("wafer_count")
        if wafer_count is None:
            continue
        wafer_map[(row["product_id"], row["lot_id"])] = int(wafer_count)

    return wafer_map


@router.post("/stage_dur", response_model=StageDurationResponse)
async def get_stage_duration_data(
    payload: StageDurationRequest | None = None,
) -> StageDurationResponse:
    """
    Get stage duration (cycle time) statistics showing how long lots spent in each stage.

    This endpoint retrieves manufacturing stage duration data, which includes:
    - Entry and exit timestamps for each lot in each stage
    - Duration (cycle time) spent in each stage
    - Initial wafer quantity for each lot from the snapshot table
    - Optional filtering by stage name, product ID, and start-time window

    Args:
        payload: Optional request body with stage, product, and time-window filters.

    Returns:
        StageDurationResponse containing the duration statistics

    Raises:
        HTTPException: 500 if duration retrieval fails.
    """
    payload = payload or StageDurationRequest()
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

    logger.info(
        "Fetching stage duration data for stage_name={}, product_id={}, "
        "snapshot_start={}, snapshot_end={}",
        stage_name,
        product_id,
        snapshot_start,
        snapshot_end,
    )

    try:
        duration_df = await run_in_threadpool(
            get_stage_duration,
            stage_name=stage_name,
            product_id=product_id,
            lot_id=None,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )

        if duration_df.is_empty():
            logger.warning(
                "No stage duration data found for stage_name={}, product_id={}, "
                "snapshot_start={}, snapshot_end={}",
                stage_name,
                product_id,
                snapshot_start,
                snapshot_end,
            )
            return StageDurationResponse(
                product_id=product_id,
                stage_name=stage_name,
                duration_data=[],
                reason="No stage duration data found for the given filters",
            )

        wafer_map = await run_in_threadpool(
            _query_initial_wafers,
            product_id,
            duration_df.get_column("lot_id").unique().to_list(),
        )

        # Convert DataFrame to list of StageDurationRow objects
        duration_data: list[StageDurationRow] = []
        for row in duration_df.iter_rows(named=True):
            wafer_count = wafer_map.get((row["product_id"], row["lot_id"]))
            duration_row = StageDurationRow(
                product_id=row["product_id"],
                lot_id=row["lot_id"],
                stage_name=row["stage_name"],
                entry_date=row.get("entry_date") or row["start_time"].date(),
                start_time=row["start_time"],
                end_time=row["end_time"],
                wafer_count=wafer_count,
                duration_seconds=row["duration_seconds"],
            )
            duration_data.append(duration_row)

        logger.info(f"Successfully retrieved {len(duration_data)} stage duration records")

        return StageDurationResponse(
            product_id=product_id, stage_name=stage_name, duration_data=duration_data
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
        logger.exception(
            "Error fetching stage duration data for stage_name={}, product_id={}, "
            "snapshot_start={}, snapshot_end={}: {}",
            stage_name,
            product_id,
            snapshot_start,
            snapshot_end,
            e,
        )
        raise_api_error(
            status_code=500,
            reason="Failed to fetch stage duration data",
            product_id=product_id,
            stage_name=stage_name,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )
