from dataclasses import dataclass
from datetime import date, datetime

import polars as pl
from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql
from wip_analysis.hwm.stage.utils.lot_progress_readiness import (
    ensure_date_out_pred_populated,
    ensure_lot_progress_snapshot_cuts_populated,
)
from wip_api.models.pid_date_out import (
    ProductDateOutCount,
    ProductDateOutDayOutput,
    ProductDateOutRequest,
    ProductDateOutResponse,
    ProductDateOutSplit,
    ProductDateOutStageOutput,
)
from wip_api.utils.api_errors import raise_api_error

router = APIRouter()


@dataclass(frozen=True)
class ProductDateOutCalculation:
    lot_counts: list[int]
    wafer_counts: list[int]
    actual_lot_counts: list[int]
    actual_wafer_counts: list[int]
    predicted_lot_counts: list[int]
    predicted_wafer_counts: list[int]
    date_outputs: list[ProductDateOutDayOutput]


def _query_finish_lots(product_id: str, target_day: date) -> pl.DataFrame:
    query = render_sql(
        "get_pid_finish_lots_on_date.sql.j2",
        {"product_id": product_id, "target_date": target_day},
    )
    return fetch_data(query)


def _query_date_out_lots(product_id: str, target_days: list[date]) -> pl.DataFrame:
    query = render_sql(
        "get_pid_date_out_lots.sql.j2",
        {"product_id": product_id, "target_days": target_days},
    )
    return fetch_data(query)


def _query_date_out_outputs(product_id: str, target_days: list[date]) -> pl.DataFrame:
    query = render_sql(
        "get_pid_date_out_outputs.sql.j2",
        {"product_id": product_id, "target_days": target_days},
    )
    return fetch_data(query)


def _query_latest_wafers(product_id: str, target_day: date | None = None) -> pl.DataFrame:
    query = render_sql(
        "get_pid_latest_wafers.sql.j2",
        {"product_id": product_id, "target_date": target_day},
    )
    return fetch_data(query)


def _count_pid_date_out(finish_df: pl.DataFrame, wafers_df: pl.DataFrame) -> tuple[int, int]:
    if finish_df.is_empty():
        return 0, 0

    joined = finish_df.join(wafers_df, on="lot_id", how="left").with_columns(
        pl.col("current_wafer_qty").fill_null(0).cast(pl.Int64)
    )

    total_wafers = int(joined.select(pl.col("current_wafer_qty").sum()).item())
    lot_count = int(joined.height)
    return lot_count, total_wafers


def _date_from_value(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"Expected date value, got {type(value).__name__}")


def _count_from_row(row: dict[str, object]) -> ProductDateOutCount:
    return ProductDateOutCount(
        lot_count=int(row.get("lot_count") or 0),
        wafer_count=int(row.get("wafer_count") or 0),
    )


def _split_total(split: ProductDateOutSplit) -> ProductDateOutCount:
    return ProductDateOutCount(
        lot_count=split.actual.lot_count + split.predicted.lot_count,
        wafer_count=split.actual.wafer_count + split.predicted.wafer_count,
    )


def _finish_lot_snapshot_cuts(finish_df: pl.DataFrame) -> list[tuple[str, datetime]]:
    if finish_df.is_empty() or "snapshot_cut" not in finish_df.columns:
        return []

    snapshot_cuts: dict[tuple[str, datetime], tuple[str, datetime]] = {}
    for row in finish_df.iter_rows(named=True):
        lot_id = str(row["lot_id"])
        snapshot_cut = row["snapshot_cut"]
        if not isinstance(snapshot_cut, datetime):
            continue
        snapshot_cuts[(lot_id, snapshot_cut)] = (lot_id, snapshot_cut)
    return list(snapshot_cuts.values())


def _ensure_finish_lot_progress_ready(
    product_id: str,
    snapshot_cuts: list[tuple[str, datetime]],
) -> None:
    ensure_lot_progress_snapshot_cuts_populated(
        product_id=product_id,
        snapshot_cuts=[
            {"lot_id": lot_id, "snapshot_cut": snapshot_cut}
            for lot_id, snapshot_cut in snapshot_cuts
        ],
        context="product date-out finish lots",
        max_wait_seconds=None,
        poll_interval_seconds=1.0,
    )


def _build_date_out_calculation(
    target_days: list[date],
    output_df: pl.DataFrame,
) -> ProductDateOutCalculation:
    day_outputs_by_date: dict[date, ProductDateOutDayOutput] = {
        target_day: ProductDateOutDayOutput(target_date=target_day) for target_day in target_days
    }
    stage_outputs_by_date: dict[date, dict[str, dict[str, ProductDateOutCount]]] = {
        target_day: {} for target_day in target_days
    }

    if not output_df.is_empty():
        for row in output_df.iter_rows(named=True):
            target_day = _date_from_value(row["target_day"])
            day_output = day_outputs_by_date.get(target_day)
            if day_output is None:
                continue

            output_type = str(row["output_type"])
            count = _count_from_row(row)
            if output_type not in {"actual", "predicted"}:
                continue

            if row["output_scope"] == "final":
                setattr(day_output.final, output_type, count)
                continue

            stage_name = row.get("stage_name")
            if not stage_name:
                continue

            stage_map = stage_outputs_by_date[target_day].setdefault(str(stage_name), {})
            stage_map[output_type] = count

    date_outputs: list[ProductDateOutDayOutput] = []
    for target_day in target_days:
        day_output = day_outputs_by_date[target_day]
        stage_outputs: list[ProductDateOutStageOutput] = []
        for stage_name, split_counts in stage_outputs_by_date[target_day].items():
            stage_outputs.append(
                ProductDateOutStageOutput(
                    stage_name=stage_name,
                    actual=split_counts.get("actual", ProductDateOutCount()),
                    predicted=split_counts.get("predicted", ProductDateOutCount()),
                )
            )
        day_output.stages = stage_outputs
        date_outputs.append(day_output)

    actual_lot_counts = [day_output.final.actual.lot_count for day_output in date_outputs]
    actual_wafer_counts = [day_output.final.actual.wafer_count for day_output in date_outputs]
    predicted_lot_counts = [day_output.final.predicted.lot_count for day_output in date_outputs]
    predicted_wafer_counts = [day_output.final.predicted.wafer_count for day_output in date_outputs]
    totals = [_split_total(day_output.final) for day_output in date_outputs]

    return ProductDateOutCalculation(
        lot_counts=[total.lot_count for total in totals],
        wafer_counts=[total.wafer_count for total in totals],
        actual_lot_counts=actual_lot_counts,
        actual_wafer_counts=actual_wafer_counts,
        predicted_lot_counts=predicted_lot_counts,
        predicted_wafer_counts=predicted_wafer_counts,
        date_outputs=date_outputs,
    )


def _calculate_pid_date_out(product_id: str, target_days: list[date]) -> ProductDateOutCalculation:
    ensure_date_out_pred_populated(
        product_id=product_id,
        target_days=target_days,
        context="product date-out",
        max_wait_seconds=None,
        poll_interval_seconds=1.0,
    )

    finish_lots_df = _query_date_out_lots(product_id, target_days)
    if finish_lots_df.is_empty():
        wafers_dfs = [_query_latest_wafers(product_id, target_day) for target_day in target_days]
        if all(wafers_df.is_empty() for wafers_df in wafers_dfs):
            raise ValueError(f"No snapshot wafer data found for pid={product_id}")
        return _build_date_out_calculation(target_days, pl.DataFrame())

    _ensure_finish_lot_progress_ready(product_id, _finish_lot_snapshot_cuts(finish_lots_df))

    output_df = _query_date_out_outputs(product_id, target_days)
    return _build_date_out_calculation(target_days, output_df)


@router.post("/products/{product_id:path}/date_out", response_model=ProductDateOutResponse)
async def get_product_date_out(
    product_id: str,
    payload: ProductDateOutRequest,
) -> ProductDateOutResponse:
    """
    Get expected lot and wafer output counts for a product on target dates.

    Args:
        product_id: Product identifier to query.
        payload: Request body containing target_dates.

    Returns:
        ProductDateOutResponse with lot and wafer count lists.

    Raises:
        HTTPException: 404 if no wafer data is available, 500 for query failures.
    """
    target_dates = payload.target_dates
    logger.info(f"Fetching date-out counts for product {product_id} on {len(target_dates)} dates")

    try:
        calculation = await run_in_threadpool(_calculate_pid_date_out, product_id, target_dates)

        empty_reason: str | None = None
        if all(lot_count == 0 for lot_count in calculation.lot_counts):
            empty_reason = "No finished lots found for the requested target_dates"

        for target_date, lot_count, wafer_count in zip(
            target_dates,
            calculation.lot_counts,
            calculation.wafer_counts,
            strict=True,
        ):
            if lot_count == 0:
                logger.warning(f"No lots found for pid={product_id} with target date={target_date}")

            logger.info(
                f"pid={product_id} date={target_date.isoformat()} lots={lot_count} wafers={wafer_count}"
            )

        return ProductDateOutResponse(
            product_id=product_id,
            target_dates=target_dates,
            lot_counts=calculation.lot_counts,
            wafer_counts=calculation.wafer_counts,
            actual_lot_counts=calculation.actual_lot_counts,
            actual_wafer_counts=calculation.actual_wafer_counts,
            predicted_lot_counts=calculation.predicted_lot_counts,
            predicted_wafer_counts=calculation.predicted_wafer_counts,
            date_outputs=calculation.date_outputs,
            reason=empty_reason,
        )
    except ValueError as e:
        logger.warning(str(e))
        status_code = 409 if "lot_progress_table data is not ready yet" in str(e) else 404
        raise_api_error(
            status_code=status_code,
            reason=str(e),
            product_id=product_id,
            target_dates=[target_date.isoformat() for target_date in target_dates],
        )
    except TimeoutError as e:
        logger.warning(str(e))
        raise_api_error(
            status_code=504,
            reason=str(e),
            product_id=product_id,
            target_dates=[target_date.isoformat() for target_date in target_dates],
        )
    except Exception as e:
        logger.exception(
            f"Error fetching date-out counts for product {product_id} on {target_dates}: {e}"
        )
        raise_api_error(
            status_code=500,
            reason="Failed to fetch date-out counts",
            product_id=product_id,
            target_dates=[target_date.isoformat() for target_date in target_dates],
        )
