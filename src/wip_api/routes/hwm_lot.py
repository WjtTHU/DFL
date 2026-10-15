import html
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from loguru import logger

from wip_analysis.hwm.lot_level import get_lot_prog_over_snapshots
from wip_analysis.hwm.lot_level.stage_dur_z import (
    LotProgressZNoDataError,
    LotProgressZPlot,
    build_lot_progress_z_plot,
)
from wip_analysis.hwm.product_level import get_pid_primary_path
from wip_analysis.utils.lot_id_check import check_lot_id_legality
from wip_api.models.hwm_lot import (
    LotProgressNote,
    LotProgressRequest,
    LotProgressResponse,
    LotProgressRow,
    StageDetail,
    StageDetailComp,
)
from wip_api.utils.api_errors import raise_api_error
from wip_api.utils.refresh_cache import clear_refresh_lru_caches

router = APIRouter()


def _lot_progress_z_warning_banner(warnings: tuple[str, ...]) -> str:
    if not warnings:
        return ""

    items = "".join(f"<li>{html.escape(warning)}</li>" for warning in warnings)
    return (
        '<div class="plot-warning" '
        'style="box-sizing:border-box;margin:12px 12px 0;padding:10px 12px;'
        "border:1px solid #d97706;background:#fffbeb;color:#78350f;"
        "font:14px/1.4 system-ui,-apple-system,BlinkMacSystemFont,"
        "'Segoe UI',sans-serif;\">"
        "<strong>Warning</strong>"
        f'<ul style="margin:6px 0 0 18px;padding:0;">{items}</ul>'
        "</div>"
    )


def _lot_progress_z_html(plot: LotProgressZPlot) -> str:
    html_doc = plot.fig.to_html(
        full_html=True,
        include_plotlyjs=True,
        config={"responsive": True},
        default_width="100%",
        default_height="900px",
    )
    banner = _lot_progress_z_warning_banner(plot.warnings)
    if not banner:
        return html_doc

    if "<body>" in html_doc:
        return html_doc.replace("<body>", f"<body>{banner}", 1)
    return banner + html_doc


def _build_comparisons(sd: dict[str, Any]) -> StageDetailComp:
    windows = ["1d", "7d", "1m", "1q", "6m", "1y"]
    comp: dict[str, Optional[float]] = {}
    for w in windows:
        comp[f"dev_vs_{w}"] = sd.get(f"dev_vs_{w}_h")
        comp[f"avg_cum_{w}"] = sd.get(f"avg_cum_{w}_h")
    return StageDetailComp(**comp)


async def _lot_progress_z_html_response(
    lot_id: str,
    refresh: bool,
    snapshot_start: datetime | None,
    snapshot_end: datetime | None,
) -> HTMLResponse:
    if snapshot_start and snapshot_end and snapshot_start >= snapshot_end:
        raise_api_error(
            status_code=400,
            reason="snapshot_start must be before snapshot_end",
            lot_id=lot_id,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )

    logger.info(
        "Fetching lot progress z-score plot for {}, refresh={}, snapshot_start={}, snapshot_end={}",
        lot_id,
        refresh,
        snapshot_start,
        snapshot_end,
    )

    try:
        legality = check_lot_id_legality(lot_id)
        if not legality.exists:
            raise_api_error(status_code=404, reason=legality.detail, lot_id=lot_id)
        if not legality.is_legal:
            raise_api_error(status_code=400, reason=legality.detail, lot_id=lot_id)

        plot = await run_in_threadpool(
            build_lot_progress_z_plot,
            lot_id,
            refresh=refresh,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )

        for warning in plot.warnings:
            logger.warning("Lot progress z-score plot warning for {}: {}", lot_id, warning)

        return HTMLResponse(_lot_progress_z_html(plot))
    except HTTPException:
        raise
    except LotProgressZNoDataError as e:
        raise_api_error(
            status_code=404,
            reason=str(e),
            lot_id=lot_id,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )
    except Exception as e:
        logger.exception("Error building lot progress z-score plot for {}: {}", lot_id, e)
        raise_api_error(
            status_code=500,
            reason="Failed to build lot progress z-score plot",
            lot_id=lot_id,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )


@router.post("/lots/{lot_id:path}/progress-z.html", response_class=HTMLResponse)
async def get_lot_progress_z_html(
    lot_id: str,
    refresh: bool = False,
    snapshot_start: datetime | None = None,
    snapshot_end: datetime | None = None,
) -> HTMLResponse:
    return await _lot_progress_z_html_response(
        lot_id=lot_id,
        refresh=refresh,
        snapshot_start=snapshot_start,
        snapshot_end=snapshot_end,
    )


@router.post("/lots/{lot_id:path}", response_model=LotProgressResponse)
async def get_lot_hwm(
    lot_id: str,
    payload: LotProgressRequest | None = None,
) -> LotProgressResponse:
    """
    Get lot-level historical WIP metrics across snapshots.

    Args:
        lot_id: Lot identifier to analyze.
        payload: Optional request body with refresh/snapshot filters and product guard.

    Returns:
        LotProgressResponse with snapshot rows and stage order.

    Raises:
        HTTPException: 400 for illegal lot_id or mismatched product_id, 404 for missing lot/data.
    """
    payload = payload or LotProgressRequest()
    requested_product_id = payload.product_id
    refresh = payload.refresh
    snapshot_start = payload.snapshot_start
    snapshot_end = payload.snapshot_end

    if snapshot_start and snapshot_end and snapshot_start >= snapshot_end:
        raise_api_error(
            status_code=400,
            reason="snapshot_start must be before snapshot_end",
            lot_id=lot_id,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )

    logger.info(
        "Fetching lot HWM data for {}, product_id={}, force refresh={}, snapshot_start={}, snapshot_end={}",
        lot_id,
        requested_product_id,
        refresh,
        snapshot_start,
        snapshot_end,
    )
    try:
        if refresh:
            clear_refresh_lru_caches()

        legality = check_lot_id_legality(lot_id)
        if not legality.exists:
            raise_api_error(status_code=404, reason=legality.detail, lot_id=lot_id)
        if not legality.is_legal:
            raise_api_error(status_code=400, reason=legality.detail, lot_id=lot_id)

        if (
            requested_product_id
            and legality.unique_product_id
            and requested_product_id != legality.unique_product_id
        ):
            raise_api_error(
                status_code=400,
                reason="lot_id and product_id do not match",
                lot_id=lot_id,
                requested_product_id=requested_product_id,
                actual_product_id=legality.unique_product_id,
            )

        lot_full_data = await run_in_threadpool(
            get_lot_prog_over_snapshots,
            lot_id,
            refresh,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )
        if not lot_full_data:
            raise_api_error(
                status_code=404,
                reason="Lot not found or no data available",
                lot_id=lot_id,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
            )

        actual_product_id = lot_full_data[0]["product_id"]
        if requested_product_id and requested_product_id != actual_product_id:
            raise_api_error(
                status_code=400,
                reason="lot_id and product_id do not match",
                lot_id=lot_id,
                requested_product_id=requested_product_id,
                actual_product_id=actual_product_id,
            )

        stage_order = await run_in_threadpool(
            get_pid_primary_path,
            actual_product_id,
            path_start_time=None,
            path_end_time=snapshot_end,
        )

        rows: list[LotProgressRow] = []
        # Request-level summary: collect the snapshots that were skipped/degraded (the producer
        # tags them with a per-cut note) grouped by reason, so the caller sees up front which
        # snapshot times were not fully projected -- without noising up every row.
        notes_by_detail: dict[str, list[datetime]] = {}
        for entry in lot_full_data:
            stage_details_src = entry.get("stage_details") or []
            stage_details = [
                StageDetail(
                    stage_name=sd["stage_name"],
                    start_time=datetime.fromisoformat(sd["start_time"]),
                    stage_type=sd["stage_type"],
                    cumulate_h=sd["cumulate_h"],
                    duration_z=sd["duration_z"],
                    comparisons=_build_comparisons(sd),
                )
                for sd in stage_details_src
            ]

            note = entry.get("note")
            if note:
                notes_by_detail.setdefault(note, []).append(entry["snapshot_t"])

            rows.append(
                LotProgressRow(
                    snapshot_at=entry["snapshot_t"],
                    path_id=entry.get("path_id"),
                    path_edit_distance=entry.get("path_edit_distance"),
                    stage_details=stage_details,
                )
            )

        rows.sort(key=lambda r: r.snapshot_at)

        progress_notes = [
            LotProgressNote(detail=detail, snapshot_times=sorted(times))
            for detail, times in notes_by_detail.items()
        ]

        response = LotProgressResponse(
            lot_id=lot_id,
            product_id=actual_product_id,
            snapshots=rows,
            stage_order=stage_order,
            notes=progress_notes,
        )
        logger.opt(lazy=True).trace("Response: \n{}", lambda: response)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error fetching lot progress for {lot_id}: {e}")
        raise_api_error(status_code=500, reason="Failed to fetch lot progress", lot_id=lot_id)
