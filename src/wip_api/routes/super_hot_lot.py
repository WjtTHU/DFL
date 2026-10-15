from datetime import datetime

import plotly.graph_objects as go
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from loguru import logger

from wip_analysis.hwm.product_level.super_hot_lot import (
    SuperHotLotNoDataError,
    build_super_hot_lot_plot,
)
from wip_api.models.super_hot_lot import SuperHotLotPlotRequest
from wip_api.utils.api_errors import raise_api_error
from wip_api.utils.refresh_cache import clear_refresh_lru_caches

router = APIRouter()


def _super_hot_lot_html(fig: go.Figure) -> str:
    return fig.to_html(
        full_html=True,
        include_plotlyjs=True,
        config={"responsive": True},
        default_width="100%",
        default_height="900px",
    )


async def _super_hot_lot_html_response(
    product_id: str,
    snapshot_cut: datetime,
    refresh: bool,
) -> HTMLResponse:
    logger.info(
        "Fetching super hot lot plot for {}, refresh={}, snapshot_cut={}",
        product_id,
        refresh,
        snapshot_cut,
    )

    try:
        if refresh:
            clear_refresh_lru_caches()

        fig = await run_in_threadpool(build_super_hot_lot_plot, product_id, snapshot_cut)
        return HTMLResponse(_super_hot_lot_html(fig))
    except HTTPException:
        raise
    except SuperHotLotNoDataError as e:
        raise_api_error(
            status_code=404,
            reason=str(e),
            product_id=product_id,
            snapshot_cut=snapshot_cut,
        )
    except Exception as e:
        logger.exception(
            "Error building super hot lot plot for {} at {}: {}",
            product_id,
            snapshot_cut,
            e,
        )
        raise_api_error(
            status_code=500,
            reason="Failed to build super hot lot plot",
            product_id=product_id,
            snapshot_cut=snapshot_cut,
        )


@router.post("/products/{product_id:path}/super-hot-lot.html", response_class=HTMLResponse)
async def get_super_hot_lot_plot_html(
    product_id: str,
    request: SuperHotLotPlotRequest,
) -> HTMLResponse:
    return await _super_hot_lot_html_response(
        product_id=product_id,
        snapshot_cut=request.snapshot_cut,
        refresh=request.refresh,
    )
