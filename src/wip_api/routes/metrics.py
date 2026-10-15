from fastapi import APIRouter

from wip_api.utils.inflight import inflight_requests

router = APIRouter()


@router.get("/metrics")
async def metrics():
    """Return lightweight process-local runtime metrics."""
    return {
        "inflight_requests": inflight_requests.snapshot(),
    }
