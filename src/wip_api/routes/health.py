from fastapi import APIRouter

router = APIRouter()


@router.post("/health")
async def health_check():
    """Return API liveness status for readiness checks."""
    return {"status": "ok", "message": "WIP Prediction API is running"}
