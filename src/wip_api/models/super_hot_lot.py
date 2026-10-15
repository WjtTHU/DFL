from datetime import datetime

from pydantic import BaseModel, Field


class SuperHotLotPlotRequest(BaseModel):
    """Request model for super hot lot cumulative-time plot."""

    snapshot_cut: datetime = Field(..., description="Snapshot cut time for lot progress data")
    refresh: bool = Field(default=False, description="Force recalculation and bypass cache")
