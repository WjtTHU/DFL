from datetime import datetime
from math import isfinite
from typing import Optional

from pydantic import BaseModel, Field, field_serializer


class LotProgressRequest(BaseModel):
    """Request model for lot-level HWM queries."""

    product_id: Optional[str] = Field(
        default=None,
        description="Optional product ID to validate against the lot's actual product",
    )
    refresh: bool = Field(default=False, description="Force recalculation and bypass cache")
    snapshot_start: Optional[datetime] = Field(
        default=None, description="Optional snapshot window start time"
    )
    snapshot_end: Optional[datetime] = Field(
        default=None, description="Optional snapshot window end time"
    )


class StageDetailComp(BaseModel):
    dev_vs_1d: Optional[float] = Field(None)
    dev_vs_7d: Optional[float] = Field(None)
    dev_vs_1m: Optional[float] = Field(None)
    dev_vs_1q: Optional[float] = Field(None)
    dev_vs_6m: Optional[float] = Field(None)
    dev_vs_1y: Optional[float] = Field(None)

    avg_cum_1d: Optional[float] = Field(None)
    avg_cum_7d: Optional[float] = Field(None)
    avg_cum_1m: Optional[float] = Field(None)
    avg_cum_1q: Optional[float] = Field(None)
    avg_cum_6m: Optional[float] = Field(None)
    avg_cum_1y: Optional[float] = Field(None)

    @field_serializer(
        "dev_vs_1d",
        "dev_vs_7d",
        "dev_vs_1m",
        "dev_vs_1q",
        "dev_vs_6m",
        "dev_vs_1y",
        "avg_cum_1d",
        "avg_cum_7d",
        "avg_cum_1m",
        "avg_cum_1q",
        "avg_cum_6m",
        "avg_cum_1y",
    )
    def serialize_float(self, value: Optional[float]) -> Optional[float]:
        if value is None or isfinite(value):
            return value
        return None

    class Config:
        validate_by_name = True


class StageDetail(BaseModel):
    """Detailed time when a Stage occurs, and the way it is classified"""

    stage_name: str
    start_time: datetime
    stage_type: str = Field(..., pattern="^(actual|interpolated|predicted)$")
    cumulate_h: float
    duration_z: Optional[float]
    comparisons: StageDetailComp

    @field_serializer("cumulate_h", "duration_z")
    def serialize_float(self, value: Optional[float]) -> Optional[float]:
        if value is None or isfinite(value):
            return value
        return None


class LotProgressRow(BaseModel):
    """A Row in Lot-level HWM"""

    snapshot_at: datetime
    path_id: int | None = Field(
        default=None,
        description="Path id selected for this snapshot, if available",
    )
    path_edit_distance: int | None = Field(
        default=None,
        description="Levenshtein edit distance from the previous snapshot path",
    )
    stage_details: list[StageDetail]


class LotProgressNote(BaseModel):
    """A request-level annotation covering one or more snapshot times skipped for the same reason."""

    detail: str = Field(
        description="Human-readable reason the affected snapshots were skipped/degraded"
    )
    snapshot_times: list[datetime] = Field(
        description="The snapshot_at times this annotation applies to"
    )


class LotProgressResponse(BaseModel):
    """Response model for lot progress data"""

    lot_id: str
    product_id: str
    snapshots: list[LotProgressRow]
    stage_order: list[str]
    notes: list[LotProgressNote] = Field(
        default_factory=list,
        description=(
            "Request-level annotations: snapshot times whose full projection was skipped/degraded "
            "and why (e.g. insufficient product history). Empty when every snapshot was fully built."
        ),
    )
