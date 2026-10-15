"""Pydantic models for stage load ratio API responses."""

from datetime import datetime
from math import isfinite
from typing import Dict

from pydantic import BaseModel, Field, field_serializer


class StageLoadRatioRequest(BaseModel):
    """Request model for stage load ratio queries."""

    product_id: str | None = Field(default=None, description="Product ID to analyze (optional)")
    snapshot_start: datetime | None = Field(
        default=None,
        description="Optional lower bound for snapshot_cut filtering",
    )
    snapshot_end: datetime | None = Field(
        default=None,
        description="Optional upper bound for snapshot_cut filtering",
    )


class StageLoadMetrics(BaseModel):
    """Load metrics for a single stage."""

    avg_duration_seconds: float | None = Field(
        description="Average duration of the stage in seconds"
    )
    lot_counts: list[int] = Field(
        description="Daily lot counts aligned with calendar_times for the given snapshot"
    )
    wafer_sums: list[int] = Field(
        description=(
            "Daily wafer sums aligned with calendar_times, using each lot's first snapshot wafer "
            "quantity"
        )
    )
    rolling_7d: list[float | None] = Field(
        description="7-day rolling stage load on calendar_times, expressed as wafers/day"
    )
    rolling_1m: list[float | None] = Field(
        description="1-month rolling stage load on calendar_times, expressed as wafers/day"
    )
    rolling_3m: list[float | None] = Field(
        description="3-month rolling stage load on calendar_times, expressed as wafers/day"
    )
    rolling_6m: list[float | None] = Field(
        description="6-month rolling stage load on calendar_times, expressed as wafers/day"
    )
    rolling_1y: list[float | None] = Field(
        description="1-year rolling stage load on calendar_times, expressed as wafers/day"
    )

    @field_serializer("avg_duration_seconds")
    def serialize_float(self, value: float | None) -> float | None:
        if value is None:
            return None
        return value if isfinite(value) else None

    @field_serializer("rolling_7d", "rolling_1m", "rolling_3m", "rolling_6m", "rolling_1y")
    def serialize_float_list(self, values: list[float | None]) -> list[float | None]:
        out: list[float | None] = []
        for value in values:
            if value is None:
                out.append(None)
            else:
                out.append(value if isfinite(value) else None)
        return out


class SnapshotLoadRatio(BaseModel):
    """Load ratios for all stages at a specific snapshot cut time."""

    snapshot_cut: datetime = Field(
        description="The snapshot cut time for which these metrics were calculated"
    )
    stage_metrics: Dict[str, StageLoadMetrics] = Field(
        description="Dictionary mapping stage names to their load metrics"
    )
    calendar_times: list[datetime] = Field(
        description="Calendar times corresponding to the rolling metric arrays"
    )


class StageLoadRatioResponse(BaseModel):
    """Response model for stage load ratio calculation."""

    product_id: str = Field(description="Product ID for which load ratios were calculated")
    stage_order: list[str] = Field(
        description="Ordered list of stage names in the product's primary path"
    )
    load_ratios: list[SnapshotLoadRatio] = Field(
        description="Load ratio data for each snapshot cut time, including raw lot counts and wafer sums"
    )
    reason: str | None = Field(
        default=None,
        description="Reason for empty or partial response; null when data is available",
    )
