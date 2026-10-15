from datetime import date, datetime
from math import isfinite
from typing import Optional

from pydantic import BaseModel, Field, field_serializer


class StageDurationRequest(BaseModel):
    """Request model for stage duration queries."""

    stage_name: Optional[str] = Field(default=None, description="Filter by specific stage name")
    product_id: Optional[str] = Field(default=None, description="Filter by specific product ID")
    snapshot_start: Optional[datetime] = Field(
        default=None,
        description="Optional lower bound for stage start_time filtering",
    )
    snapshot_end: Optional[datetime] = Field(
        default=None,
        description="Optional upper bound for stage start_time filtering",
    )


class StageDurationRow(BaseModel):
    """
    Data model for a row in Stage Duration (Cycle Time) statistics.
    Represents the time a lot spent in a specific stage, enriched with its initial wafer count.
    """

    product_id: str = Field(description="Product identifier")
    lot_id: str = Field(description="Lot identifier")
    stage_name: str = Field(description="Name of the manufacturing stage")
    entry_date: date = Field(description="Date when the lot entered the stage")
    start_time: datetime = Field(description="Timestamp when the lot entered the stage")
    end_time: datetime = Field(description="Timestamp when the lot exited the stage")
    wafer_count: int | None = Field(
        default=None,
        description=(
            "Wafer quantity for this lot from its first snapshot occurrence; null when unavailable"
        ),
    )
    duration_seconds: float = Field(
        description="Duration the lot spent in this stage (formatted as interval)"
    )

    @field_serializer("duration_seconds")
    def serialize_float(self, value: float) -> float | None:
        return value if isfinite(value) else None


class StageDurationResponse(BaseModel):
    """Response model for stage duration query"""

    product_id: Optional[str] = Field(
        default=None, description="Product ID filter applied (None if all products)"
    )
    stage_name: Optional[str] = Field(
        default=None, description="Stage name filter applied (None if all stages)"
    )
    duration_data: list[StageDurationRow] = Field(
        description="List of stage duration statistics showing cycle time and wafer count for each lot"
    )
    reason: str | None = Field(
        default=None,
        description="Reason for empty or partial response; null when data is available",
    )
