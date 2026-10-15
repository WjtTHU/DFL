from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator


class StageCapacityRequest(BaseModel):
    """Request model for stage capacity queries."""

    stage_name: str | list[str] | None = Field(
        default=None,
        description=(
            "Filter by a specific stage name or an ordered list of stage names. "
            "Blank values are treated as no stage filter."
        ),
    )
    product_id: str | None = Field(
        default=None,
        description="Filter by specific product ID. Blank values are treated as no product filter.",
    )
    snapshot_start: datetime | None = Field(
        default=None, description="Start time (inclusive) for stage entry time filter"
    )
    snapshot_end: datetime | None = Field(
        default=None, description="End time (inclusive) for stage entry time filter"
    )

    @field_validator("stage_name", mode="before")
    @classmethod
    def normalize_stage_name(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        if isinstance(value, list):
            normalized: list[object] = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, str):
                    stripped = item.strip()
                    if stripped:
                        normalized.append(stripped)
                    continue
                normalized.append(item)
            return normalized or None
        return value

    @field_validator("product_id", mode="before")
    @classmethod
    def normalize_product_id(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class StageCapacityRow(BaseModel):
    """
    Data model for a row in Stage Capacity statistics.
    Represents lot and wafer volume entering a stage on a specific date.
    """

    calendar_date: date = Field(description="The date when lots entered the stage")
    stage_name: str = Field(description="Name of the manufacturing stage")
    entering_lot_count: int = Field(
        description="Number of distinct lots that entered this stage on this date"
    )
    entering_wafer_count: int = Field(
        description=(
            "Sum of wafer quantities for lots that entered this stage on this date, "
            "using each lot's latest snapshot at or before stage entry time"
        )
    )


class StageCapacitySeries(BaseModel):
    """Stage entry statistics for a single stage."""

    stage_name: str = Field(description="Requested manufacturing stage name")
    wafer_data: list[StageCapacityRow] = Field(
        description="Stage entry statistics for the stage ordered by calendar date"
    )


class StageCapacityResponse(BaseModel):
    """Response model for stage capacity query"""

    product_id: str | None = Field(
        default=None, description="Product ID filter applied (None if all products)"
    )
    stage_name: str | list[str] | None = Field(
        default=None,
        description="Stage filter applied. Preserves the original string or ordered list input.",
    )
    capacity_data: list[StageCapacitySeries] = Field(
        description=(
            "Per-stage lot and wafer entry statistics. When a list of stages is requested, "
            "the list order matches the request order."
        ),
    )
    reason: str | None = Field(
        default=None,
        description="Reason for empty or partial response; null when data is available",
    )
