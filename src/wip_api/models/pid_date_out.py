from datetime import date
from typing import Any, cast

from pydantic import BaseModel, Field, model_validator


class ProductDateOutRequest(BaseModel):
    """Request model for product date-out queries."""

    target_dates: list[date] = Field(
        min_length=1,
        description="Target dates for expected output (YYYY-MM-DD), returned in the same order",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_target_dates(cls, data: object) -> object:
        if isinstance(data, dict) and "target_dates" not in data and "target_date" in data:
            normalized_data = dict(cast(dict[str, Any], data))
            target_date = normalized_data.get("target_date")
            return {**normalized_data, "target_dates": [target_date]}
        return data


class ProductDateOutCount(BaseModel):
    """Lot and wafer output counts."""

    lot_count: int = Field(default=0, description="Number of lots")
    wafer_count: int = Field(default=0, description="Number of wafers")


class ProductDateOutSplit(BaseModel):
    """Output counts split by actual/interpolated versus predicted stage type."""

    actual: ProductDateOutCount = Field(default_factory=ProductDateOutCount)
    predicted: ProductDateOutCount = Field(default_factory=ProductDateOutCount)


class ProductDateOutStageOutput(BaseModel):
    """Stage-level output counts for lots that finish on a target date."""

    stage_name: str = Field(description="Stage name")
    actual: ProductDateOutCount = Field(default_factory=ProductDateOutCount)
    predicted: ProductDateOutCount = Field(default_factory=ProductDateOutCount)


class ProductDateOutDayOutput(BaseModel):
    """Output details for one target date."""

    target_date: date = Field(description="Target output date")
    final: ProductDateOutSplit = Field(default_factory=ProductDateOutSplit)
    stages: list[ProductDateOutStageOutput] = Field(default_factory=list)


class ProductDateOutResponse(BaseModel):
    """Response model for product date-out output counts."""

    product_id: str = Field(description="Product identifier")
    target_dates: list[date] = Field(
        description="Target dates for expected output (YYYY-MM-DD), matching the request order"
    )
    lot_counts: list[int] = Field(
        description="Lot counts for each target date, aligned with target_dates by index"
    )
    wafer_counts: list[int] = Field(
        description="Wafer counts for each target date, aligned with target_dates by index"
    )
    actual_lot_counts: list[int] = Field(
        default_factory=list,
        description="Actual/interpolated final lot counts aligned with target_dates by index",
    )
    actual_wafer_counts: list[int] = Field(
        default_factory=list,
        description="Actual/interpolated final wafer counts aligned with target_dates by index",
    )
    predicted_lot_counts: list[int] = Field(
        default_factory=list,
        description="Predicted final lot counts aligned with target_dates by index",
    )
    predicted_wafer_counts: list[int] = Field(
        default_factory=list,
        description="Predicted final wafer counts aligned with target_dates by index",
    )
    date_outputs: list[ProductDateOutDayOutput] = Field(
        default_factory=list,
        description="Per-target-date final and stage outputs split into actual and predicted parts",
    )
    reason: str | None = Field(
        default=None,
        description="Reason for empty or all-zero response; null when output lots are found",
    )
