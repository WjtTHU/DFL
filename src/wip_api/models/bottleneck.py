"""Pydantic models for bottleneck detection API."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BottleneckMethod(str, Enum):
    """Supported bottleneck detection methods."""

    HOLD_TIME = "hold_time"
    DAILY_QTY_VARIANCE = "daily_qty_variance"
    THROUGHPUT_VAR_AND_LOAD_RATE = "throughput_var_and_load_rate"


class BottleneckRules(BaseModel):
    """Optional rule overrides for the selected detection method."""

    min_wafer_count: Optional[int] = Field(
        default=None,
        description="Minimum wafer count in hold (hold_time only). Defaults to 10.",
    )
    min_hold_days: Optional[int] = Field(
        default=None,
        description="Minimum consecutive hold days (hold_time only). Defaults to 3.",
    )
    processing_statuses: Optional[list[str]] = Field(
        default=None,
        description="Processing statuses to consider (daily_qty_variance only). Defaults to ['Active'].",
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of top stages to return for ranking-based methods. Defaults to 5.",
    )
    min_daily_throughput: Optional[int] = Field(
        default=None,
        description="Minimum max daily throughput for a stage (throughput_var_and_load_rate only). Defaults to 1.",
    )
    min_days: Optional[int] = Field(
        default=None,
        description="Minimum number of days with throughput records for a stage (throughput_var_and_load_rate only). Defaults to 5.",
    )
    min_utilization: Optional[float] = Field(
        default=None,
        description="Minimum average utilization (mean/max) for a stage (throughput_var_and_load_rate only). Defaults to 0.5.",
    )
    full_load_threshold: Optional[float] = Field(
        default=None,
        description="Threshold to treat a day as 'full load' relative to max throughput (throughput_var_and_load_rate only). Defaults to 0.85.",
    )
    weight_variance: Optional[float] = Field(
        default=None,
        description="Weight of normalized variance in the final score (throughput_var_and_load_rate only). Defaults to 0.5.",
    )
    weight_utilization: Optional[float] = Field(
        default=None,
        description="Weight of utilization in the final score (throughput_var_and_load_rate only). Defaults to 0.25.",
    )
    weight_full_load: Optional[float] = Field(
        default=None,
        description="Weight of full load rate in the final score (throughput_var_and_load_rate only). Defaults to 0.25.",
    )


class BottleneckDetectionRequest(BaseModel):
    """Request body for bottleneck detection."""

    start_time: datetime = Field(description="Start time of the analysis window")
    end_time: datetime = Field(description="End time of the analysis window")
    product_ids: Optional[list[str]] = Field(
        default=None,
        description="Product IDs to analyze (None = all products in range)",
    )
    method: BottleneckMethod = Field(
        description="Detection method: 'hold_time', 'daily_qty_variance' or 'throughput_var_and_load_rate'"
    )
    rules: BottleneckRules = Field(
        default_factory=BottleneckRules,
        description="Method-specific rule overrides",
    )


class BottleneckStage(BaseModel):
    """Detected bottleneck stage details."""

    stage: str = Field(description="Stage name")
    reason: str = Field(description="Reason describing why this stage is bottleneck")


class BottleneckResult(BaseModel):
    """Bottleneck detection result for a single product."""

    product_id: str = Field(description="Product ID")
    primary_path: list[str] = Field(
        description="Primary process route for the product (may be empty if unavailable)"
    )
    bottleneck_stages: list[BottleneckStage] = Field(
        description="Detected bottleneck stages for this product"
    )
