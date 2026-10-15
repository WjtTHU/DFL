from datetime import datetime

from pydantic import BaseModel, Field


class ProdPathResponse(BaseModel):
    """Response model for product path"""

    product_id: str
    primary_path: list[str]


class ProdStageStatsRow(BaseModel):
    """
    Data Model for a row in Product HWM.
    Should have multiple `stage_lot_cnt` on a certain `calendar_time`.
    """

    calendar_time: datetime = Field(
        description="The Calendar Time to count lot numbers on, would be used as the first column of the product-level HWM."
    )
    snapshot_cut: datetime = Field(
        description="The snapshots before this time would be used to count/predict the lot counts on the calendar_time."
    )
    stage_lot_cnt: list[int] = Field(
        description="List of lot counts at each stage, in the order of product primary path."
    )


class ProdStageStatsResponse(BaseModel):
    product_id: str
    stage_order: list[str]
    wip_stats: list[ProdStageStatsRow]
    reason: str | None = Field(
        default=None,
        description="Reason for empty or partial response; null when data is available",
    )


class ProdStageStatsRequest(BaseModel):
    product_id: str | None = Field(
        default=None,
        description="Optional product guard; must match the path product_id when provided",
    )
    snapshot_start: datetime | None = Field(
        default=None,
        description="Optional lower bound for snapshot_cut filtering",
    )
    snapshot_end: datetime | None = Field(
        default=None,
        description="Optional upper bound for snapshot_cut filtering",
    )
