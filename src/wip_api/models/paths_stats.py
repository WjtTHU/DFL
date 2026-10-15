from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TimeStats(BaseModel):
    """Time statistics for a cluster."""

    earliest_start: datetime | None = Field(
        description="Earliest start time of lots in the cluster"
    )
    latest_start: datetime | None = Field(description="Latest start time of lots in the cluster")
    mean_start: datetime | None = Field(description="Mean start time of lots in the cluster")
    median_start: datetime | None = Field(description="Median start time of lots in the cluster")


class ClusterInfo(BaseModel):
    """Information about a single cluster."""

    cluster_id: int = Field(description="Cluster ID")
    lot_count: int = Field(description="Number of lots in the cluster")
    wafer_qty: int = Field(description="Wafer quantity (lot_count * 25)")
    path: list[str] = Field(description="Primary path for the cluster")
    time_stats: TimeStats = Field(description="Time statistics for the cluster")
    type: str = Field(description="Type of path: 'Primary' or 'Secondary'")


class ProductPathsStats(BaseModel):
    """Paths and statistics for a single product at one snapshot cut."""

    product_id: str = Field(description="Product ID")
    snapshot_end: datetime = Field(
        description="Snapshot cut time; stats are cumulative from earliest history up to this time"
    )
    clusters: list[ClusterInfo] = Field(description="List of clusters with their paths and stats")


class PathsStatsRequest(BaseModel):
    """Request model for getting paths with statistics."""

    snapshot_start: datetime = Field(description="Start time of snapshots to analyze (inclusive)")
    snapshot_end: datetime = Field(description="End time of snapshots to analyze (inclusive)")
    product_ids: list[str] | None = Field(
        default=None,
        description="Product IDs to analyze. If None, analyze all products in the time range",
    )
    rules: dict[str, Any] | None = Field(
        default=None,
        description="Identification rules. May include clustering parameters (clustering_method, distance_metric, eps, min_samples, primary_cluster_ratio) and sol_timeout (int, seconds, default 300)",
    )


class PathsStatsResponse(BaseModel):
    """Response model for paths with statistics."""

    products: list[ProductPathsStats] = Field(
        description="Per-product path clusters for each snapshot cut in the requested window"
    )
    reason: str | None = Field(
        default=None,
        description="Reason for empty or partial response; null when products are available",
    )


class PathsPlotRequest(BaseModel):
    """Request model for primary/secondary path network visualization."""

    snapshot_start: datetime = Field(description="Start time of snapshots to analyze (inclusive)")
    snapshot_end: datetime = Field(description="End time of snapshots to analyze (inclusive)")
    align: bool = Field(default=True, description="Align the main path horizontally in the graph")
    refresh: bool = Field(default=False, description="Force recalculation and bypass cache")
    rules: dict[str, Any] | None = Field(
        default=None,
        description="Identification rules. Must include clustering_method; may include distance_metric, eps, min_samples, primary_cluster_ratio, and sol_timeout (int, seconds, default 300)",
    )
