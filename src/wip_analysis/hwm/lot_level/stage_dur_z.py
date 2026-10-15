from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any

import plotly.graph_objects as go

from .all_history import get_lot_prog_over_snapshots


class LotProgressZNoDataError(ValueError):
    """Raised when no lot progress rows are available for plotting."""


@dataclass(frozen=True)
class LotProgressZPlot:
    fig: go.Figure
    warnings: tuple[str, ...]
    snapshot_count: int
    stage_count: int
    z_value_count: int
    filtered_value_count: int


def _snapshot_label(snapshot_t: datetime) -> str:
    return snapshot_t.strftime("%Y-%m-%d %H:%M")


def _first_stage_details_by_name(stage_details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for stage_detail in stage_details:
        stage_name = stage_detail.get("stage_name")
        if stage_name and stage_name not in by_name:
            by_name[stage_name] = stage_detail
    return by_name


def build_lot_progress_z_plot(
    lot_id: str,
    *,
    refresh: bool = False,
    snapshot_start: datetime | None = None,
    snapshot_end: datetime | None = None,
    z_abs_limit: float = 10.0,
) -> LotProgressZPlot:
    data = get_lot_prog_over_snapshots(
        lot_id,
        refresh=refresh,
        snapshot_start=snapshot_start,
        snapshot_end=snapshot_end,
    )

    if not data:
        raise LotProgressZNoDataError(f"No data found for lot {lot_id}")

    stages_set: set[str] = set()
    snapshots: list[datetime] = []
    stage_cumulate_map: dict[str, float] = {}

    for entry in data:
        snapshots.append(entry["snapshot_t"])
        for stage_detail in entry.get("stage_details", []):
            stage_name = stage_detail.get("stage_name")
            cumulate_h = stage_detail.get("cumulate_h")
            if not stage_name:
                continue

            stages_set.add(stage_name)
            if isinstance(cumulate_h, int | float) and isfinite(float(cumulate_h)):
                current = stage_cumulate_map.get(stage_name)
                if current is None or float(cumulate_h) < current:
                    stage_cumulate_map[stage_name] = float(cumulate_h)

    stages = sorted(stages_set, key=lambda stage: stage_cumulate_map.get(stage, float("inf")))
    z_matrix: list[list[float | None]] = []
    hover_text: list[list[str]] = []
    z_value_count = 0
    filtered_value_count = 0

    for entry in data:
        stage_details_by_name = _first_stage_details_by_name(entry.get("stage_details", []))
        snapshot_z_values: list[float | None] = []
        snapshot_hover: list[str] = []
        snapshot_t = entry["snapshot_t"]

        for stage in stages:
            stage_detail = stage_details_by_name.get(stage)
            z_raw = stage_detail.get("duration_z") if stage_detail else None

            if z_raw is None:
                snapshot_z_values.append(None)
                snapshot_hover.append(f"Stage: {stage}<br>No duration z-score")
                continue

            assert stage_detail is not None
            z_value = float(z_raw)
            if not isfinite(z_value):
                snapshot_z_values.append(None)
                snapshot_hover.append(f"Stage: {stage}<br>No finite duration z-score")
                continue

            if abs(z_value) > z_abs_limit:
                filtered_value_count += 1
                snapshot_z_values.append(None)
                snapshot_hover.append(
                    f"Stage: {stage}<br>"
                    f"Snapshot: {_snapshot_label(snapshot_t)}<br>"
                    f"Z-score: {z_value:.2f} (filtered)<br>"
                    f"Type: {stage_detail.get('stage_type')}"
                )
                continue

            z_value_count += 1
            snapshot_z_values.append(z_value)
            snapshot_hover.append(
                f"Stage: {stage}<br>"
                f"Snapshot: {_snapshot_label(snapshot_t)}<br>"
                f"Z-score: {z_value:.2f}<br>"
                f"Type: {stage_detail.get('stage_type')}"
            )

        z_matrix.append(snapshot_z_values)
        hover_text.append(snapshot_hover)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=stages,
            y=[_snapshot_label(s) for s in snapshots],
            colorscale="RdBu_r",
            zmid=0,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorbar=dict(title="Duration<br>Z-Score"),
        )
    )

    fig.update_layout(
        title=f"Lot Progress Duration Z-Scores<br>Lot ID: {lot_id}",
        xaxis_title="Stage (ordered by earliest occurrence)",
        yaxis_title="Snapshot Time",
        height=max(650, min(1800, 220 + 12 * len(snapshots))),
        margin=dict(l=90, r=40, t=90, b=160),
        xaxis=dict(
            tickangle=-45,
            scaleanchor="y",
            scaleratio=1,
            constrain="domain",
        ),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
    )

    warnings: list[str] = []
    total_cells = len(snapshots) * len(stages)
    if total_cells and z_value_count == 0:
        warnings.append(
            "No duration_z values are available for this lot, so the heatmap has no "
            "colored cells. Check whether product-level duration stats have at least "
            "two samples per stage and non-null std_duration_seconds."
        )
    if filtered_value_count:
        warnings.append(
            f"Filtered {filtered_value_count} duration_z values outside +/-{z_abs_limit:g}."
        )

    return LotProgressZPlot(
        fig=fig,
        warnings=tuple(warnings),
        snapshot_count=len(snapshots),
        stage_count=len(stages),
        z_value_count=z_value_count,
        filtered_value_count=filtered_value_count,
    )
