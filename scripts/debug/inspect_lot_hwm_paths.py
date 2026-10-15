#!/usr/bin/env python
"""Inspect lot HWM response paths against cached path rows.

Examples:
    python -B scripts/debug/inspect_lot_hwm_paths.py lot_hwm_dedup.json
    python -B scripts/debug/inspect_lot_hwm_paths.py lot_hwm_dedup.json --details 20
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import yaml
from rapidfuzz.distance import Levenshtein


@dataclass(frozen=True)
class DbConfig:
    conninfo: str


@dataclass(frozen=True)
class SqlConfig:
    path_table: str
    path_id_col: str
    product_id_col: str
    path_col: str
    start_time_col: str
    end_time_col: str


@dataclass(frozen=True)
class SnapshotPath:
    index: int
    snapshot_at: str
    response_path_id: int | None
    response_edit_distance: int | None
    stage_order: list[str]


def _load_db_config(path: Path) -> DbConfig:
    with path.open("r", encoding="utf-8") as f:
        db_cfg = yaml.safe_load(f)["database"]
    conninfo = (
        f"postgresql://{db_cfg['user']}:{db_cfg['password']}"
        f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['dbname']}"
    )
    return DbConfig(conninfo=conninfo)


def _load_sql_config(path: Path) -> SqlConfig:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["database"]
    tables = cfg["tables"]
    columns = cfg["columns"]
    return SqlConfig(
        path_table=tables["path_table"],
        path_id_col=columns["path_id"],
        product_id_col=columns["product_id"],
        path_col=columns["path"],
        start_time_col=columns["start_time"],
        end_time_col=columns["end_time"],
    )


def _load_snapshots(json_path: Path) -> tuple[str | None, list[SnapshotPath]]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Expected top-level JSON object")

    product_id = payload.get("product_id")
    snapshots_raw = payload.get("snapshots")
    if not isinstance(snapshots_raw, list):
        raise ValueError("Expected top-level 'snapshots' list")

    snapshots: list[SnapshotPath] = []
    for idx, snapshot in enumerate(snapshots_raw):
        if not isinstance(snapshot, dict):
            raise ValueError(f"Snapshot #{idx} is not an object")

        stage_details = snapshot.get("stage_details")
        if not isinstance(stage_details, list):
            raise ValueError(f"Snapshot #{idx} has no stage_details list")

        stage_order = [
            str(row["stage_name"])
            for row in stage_details
            if isinstance(row, dict) and row.get("stage_name") is not None
        ]
        path_id_raw = snapshot.get("path_id")
        path_id = int(path_id_raw) if path_id_raw is not None else None
        response_edit_raw = snapshot.get("path_edit_distance")
        response_edit = int(response_edit_raw) if response_edit_raw is not None else None
        snapshots.append(
            SnapshotPath(
                index=idx,
                snapshot_at=str(snapshot.get("snapshot_at")),
                response_path_id=path_id,
                response_edit_distance=response_edit,
                stage_order=stage_order,
            )
        )

    return str(product_id) if product_id is not None else None, snapshots


def _fetch_paths_by_id(
    db_config: DbConfig,
    sql_config: SqlConfig,
    path_ids: list[int],
) -> dict[int, dict[str, Any]]:
    if not path_ids:
        return {}

    placeholders = ", ".join(["%s"] * len(path_ids))
    query = f"""
        SELECT
            {sql_config.path_id_col},
            {sql_config.product_id_col},
            {sql_config.path_col},
            {sql_config.start_time_col},
            {sql_config.end_time_col}
        FROM {sql_config.path_table}
        WHERE {sql_config.path_id_col} IN ({placeholders})
        ORDER BY {sql_config.path_id_col}
    """
    with psycopg.connect(db_config.conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(query, path_ids)
            rows = cur.fetchall()

    out: dict[int, dict[str, Any]] = {}
    for path_id, product_id, path, start_time, end_time in rows:
        out[int(path_id)] = {
            "product_id": product_id,
            "path": [str(stage) for stage in (path or [])],
            "start_time": start_time,
            "end_time": end_time,
        }
    return out


def _edit_distance(left: list[str], right: list[str]) -> int:
    return int(Levenshtein.distance(left, right))


def _same_position_count(left: list[str], right: list[str]) -> int:
    return sum(1 for a, b in zip(left, right, strict=False) if a == b)


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def _format_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def _norm_distance(distance: int | None, left_len: int, right_len: int) -> float | None:
    if distance is None:
        return None
    denom = max(left_len, right_len)
    if denom == 0:
        return 0.0
    return distance / denom


def _table(headers: list[str], rows: list[list[object]]) -> str:
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in string_rows))
        for idx in range(len(headers))
    ]
    header_line = "  ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers)))
    sep_line = "  ".join("-" * widths[idx] for idx in range(len(headers)))
    body = [
        "  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) for row in string_rows
    ]
    return "\n".join([header_line, sep_line, *body])


def _position_mismatches(
    left: list[str],
    right: list[str],
    limit: int,
) -> list[tuple[int, str | None, str | None]]:
    out: list[tuple[int, str | None, str | None]] = []
    max_len = max(len(left), len(right))
    for idx in range(max_len):
        left_stage = left[idx] if idx < len(left) else None
        right_stage = right[idx] if idx < len(right) else None
        if left_stage != right_stage:
            out.append((idx, left_stage, right_stage))
            if len(out) >= limit:
                break
    return out


def _set_diff_sample(left: list[str], right: list[str], limit: int) -> tuple[list[str], list[str]]:
    left_set = set(left)
    right_set = set(right)
    return sorted(left_set - right_set)[:limit], sorted(right_set - left_set)[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare lot HWM response stage order with DB path rows and adjacent path_id paths."
        )
    )
    parser.add_argument("json_file", type=Path, help="Path to lot_hwm*.json response file")
    parser.add_argument(
        "--sql-config",
        type=Path,
        default=Path("config/sql_base.yaml"),
        help="SQL table/column config YAML",
    )
    parser.add_argument(
        "--db-config",
        type=Path,
        default=Path("config/postgres.yaml"),
        help="DB connection config YAML",
    )
    parser.add_argument(
        "--details",
        type=int,
        default=0,
        help="Print first N positional mismatches for each non-exact comparison",
    )
    parser.add_argument(
        "--set-diff-limit",
        type=int,
        default=8,
        help="Number of missing/extra stage names to show per row",
    )
    args = parser.parse_args()

    product_id, snapshots = _load_snapshots(args.json_file)
    path_ids = sorted(
        {
            snapshot.response_path_id
            for snapshot in snapshots
            if snapshot.response_path_id is not None
        }
    )
    sql_config = _load_sql_config(args.sql_config)
    db_config = _load_db_config(args.db_config)
    paths_by_id = _fetch_paths_by_id(db_config, sql_config, path_ids)

    print(f"file: {args.json_file}")
    print(f"response_product_id: {product_id}")
    print(f"snapshots: {len(snapshots)}")
    print(f"distinct_path_ids: {len(path_ids)}")
    missing_path_ids = [path_id for path_id in path_ids if path_id not in paths_by_id]
    if missing_path_ids:
        print(f"missing_path_ids_in_db: {missing_path_ids}")
    print()

    rows: list[list[object]] = []
    prev_db_path: list[str] | None = None
    prev_path_id: int | None = None
    details: list[str] = []

    for snapshot in snapshots:
        path_row = (
            paths_by_id.get(snapshot.response_path_id)
            if snapshot.response_path_id is not None
            else None
        )
        db_path = path_row["path"] if path_row is not None else []
        stage_order = snapshot.stage_order

        stage_vs_db_dist = _edit_distance(db_path, stage_order) if path_row is not None else None
        prev_db_path_dist = (
            None
            if prev_db_path is None or path_row is None
            else _edit_distance(prev_db_path, db_path)
        )
        same_pos = _same_position_count(db_path, stage_order) if path_row is not None else 0
        exact = path_row is not None and db_path == stage_order
        missing_from_stage, extra_in_stage = _set_diff_sample(
            db_path,
            stage_order,
            args.set_diff_limit,
        )
        row_product_id = path_row["product_id"] if path_row is not None else None

        rows.append(
            [
                snapshot.index,
                snapshot.snapshot_at,
                snapshot.response_path_id,
                row_product_id,
                len(stage_order),
                len(db_path),
                _format_bool(exact),
                stage_vs_db_dist if stage_vs_db_dist is not None else "NA",
                _format_float(_norm_distance(stage_vs_db_dist, len(db_path), len(stage_order))),
                same_pos,
                prev_path_id if prev_path_id is not None else "NA",
                prev_db_path_dist if prev_db_path_dist is not None else "NA",
                _format_float(
                    _norm_distance(prev_db_path_dist, len(prev_db_path or []), len(db_path))
                ),
                snapshot.response_edit_distance
                if snapshot.response_edit_distance is not None
                else "NA",
                len(set(db_path) - set(stage_order)),
                len(set(stage_order) - set(db_path)),
            ]
        )

        if args.details and path_row is not None and not exact:
            details.append(
                f"snapshot #{snapshot.index} {snapshot.snapshot_at} path_id={snapshot.response_path_id}"
            )
            details.append(
                f"  db_path vs response_stage positional mismatches (first {args.details}):"
            )
            for idx, db_stage, response_stage in _position_mismatches(
                db_path,
                stage_order,
                args.details,
            ):
                details.append(f"    pos={idx}: db={db_stage!r} response={response_stage!r}")
            if missing_from_stage or extra_in_stage:
                details.append(f"  db_path stages missing from response: {missing_from_stage}")
                details.append(f"  response stages not in db_path: {extra_in_stage}")
            details.append("")

        if args.details and prev_db_path is not None and path_row is not None and prev_db_path_dist:
            details.append(
                "DB path change "
                f"{prev_path_id} -> {snapshot.response_path_id} "
                f"at snapshot #{snapshot.index} {snapshot.snapshot_at}: "
                f"lev={prev_db_path_dist}"
            )
            details.append(
                "  previous db_path vs current db_path positional mismatches "
                f"(first {args.details}):"
            )
            for idx, prev_stage, current_stage in _position_mismatches(
                prev_db_path,
                db_path,
                args.details,
            ):
                details.append(f"    pos={idx}: prev={prev_stage!r} current={current_stage!r}")
            missing_from_current, added_in_current = _set_diff_sample(
                prev_db_path,
                db_path,
                args.set_diff_limit,
            )
            if missing_from_current or added_in_current:
                details.append(f"  previous stages missing from current: {missing_from_current}")
                details.append(f"  current stages not in previous: {added_in_current}")
            details.append("")

        if path_row is not None:
            prev_db_path = db_path
            prev_path_id = snapshot.response_path_id

    print(
        _table(
            [
                "#",
                "snapshot_at",
                "path_id",
                "path_product_id",
                "stage_len",
                "db_path_len",
                "exact",
                "stage_vs_db_lev",
                "stage_vs_db_norm",
                "same_pos",
                "prev_path_id",
                "db_path_vs_prev_lev",
                "db_path_vs_prev_norm",
                "response_path_edit",
                "db_missing",
                "resp_extra",
            ],
            rows,
        )
    )

    if details:
        print()
        print("details:")
        print("\n".join(details))

    return 0


if __name__ == "__main__":
    sys.exit(main())
