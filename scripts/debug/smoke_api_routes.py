#!/usr/bin/env python3
"""Smoke-test API routes against a running local backend.

This is intentionally not a correctness test: most endpoints have no stable
golden answer for live data. The script checks HTTP status, response shape, and
basic non-empty signals so regressions are visible without asserting business
counts.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class SmokeCase:
    name: str
    method: str
    path: str
    body: dict[str, Any] | None = None
    expect_statuses: tuple[int, ...] = (200,)
    artifact_ext: str = "json"
    timeout_seconds: float = 180.0


def _json_len(value: Any, key: str) -> int | None:
    if not isinstance(value, dict):
        return None
    item = value.get(key)
    return len(item) if isinstance(item, list | dict | str) else None


def _summarize_json(name: str, payload: Any, status: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": status, "json_type": type(payload).__name__}

    if name == "lot_hwm":
        return {
            "status": status,
            "product_id": payload.get("product_id"),
            "snapshots": _json_len(payload, "snapshots"),
            "stage_order": _json_len(payload, "stage_order"),
        }
    if name in {"load_ratio", "stage_load_ratio"}:
        return {
            "status": status,
            "product_id": payload.get("product_id"),
            "load_ratios": _json_len(payload, "load_ratios"),
            "stage_order": _json_len(payload, "stage_order"),
            "reason": payload.get("reason"),
        }
    if name == "product_hwm":
        return {
            "status": status,
            "product_id": payload.get("product_id"),
            "wip_stats": _json_len(payload, "wip_stats"),
            "stage_order": _json_len(payload, "stage_order"),
            "reason": payload.get("reason"),
        }
    if name == "stage_cap":
        return {
            "status": status,
            "series": _json_len(payload, "capacity_data"),
            "reason": payload.get("reason"),
        }
    if name == "stage_dur":
        return {
            "status": status,
            "rows": _json_len(payload, "duration_data"),
            "reason": payload.get("reason"),
        }
    if name == "paths_stats":
        return {
            "status": status,
            "products": _json_len(payload, "products"),
            "reason": payload.get("reason"),
        }
    if name == "date_out":
        return {
            "status": status,
            "target_dates": payload.get("target_dates"),
            "lot_counts": payload.get("lot_counts"),
            "wafer_counts": payload.get("wafer_counts"),
            "reason": payload.get("reason"),
        }
    if name == "super_hot":
        detail = payload.get("detail") if isinstance(payload.get("detail"), dict) else {}
        return {
            "status": status,
            "reason": detail.get("reason") or payload.get("reason"),
        }

    return {"status": status, "keys": sorted(payload)}


def _request(
    base_url: str,
    case: SmokeCase,
    artifacts_dir: Path,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + case.path
    data = None
    headers: dict[str, str] = {}
    if case.body is not None:
        data = json.dumps(case.body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=case.method)
    started = time.perf_counter()
    response_body: bytes
    status: int
    try:
        with urlopen(request, timeout=case.timeout_seconds) as response:
            status = response.status
            response_body = response.read()
            content_type = response.headers.get("content-type", "")
    except HTTPError as error:
        status = error.code
        response_body = error.read()
        content_type = error.headers.get("content-type", "")
    except URLError as error:
        elapsed = time.perf_counter() - started
        return {
            "name": case.name,
            "ok": False,
            "status": None,
            "elapsed_seconds": round(elapsed, 3),
            "error": str(error),
        }

    elapsed = time.perf_counter() - started
    artifact_path = artifacts_dir / f"{case.name}.{case.artifact_ext}"
    artifact_path.write_bytes(response_body)

    ok = status in case.expect_statuses
    summary: dict[str, Any] = {
        "name": case.name,
        "ok": ok,
        "status": status,
        "elapsed_seconds": round(elapsed, 3),
        "bytes": len(response_body),
        "artifact": str(artifact_path),
    }

    if "application/json" in content_type:
        try:
            payload = json.loads(response_body.decode("utf-8"))
        except json.JSONDecodeError:
            payload = None
        summary.update(_summarize_json(case.name, payload, status))
    elif "text/html" in content_type:
        summary["content_type"] = "html"

    return summary


def _cases(args: argparse.Namespace) -> list[SmokeCase]:
    product = quote(args.product_id, safe="")
    lot = quote(args.lot_id, safe="")
    start = args.snapshot_start
    end = args.snapshot_end
    target_date = args.target_date
    common_window = {"snapshot_start": start, "snapshot_end": end}

    cases = [
        SmokeCase("health", "POST", "/api/v1/health"),
        SmokeCase(
            "lot_hwm",
            "POST",
            f"/api/v1/lots/{lot}",
            {"snapshot_start": start, "snapshot_end": end, "refresh": args.refresh_lot},
        ),
        SmokeCase(
            "lot_progress_z",
            "POST",
            f"/api/v1/lots/{lot}/progress-z.html?"
            + urlencode({"snapshot_start": start, "snapshot_end": end}),
            artifact_ext="html",
        ),
        SmokeCase(
            "load_ratio",
            "POST",
            f"/api/v1/products/{product}/load_ratio",
            common_window,
        ),
        SmokeCase(
            "product_hwm",
            "POST",
            f"/api/v1/products/{product}",
            {"product_id": args.product_id, **common_window},
        ),
        SmokeCase(
            "stage_load_ratio",
            "POST",
            "/api/v1/stage_load_ratio",
            {"product_id": args.product_id, **common_window},
        ),
        SmokeCase(
            "stage_cap",
            "POST",
            "/api/v1/stage_cap",
            {
                "product_id": args.product_id,
                "stage_name": args.stage_names,
                **common_window,
            },
        ),
        SmokeCase(
            "stage_dur",
            "POST",
            "/api/v1/stage_dur",
            {
                "product_id": args.product_id,
                "stage_name": args.stage_names[0],
                **common_window,
            },
        ),
        SmokeCase(
            "paths_stats",
            "POST",
            "/api/v1/paths_stats",
            {
                "snapshot_start": start,
                "snapshot_end": end,
                "product_ids": [args.product_id],
                "rules": {"clustering_method": "hdbscan", "sol_timeout": args.sol_timeout},
            },
        ),
        SmokeCase(
            "paths_html",
            "POST",
            f"/api/v1/products/{product}/paths.html",
            {
                "snapshot_start": start,
                "snapshot_end": end,
                "rules": {"clustering_method": "hdbscan", "sol_timeout": args.sol_timeout},
            },
            artifact_ext="html",
        ),
        SmokeCase(
            "date_out",
            "POST",
            f"/api/v1/products/{product}/date_out",
            {"target_dates": [target_date]},
        ),
        SmokeCase(
            "bottleneck",
            "POST",
            "/api/v1/bottleneck_detection",
            {
                "start_time": args.bottleneck_start,
                "end_time": args.bottleneck_end,
                "product_ids": [args.product_id],
                "method": "daily_qty_variance",
                "rules": {"top_k": 3},
            },
            expect_statuses=(200, 404),
        ),
        SmokeCase(
            "super_hot",
            "POST",
            f"/api/v1/products/{product}/super-hot-lot.html",
            {"snapshot_cut": end},
            expect_statuses=(200, 404),
            artifact_ext="html",
        ),
    ]

    if args.include_full_product_hwm:
        cases.append(
            SmokeCase(
                "product_hwm_full",
                "POST",
                f"/api/v1/products/{product}",
                timeout_seconds=args.heavy_timeout,
            )
        )

    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8724")
    parser.add_argument("--product-id", default="5H0mKnT7qf6hZjrm9r4SfA==")
    parser.add_argument("--lot-id", default="+EzHRhFT3SjBC3CqHWQSOw==")
    parser.add_argument("--snapshot-start", default="2025-02-05 08:30:25")
    parser.add_argument("--snapshot-end", default="2025-02-05 09:30:26")
    parser.add_argument("--target-date", default="2025-02-06")
    parser.add_argument("--bottleneck-start", default="2025-02-01 00:00:00")
    parser.add_argument("--bottleneck-end", default="2025-02-10 00:00:00")
    parser.add_argument("--stage-names", nargs="+", default=["AAHM-DEP", "AAHM-DEP2"])
    parser.add_argument("--sol-timeout", type=int, default=3)
    parser.add_argument("--artifacts-dir", default="/tmp/wip-api-smoke")
    parser.add_argument("--refresh-lot", action="store_true")
    parser.add_argument("--include-full-product-hwm", action="store_true")
    parser.add_argument("--heavy-timeout", type=float, default=240.0)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    results = [_request(args.base_url, case, artifacts_dir) for case in _cases(args)]
    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
