#!/usr/bin/env python
"""
Parallel POST /api/v1/lots/{lot_id} against the local API (see scripts/api/start_api.py).

Jobs are sampled from test_samples.csv (see --samples-csv, --num-samples, --sample-seed).
"""

from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import quote

import polars as pl
import requests


@dataclass(frozen=True)
class LotSnapshotJob:
    lot_id: str
    snapshot_start: str
    snapshot_end: str


def _default_samples_csv() -> Path:
    return Path(__file__).resolve().parents[2] / "test_samples.csv"


def _normalize_snapshot_start(value: str) -> str:
    v = value.strip()
    if "T" in v:
        return v
    return f"{v}T00:00:00"


def _normalize_snapshot_end(value: str) -> str:
    v = value.strip()
    if "T" in v:
        return v
    if " " in v:
        return v.replace(" ", "T", 1)
    return f"{v}T23:59:59"


def fetch_job_params(
    *,
    csv_path: Path,
    num_samples: int,
    sample_seed: int,
) -> Iterator[LotSnapshotJob]:
    """Randomly sample ``num_samples`` rows using ``random.Random(sample_seed)`` (reproducible)."""
    df = pl.read_csv(csv_path)
    lot_col = "Lot ID"
    if lot_col not in df.columns:
        msg = f"CSV {csv_path} must contain a {lot_col!r} column; got {df.columns!r}"
        raise ValueError(msg)
    n = len(df)
    if num_samples > n:
        msg = f"--num-samples ({num_samples}) exceeds rows in CSV ({n})"
        raise ValueError(msg)
    rng = random.Random(sample_seed)
    row_indices = rng.sample(range(n), num_samples)
    # Polars 1.x: select rows by index list (DataFrame.take was removed).
    sampled = df[row_indices, :]
    for row in sampled.iter_rows(named=True):
        yield LotSnapshotJob(
            lot_id=str(row[lot_col]),
            snapshot_start=_normalize_snapshot_start(str(row["snapshot_start"])),
            snapshot_end=_normalize_snapshot_end(str(row["snapshot_end"])),
        )


def build_lot_url(base_url: str, lot_id: str) -> str:
    root = base_url.rstrip("/")
    encoded = quote(lot_id, safe="")
    return f"{root}/api/v1/lots/{encoded}"


def post_lot_snapshot(
    session: requests.Session,
    base_url: str,
    job: LotSnapshotJob,
    timeout_s: float,
):
    url = build_lot_url(base_url, job.lot_id)
    payload = {
        "snapshot_start": job.snapshot_start,
        "snapshot_end": job.snapshot_end,
    }
    try:
        resp = session.post(url, json=payload, timeout=timeout_s)
        return job, resp, None
    except Exception as exc:
        return job, None, exc


def run_parallel(
    jobs: Iterable[LotSnapshotJob],
    *,
    base_url: str,
    max_workers: int,
    timeout_s: float,
) -> int:
    jobs_list = list(jobs)
    if not jobs_list:
        print("No jobs to run.", file=sys.stderr)
        return 1

    exit_code = 0
    with requests.Session() as session, ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {
            pool.submit(post_lot_snapshot, session, base_url, job, timeout_s): job
            for job in jobs_list
        }
        for fut in as_completed(future_to_job):
            job, resp, err = fut.result()
            if err is not None:
                exit_code = 1
                print(f"[FAIL] {job.lot_id}: {err!r}", file=sys.stderr)
                continue
            assert resp is not None
            if not resp.ok:
                exit_code = 1
                body_preview = (resp.text or "")[:500]
                print(
                    f"[HTTP {resp.status_code}] {job.lot_id}: {body_preview}",
                    file=sys.stderr,
                )
            else:
                print(f"[OK] {job.lot_id} -> {resp.status_code}")

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel POST lot snapshot jobs to WIP API.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8724",
        help="API root, e.g. http://localhost:8724",
    )
    parser.add_argument(
        "-n",
        "--max-workers",
        type=int,
        default=8,
        help="Maximum concurrent in-flight requests (thread pool size).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10800.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--samples-csv",
        type=Path,
        default=_default_samples_csv(),
        help="Path to test_samples.csv (default: repo root test_samples.csv).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of rows to randomly draw from the CSV.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for random.Random when drawing rows; same seed + CSV yields the same subset.",
    )
    args = parser.parse_args()

    return run_parallel(
        fetch_job_params(
            csv_path=args.samples_csv,
            num_samples=args.num_samples,
            sample_seed=args.sample_seed,
        ),
        base_url=args.base_url,
        max_workers=args.max_workers,
        timeout_s=args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
