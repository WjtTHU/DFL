#!/usr/bin/env python
"""Generate DB-backed lot HWM jobs and run them concurrently against the API.

Examples:
    # Evenly sample 2,000 lot+created_at jobs across lots, then run 16 concurrent requests.
    uv run python -B scripts/debug/pressure_lot_hwm_requests.py --max-requests 2000 -n 16

    # Run every parent-lot snapshot in the configured snapshot table.
    uv run python -B scripts/debug/pressure_lot_hwm_requests.py --all -n 32

    # Dry-run the planned jobs without sending HTTP requests.
    uv run python -B scripts/debug/pressure_lot_hwm_requests.py --max-requests 100 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import random
import resource
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Literal
from urllib.parse import quote

import httpx
import polars as pl
import psycopg
import requests
import yaml

from wip_analysis.data_loader.sql_template import render_sql

WindowMode = Literal["exact", "cumulative"]
SnapshotSelection = Literal["spread", "latest", "earliest", "random"]
TransportMode = Literal["async", "threads"]

_THREAD_LOCAL = threading.local()
_JSONL_SENTINEL = object()


@dataclass(frozen=True)
class SnapshotRow:
    lot_id: str
    product_id: str
    created_at: datetime


@dataclass(frozen=True)
class LotHwmJob:
    lot_id: str
    product_id: str
    created_at: datetime
    snapshot_start: datetime | None
    snapshot_end: datetime


@dataclass(frozen=True)
class RequestResult:
    job: LotHwmJob
    ok: bool
    status_code: int | None
    elapsed_s: float
    response_bytes: int
    error: str | None
    body_preview: str | None


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    return datetime.fromisoformat(normalized)


def _load_db_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return dict(config["database"])


def _fetch_direct(query: str, config_path: str) -> pl.DataFrame:
    db_cfg = _load_db_config(config_path)
    conninfo = (
        f"postgresql://{db_cfg['user']}:{db_cfg['password']}"
        f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['dbname']}"
    )
    with psycopg.connect(conninfo) as conn:
        return pl.read_database(query=query, connection=conn)


def fetch_snapshot_rows(
    *,
    sql_config: str,
    db_config: str,
    product_ids: list[str] | None,
    lot_ids: list[str] | None,
    snapshot_start: datetime | None,
    snapshot_end: datetime | None,
    query_limit: int | None,
    show_sql: bool,
) -> list[SnapshotRow]:
    query = render_sql(
        "get_lot_hwm_pressure_jobs.sql.j2",
        {
            "product_ids": product_ids,
            "lot_ids": lot_ids,
            "snapshot_start": snapshot_start,
            "snapshot_end": snapshot_end,
            "query_limit": query_limit,
        },
        config_path=sql_config,
    )
    if show_sql:
        print("===== SQL =====", file=sys.stderr)
        print(query, file=sys.stderr)
        print("===============", file=sys.stderr)

    df = _fetch_direct(query, db_config)
    rows: list[SnapshotRow] = []
    for row in df.iter_rows(named=True):
        created_at = row["created_at"]
        if not isinstance(created_at, datetime):
            created_at = datetime.fromisoformat(str(created_at))
        rows.append(
            SnapshotRow(
                lot_id=str(row["lot_id"]),
                product_id=str(row["product_id"]),
                created_at=created_at,
            )
        )
    return rows


def _select_indices(
    total: int,
    count: int,
    *,
    selection: SnapshotSelection,
    rng: random.Random,
) -> list[int]:
    if count >= total:
        return list(range(total))
    if count <= 0:
        return []
    if selection == "earliest":
        return list(range(count))
    if selection == "latest":
        return list(range(total - count, total))
    if selection == "random":
        return sorted(rng.sample(range(total), count))

    # Pick the center of count equal-width buckets so limited samples are spread
    # across each lot's observed timeline.
    return sorted({min(total - 1, int((idx + 0.5) * total / count)) for idx in range(count)})


def _allocate_even_counts(
    capacities: dict[str, int],
    target_count: int,
    lot_order: list[str],
) -> dict[str, int]:
    counts = {lot_id: 0 for lot_id in lot_order}
    remaining = min(target_count, sum(capacities.values()))
    active = [lot_id for lot_id in lot_order if capacities[lot_id] > 0]

    while remaining > 0 and active:
        next_active: list[str] = []
        for lot_id in active:
            if remaining <= 0:
                next_active.append(lot_id)
                continue
            counts[lot_id] += 1
            remaining -= 1
            if counts[lot_id] < capacities[lot_id]:
                next_active.append(lot_id)
        active = next_active

    return counts


def _round_robin_jobs(
    selected_by_lot: dict[str, list[SnapshotRow]], lot_order: list[str]
) -> list[SnapshotRow]:
    positions = {lot_id: 0 for lot_id in lot_order}
    jobs: list[SnapshotRow] = []
    while True:
        progressed = False
        for lot_id in lot_order:
            rows = selected_by_lot.get(lot_id) or []
            pos = positions[lot_id]
            if pos >= len(rows):
                continue
            jobs.append(rows[pos])
            positions[lot_id] = pos + 1
            progressed = True
        if not progressed:
            return jobs


def select_snapshot_rows(
    rows: Iterable[SnapshotRow],
    *,
    max_requests: int | None,
    selection: SnapshotSelection,
    sample_seed: int,
    shuffle_lots: bool,
) -> list[SnapshotRow]:
    grouped: dict[str, list[SnapshotRow]] = defaultdict(list)
    for row in rows:
        grouped[row.lot_id].append(row)
    for lot_rows in grouped.values():
        lot_rows.sort(key=lambda row: row.created_at)

    lot_order = sorted(grouped)
    rng = random.Random(sample_seed)
    if shuffle_lots:
        rng.shuffle(lot_order)

    if max_requests is None:
        return _round_robin_jobs(grouped, lot_order)

    capacities = {lot_id: len(grouped[lot_id]) for lot_id in lot_order}
    counts = _allocate_even_counts(capacities, max_requests, lot_order)
    selected_by_lot: dict[str, list[SnapshotRow]] = {}
    for lot_id in lot_order:
        lot_rows = grouped[lot_id]
        indices = _select_indices(
            len(lot_rows),
            counts[lot_id],
            selection=selection,
            rng=rng,
        )
        selected_by_lot[lot_id] = [lot_rows[idx] for idx in indices]
    return _round_robin_jobs(selected_by_lot, lot_order)


def build_jobs(
    rows: Iterable[SnapshotRow],
    *,
    window_mode: WindowMode,
    exact_window_us: int,
) -> list[LotHwmJob]:
    jobs: list[LotHwmJob] = []
    exact_delta = timedelta(microseconds=exact_window_us)
    for row in rows:
        if window_mode == "exact":
            snapshot_start = row.created_at
            snapshot_end = row.created_at + exact_delta
        else:
            snapshot_start = None
            snapshot_end = row.created_at
        jobs.append(
            LotHwmJob(
                lot_id=row.lot_id,
                product_id=row.product_id,
                created_at=row.created_at,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
            )
        )
    return jobs


def build_lot_url(base_url: str, lot_id: str) -> str:
    root = base_url.rstrip("/")
    return f"{root}/api/v1/lots/{quote(lot_id, safe='')}"


def _thread_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_LOCAL.session = session
    return session


def _payload_for_job(job: LotHwmJob, *, refresh: bool, send_product_id: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "refresh": refresh,
        "snapshot_end": job.snapshot_end.isoformat(),
    }
    if job.snapshot_start is not None:
        payload["snapshot_start"] = job.snapshot_start.isoformat()
    if send_product_id:
        payload["product_id"] = job.product_id
    return payload


def post_lot_hwm(
    job: LotHwmJob,
    *,
    base_url: str,
    timeout_s: float,
    refresh: bool,
    send_product_id: bool,
) -> RequestResult:
    url = build_lot_url(base_url, job.lot_id)
    payload = _payload_for_job(job, refresh=refresh, send_product_id=send_product_id)
    started = time.perf_counter()
    try:
        resp = _thread_session().post(url, json=payload, timeout=timeout_s)
        elapsed_s = time.perf_counter() - started
        body_preview = None if resp.ok else (resp.text or "")[:500]
        return RequestResult(
            job=job,
            ok=resp.ok,
            status_code=resp.status_code,
            elapsed_s=elapsed_s,
            response_bytes=len(resp.content or b""),
            error=None,
            body_preview=body_preview,
        )
    except Exception as exc:
        elapsed_s = time.perf_counter() - started
        return RequestResult(
            job=job,
            ok=False,
            status_code=None,
            elapsed_s=elapsed_s,
            response_bytes=0,
            error=repr(exc),
            body_preview=None,
        )


async def post_lot_hwm_async(
    client: httpx.AsyncClient,
    job: LotHwmJob,
    *,
    base_url: str,
    refresh: bool,
    send_product_id: bool,
) -> RequestResult:
    url = build_lot_url(base_url, job.lot_id)
    payload = _payload_for_job(job, refresh=refresh, send_product_id=send_product_id)
    started = time.perf_counter()
    try:
        resp = await client.post(url, json=payload)
        elapsed_s = time.perf_counter() - started
        body_preview = None if resp.is_success else (resp.text or "")[:500]
        return RequestResult(
            job=job,
            ok=resp.is_success,
            status_code=resp.status_code,
            elapsed_s=elapsed_s,
            response_bytes=len(resp.content or b""),
            error=None,
            body_preview=body_preview,
        )
    except Exception as exc:
        elapsed_s = time.perf_counter() - started
        return RequestResult(
            job=job,
            ok=False,
            status_code=None,
            elapsed_s=elapsed_s,
            response_bytes=0,
            error=repr(exc),
            body_preview=None,
        )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _result_payload(result: RequestResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["job"]["created_at"] = result.job.created_at.isoformat()
    payload["job"]["snapshot_start"] = (
        result.job.snapshot_start.isoformat() if result.job.snapshot_start else None
    )
    payload["job"]["snapshot_end"] = result.job.snapshot_end.isoformat()
    return payload


def _result_json_line(result: RequestResult) -> str:
    return json.dumps(_result_payload(result), ensure_ascii=False) + "\n"


def _soft_open_file_limit() -> int | None:
    soft_limit, _hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit == resource.RLIM_INFINITY:
        return None
    return int(soft_limit)


def _resolve_worker_count(
    *,
    requested_workers: int,
    total_jobs: int,
    fd_safety_margin: int,
    unsafe_fd_concurrency: bool,
    transport: TransportMode,
) -> int:
    worker_count = min(requested_workers, total_jobs)
    soft_limit = _soft_open_file_limit()
    if soft_limit is None:
        return worker_count

    fd_safe_workers = max(1, soft_limit - fd_safety_margin)
    if worker_count <= fd_safe_workers:
        return worker_count

    message = (
        f"[warn] requested {worker_count} concurrent {transport} requests, "
        f"but soft open-file limit is {soft_limit} and --fd-safety-margin is "
        f"{fd_safety_margin}."
    )
    if unsafe_fd_concurrency:
        print(f"{message} Proceeding because --unsafe-fd-concurrency is set.", file=sys.stderr)
        return worker_count

    print(f"{message} Capping concurrency to {fd_safe_workers}.", file=sys.stderr)
    return fd_safe_workers


class ResultJsonlWriter:
    """Single-FD JSONL writer decoupled from request worker concurrency."""

    def __init__(
        self,
        path: Path,
        *,
        max_queue_size: int,
        flush_every: int,
    ) -> None:
        self.path = path
        self.flush_every = flush_every
        self._queue: queue.Queue[RequestResult | object] = queue.Queue(maxsize=max_queue_size)
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def __enter__(self) -> "ResultJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            name="lot-hwm-jsonl-writer",
            daemon=False,
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._raise_if_failed()
        if self._thread is not None and self._thread.is_alive():
            self._put(_JSONL_SENTINEL)
            self._thread.join()
        self._raise_if_failed()

    def write(self, result: RequestResult) -> None:
        self._raise_if_failed()
        self._put(result)
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"JSONL writer failed for {self.path}") from self._error

    def _put(self, item: RequestResult | object) -> None:
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _run(self) -> None:
        pending_flush = 0
        try:
            with self.path.open("w", encoding="utf-8") as f:
                while True:
                    item = self._queue.get()
                    try:
                        if item is _JSONL_SENTINEL:
                            if pending_flush:
                                f.flush()
                            return
                        assert isinstance(item, RequestResult)
                        f.write(_result_json_line(item))
                        pending_flush += 1
                        if self.flush_every > 0 and pending_flush >= self.flush_every:
                            f.flush()
                            pending_flush = 0
                    finally:
                        self._queue.task_done()
        except BaseException as exc:  # pragma: no cover - exercised via integration failures
            self._error = exc


@dataclass
class RunStats:
    total_jobs: int
    progress_every: int
    result_writer: ResultJsonlWriter | None
    started: float = field(default_factory=time.perf_counter)
    completed: int = 0
    ok_count: int = 0
    fail_count: int = 0
    latencies: list[float] = field(default_factory=list)
    status_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, result: RequestResult) -> None:
        self.completed += 1
        self.latencies.append(result.elapsed_s)
        status_key = str(result.status_code) if result.status_code is not None else "EXC"
        self.status_counts[status_key] += 1
        if result.ok:
            self.ok_count += 1
        else:
            self.fail_count += 1
            print(
                f"[FAIL] status={status_key} lot={result.job.lot_id!r} "
                f"created_at={result.job.created_at.isoformat()} "
                f"elapsed={result.elapsed_s:.2f}s err={result.error or result.body_preview}",
                file=sys.stderr,
            )

        if self.result_writer is not None:
            self.result_writer.write(result)

        if self.progress_every > 0 and self.completed % self.progress_every == 0:
            elapsed = time.perf_counter() - self.started
            rate = self.completed / elapsed if elapsed > 0 else 0.0
            print(
                f"[progress] {self.completed}/{self.total_jobs} "
                f"ok={self.ok_count} fail={self.fail_count} "
                f"rate={rate:.2f}/s p95={_percentile(self.latencies, 0.95):.2f}s",
                file=sys.stderr,
            )

    def finish(self) -> int:
        elapsed = time.perf_counter() - self.started
        print("===== Summary =====")
        print(
            f"jobs={self.total_jobs} ok={self.ok_count} "
            f"fail={self.fail_count} elapsed={elapsed:.2f}s"
        )
        print(f"throughput={self.total_jobs / elapsed:.2f}/s" if elapsed > 0 else "throughput=n/a")
        print(f"statuses={dict(sorted(self.status_counts.items()))}")
        print(
            "latency_s "
            f"min={min(self.latencies):.3f} "
            f"avg={statistics.fmean(self.latencies):.3f} "
            f"p50={_percentile(self.latencies, 0.50):.3f} "
            f"p95={_percentile(self.latencies, 0.95):.3f} "
            f"p99={_percentile(self.latencies, 0.99):.3f} "
            f"max={max(self.latencies):.3f}"
        )
        return 0 if self.fail_count == 0 else 1


def run_parallel(
    jobs: list[LotHwmJob],
    *,
    base_url: str,
    max_workers: int,
    max_in_flight: int | None,
    fd_safety_margin: int,
    unsafe_fd_concurrency: bool,
    timeout_s: float,
    refresh: bool,
    send_product_id: bool,
    progress_every: int,
    results_jsonl: Path | None,
    results_jsonl_queue_size: int,
    results_jsonl_flush_every: int,
) -> int:
    if not jobs:
        print("No jobs to run.", file=sys.stderr)
        return 1

    worker_count = _resolve_worker_count(
        requested_workers=max_workers,
        total_jobs=len(jobs),
        fd_safety_margin=fd_safety_margin,
        unsafe_fd_concurrency=unsafe_fd_concurrency,
        transport="threads",
    )

    writer_context = (
        ResultJsonlWriter(
            results_jsonl,
            max_queue_size=results_jsonl_queue_size,
            flush_every=results_jsonl_flush_every,
        )
        if results_jsonl is not None
        else nullcontext(None)
    )

    with writer_context as result_writer:
        stats = RunStats(
            total_jobs=len(jobs),
            progress_every=progress_every,
            result_writer=result_writer,
        )

        next_job_idx = 0
        requested_in_flight = max_in_flight if max_in_flight is not None else worker_count * 4
        in_flight_limit = min(len(jobs), max(1, requested_in_flight))
        pool_workers = min(worker_count, in_flight_limit)

        with ThreadPoolExecutor(max_workers=pool_workers) as pool:
            in_flight = set()

            def submit_next() -> None:
                nonlocal next_job_idx
                job = jobs[next_job_idx]
                next_job_idx += 1
                in_flight.add(
                    pool.submit(
                        post_lot_hwm,
                        job,
                        base_url=base_url,
                        timeout_s=timeout_s,
                        refresh=refresh,
                        send_product_id=send_product_id,
                    )
                )

            while next_job_idx < len(jobs) and len(in_flight) < in_flight_limit:
                submit_next()

            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    stats.record(future.result())

                while next_job_idx < len(jobs) and len(in_flight) < in_flight_limit:
                    submit_next()

        return stats.finish()


async def run_async_parallel(
    jobs: list[LotHwmJob],
    *,
    base_url: str,
    max_workers: int,
    fd_safety_margin: int,
    unsafe_fd_concurrency: bool,
    timeout_s: float,
    refresh: bool,
    send_product_id: bool,
    progress_every: int,
    results_jsonl: Path | None,
    results_jsonl_queue_size: int,
    results_jsonl_flush_every: int,
) -> int:
    if not jobs:
        print("No jobs to run.", file=sys.stderr)
        return 1

    worker_count = _resolve_worker_count(
        requested_workers=max_workers,
        total_jobs=len(jobs),
        fd_safety_margin=fd_safety_margin,
        unsafe_fd_concurrency=unsafe_fd_concurrency,
        transport="async",
    )

    writer_context = (
        ResultJsonlWriter(
            results_jsonl,
            max_queue_size=results_jsonl_queue_size,
            flush_every=results_jsonl_flush_every,
        )
        if results_jsonl is not None
        else nullcontext(None)
    )

    with writer_context as result_writer:
        stats = RunStats(
            total_jobs=len(jobs),
            progress_every=progress_every,
            result_writer=result_writer,
        )
        job_iter = iter(jobs)

        limits = httpx.Limits(
            max_connections=worker_count,
            max_keepalive_connections=min(worker_count, 1024),
        )
        timeout = httpx.Timeout(timeout_s)
        async with httpx.AsyncClient(limits=limits, timeout=timeout, trust_env=False) as client:

            async def worker() -> None:
                while True:
                    try:
                        job = next(job_iter)
                    except StopIteration:
                        return
                    result = await post_lot_hwm_async(
                        client,
                        job,
                        base_url=base_url,
                        refresh=refresh,
                        send_product_id=send_product_id,
                    )
                    stats.record(result)

            tasks = [asyncio.create_task(worker()) for _ in range(worker_count)]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        return stats.finish()


def _print_plan(
    jobs: list[LotHwmJob],
    preview_count: int,
    *,
    refresh: bool,
    send_product_id: bool,
) -> None:
    lots = {job.lot_id for job in jobs}
    print(f"planned_jobs={len(jobs)} distinct_lots={len(lots)}")
    for job in jobs[:preview_count]:
        payload = _payload_for_job(job, refresh=refresh, send_product_id=send_product_id)
        print(
            json.dumps(
                {
                    "lot_id": job.lot_id,
                    "product_id": job.product_id,
                    "created_at": job.created_at.isoformat(),
                    "payload": payload,
                },
                ensure_ascii=False,
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate DB-backed jobs and concurrently POST /api/v1/lots/{lot_id}.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8724", help="API root URL.")
    parser.add_argument(
        "--transport",
        choices=["async", "threads"],
        default="async",
        help="HTTP transport. async scales to high concurrency without one OS thread per request.",
    )
    parser.add_argument(
        "-n",
        "--max-workers",
        type=int,
        default=8,
        help="Maximum concurrent in-flight HTTP requests.",
    )
    parser.add_argument(
        "--fd-safety-margin",
        type=int,
        default=128,
        help=(
            "Reserve this many file descriptors when auto-capping concurrency "
            "against the process soft open-file limit."
        ),
    )
    parser.add_argument(
        "--unsafe-fd-concurrency",
        action="store_true",
        help="Disable open-file-limit based concurrency capping.",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="Thread transport only: maximum submitted futures. Defaults to 4x --max-workers.",
    )
    parser.add_argument(
        "--timeout", type=float, default=10800.0, help="Per-request timeout seconds."
    )
    parser.add_argument("--sql-config", default="config/sql_base.yaml")
    parser.add_argument("--db-config", default="config/postgres.yaml")
    parser.add_argument("--product-id", action="append", dest="product_ids")
    parser.add_argument("--lot-id", action="append", dest="lot_ids")
    parser.add_argument("--snapshot-start", type=_parse_datetime)
    parser.add_argument("--snapshot-end", type=_parse_datetime)
    parser.add_argument(
        "--query-limit", type=int, default=None, help="Optional SQL-side row limit."
    )
    parser.add_argument("--show-sql", action="store_true")

    request_scope = parser.add_mutually_exclusive_group(required=True)
    request_scope.add_argument("--all", action="store_true", help="Run all fetched rows.")
    request_scope.add_argument(
        "--max-requests",
        type=int,
        help="Maximum planned requests, sampled evenly across lots.",
    )

    parser.add_argument(
        "--snapshot-selection",
        choices=["spread", "latest", "earliest", "random"],
        default="spread",
        help="How to choose snapshots within each lot when --max-requests is used.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--no-shuffle-lots",
        action="store_true",
        help="Keep lot IDs sorted instead of seeded shuffle before even allocation.",
    )
    parser.add_argument(
        "--window-mode",
        choices=["exact", "cumulative"],
        default="exact",
        help="exact sends [created_at, created_at+epsilon]; cumulative sends only snapshot_end.",
    )
    parser.add_argument(
        "--exact-window-us",
        type=int,
        default=1,
        help="Microseconds added to snapshot_end in exact window mode.",
    )
    parser.add_argument("--refresh", action="store_true", help="Send refresh=true in each request.")
    parser.add_argument(
        "--send-product-id",
        action="store_true",
        help="Include product_id in each payload as a guard.",
    )
    parser.add_argument("--shuffle-jobs", action="store_true", help="Shuffle final job order.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Build and print jobs, do not request API."
    )
    parser.add_argument("--preview", type=int, default=10, help="Jobs to print in dry-run preview.")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--results-jsonl", type=Path, default=None)
    parser.add_argument(
        "--results-jsonl-queue-size",
        type=int,
        default=10000,
        help="Maximum queued completed results waiting for the single JSONL writer.",
    )
    parser.add_argument(
        "--results-jsonl-flush-every",
        type=int,
        default=1,
        help="Flush JSONL output after this many records; use 0 to flush only on close.",
    )

    args = parser.parse_args()
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    if args.fd_safety_margin < 0:
        raise ValueError("--fd-safety-margin must be non-negative")
    if args.max_in_flight is not None and args.max_in_flight <= 0:
        raise ValueError("--max-in-flight must be positive")
    if args.max_requests is not None and args.max_requests <= 0:
        raise ValueError("--max-requests must be positive")
    if args.exact_window_us <= 0:
        raise ValueError("--exact-window-us must be positive")
    if args.results_jsonl_queue_size < 0:
        raise ValueError("--results-jsonl-queue-size must be non-negative")
    if args.results_jsonl_flush_every < 0:
        raise ValueError("--results-jsonl-flush-every must be non-negative")

    rows = fetch_snapshot_rows(
        sql_config=args.sql_config,
        db_config=args.db_config,
        product_ids=args.product_ids,
        lot_ids=args.lot_ids,
        snapshot_start=args.snapshot_start,
        snapshot_end=args.snapshot_end,
        query_limit=args.query_limit,
        show_sql=args.show_sql,
    )
    selected_rows = select_snapshot_rows(
        rows,
        max_requests=None if args.all else args.max_requests,
        selection=args.snapshot_selection,
        sample_seed=args.sample_seed,
        shuffle_lots=not args.no_shuffle_lots,
    )
    jobs = build_jobs(
        selected_rows,
        window_mode=args.window_mode,
        exact_window_us=args.exact_window_us,
    )
    if args.shuffle_jobs:
        random.Random(args.sample_seed).shuffle(jobs)

    _print_plan(
        jobs,
        args.preview if args.dry_run else 0,
        refresh=args.refresh,
        send_product_id=args.send_product_id,
    )
    if args.dry_run:
        return 0

    run_kwargs = {
        "base_url": args.base_url,
        "max_workers": args.max_workers,
        "fd_safety_margin": args.fd_safety_margin,
        "unsafe_fd_concurrency": args.unsafe_fd_concurrency,
        "timeout_s": args.timeout,
        "refresh": args.refresh,
        "send_product_id": args.send_product_id,
        "progress_every": args.progress_every,
        "results_jsonl": args.results_jsonl,
        "results_jsonl_queue_size": args.results_jsonl_queue_size,
        "results_jsonl_flush_every": args.results_jsonl_flush_every,
    }
    if args.transport == "threads":
        return run_parallel(jobs, max_in_flight=args.max_in_flight, **run_kwargs)

    if args.max_in_flight is not None:
        print("[warn] --max-in-flight is ignored by async transport.", file=sys.stderr)
    return asyncio.run(run_async_parallel(jobs, **run_kwargs))


if __name__ == "__main__":
    raise SystemExit(main())
