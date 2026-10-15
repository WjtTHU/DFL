import hashlib
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from threading import Lock, Thread
from typing import Optional, TypedDict

from loguru import logger

from wip_analysis.data_loader.postgres_loader import fetch_data, wait_for_async_writes
from wip_analysis.data_loader.sql_template import render_sql
from wip_analysis.hwm.stage.utils.lot_completion_predictions import (
    compute_lot_completion_predictions,
)
from wip_analysis.utils.env_num import env_num


class LatestSnapshotCut(TypedDict):
    lot_id: str
    snapshot_cut: datetime


RecalcScope = tuple[str | None, str | None, str | None, str | None]

_INFLIGHT_RECALC_LOCK = Lock()
_INFLIGHT_RECALC_SCOPES: set[RecalcScope] = set()
_RECALC_FAILURES: dict[RecalcScope, tuple[str, ...]] = {}
_DEFAULT_RECALC_WORKERS = 8


def _recalc_worker_count(task_count: int) -> int:
    return min(
        task_count, env_num("WIP_ANALYSIS_LOT_PROGRESS_RECALC_WORKERS", _DEFAULT_RECALC_WORKERS)
    )


def _scope_key(
    product_id: Optional[str],
    lot_id: Optional[str],
    snapshot_start: Optional[datetime],
    snapshot_end: Optional[datetime],
) -> RecalcScope:
    return (
        product_id,
        lot_id,
        snapshot_start.isoformat() if snapshot_start is not None else None,
        snapshot_end.isoformat() if snapshot_end is not None else None,
    )


def _target_days_scope_key(
    product_id: str,
    marker: str,
    target_days: list[date],
) -> RecalcScope:
    target_days_key = ",".join(day.isoformat() for day in sorted(set(target_days)))
    return (product_id, marker, target_days_key, None)


def _target_day_scope_key(product_id: str, marker: str, target_day: date) -> RecalcScope:
    return (product_id, marker, target_day.isoformat(), None)


def _snapshot_cuts_scope_key(
    product_id: str,
    marker: str,
    snapshot_cuts: list[LatestSnapshotCut],
) -> RecalcScope:
    parts = [
        f"{row['lot_id']}\0{row['snapshot_cut'].isoformat()}"
        for row in sorted(
            snapshot_cuts,
            key=lambda row: (str(row["lot_id"]), row["snapshot_cut"].isoformat()),
        )
    ]
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()
    return (product_id, marker, digest, None)


def _dedupe_snapshot_cuts(snapshot_cuts: list[LatestSnapshotCut]) -> list[LatestSnapshotCut]:
    by_key: dict[tuple[str, datetime], LatestSnapshotCut] = {}
    for row in snapshot_cuts:
        lot_id = str(row["lot_id"])
        snapshot_cut = row["snapshot_cut"]
        if not isinstance(snapshot_cut, datetime):
            continue
        by_key[(lot_id, snapshot_cut)] = {"lot_id": lot_id, "snapshot_cut": snapshot_cut}
    return sorted(by_key.values(), key=lambda row: (row["lot_id"], row["snapshot_cut"]))


def _any_scope_running(scopes: list[RecalcScope]) -> bool:
    with _INFLIGHT_RECALC_LOCK:
        return any(scope in _INFLIGHT_RECALC_SCOPES for scope in scopes)


def _scope_failures(scopes: list[RecalcScope]) -> tuple[str, ...]:
    with _INFLIGHT_RECALC_LOCK:
        failures: list[str] = []
        for scope in scopes:
            failures.extend(_RECALC_FAILURES.get(scope, ()))
        return tuple(failures)


def _clear_scope_failures(scope: RecalcScope) -> None:
    _RECALC_FAILURES.pop(scope, None)


def _finish_recalc_scope(
    scope: RecalcScope,
    failures: list[str] | None = None,
    *,
    drain_async_writes: bool = True,
) -> None:
    failures = list(failures or [])
    if drain_async_writes and not wait_for_async_writes():
        failures.append("Timed out waiting for async DB writes to drain")

    with _INFLIGHT_RECALC_LOCK:
        _INFLIGHT_RECALC_SCOPES.discard(scope)
        if failures:
            _RECALC_FAILURES[scope] = tuple(failures)
        else:
            _RECALC_FAILURES.pop(scope, None)


def _build_missing_sample(
    missing_latest_snapshots: list[LatestSnapshotCut],
) -> list[dict[str, str]]:
    return [
        {
            "lot_id": str(row["lot_id"]),
            "snapshot_cut": row["snapshot_cut"].isoformat(),
        }
        for row in missing_latest_snapshots[:5]
    ]


def _raise_if_recalc_finished_but_missing(
    *,
    scopes: list[RecalcScope],
    context: str,
    missing_count: int,
    missing_latest_snapshots: list[LatestSnapshotCut],
    waited_after_trigger: bool,
    detail: str,
) -> None:
    if not waited_after_trigger:
        return

    failures = _scope_failures(scopes)
    if failures:
        raise RuntimeError(
            f"Background readiness recalculation failed for {context} "
            f"({detail}, missing_latest_lots={missing_count}, "
            f"sample={_build_missing_sample(missing_latest_snapshots)}, "
            f"errors={list(failures[:5])})."
        )

    if not _any_scope_running(scopes):
        raise RuntimeError(
            f"Background readiness recalculation finished but required rows are still missing "
            f"for {context} ({detail}, missing_latest_lots={missing_count}, "
            f"sample={_build_missing_sample(missing_latest_snapshots)})."
        )


def _query_scope_snapshot_times(
    product_id: Optional[str],
    lot_id: Optional[str],
    snapshot_start: Optional[datetime],
    snapshot_end: Optional[datetime],
) -> list[datetime]:
    q = render_sql(
        "get_scope_snapshot_times.sql.j2",
        {
            "product_id": product_id,
            "lot_id": lot_id,
            "snapshot_start": snapshot_start,
            "snapshot_end": snapshot_end,
        },
    )
    df = fetch_data(q)
    if df.is_empty():
        return []
    return [row["snapshot_time"] for row in df.iter_rows(named=True)]


def _query_scope_progress_cuts(
    product_id: Optional[str],
    lot_id: Optional[str],
    snapshot_start: Optional[datetime],
    snapshot_end: Optional[datetime],
) -> list[datetime]:
    q = render_sql(
        "get_scope_lot_progress_cuts.sql.j2",
        {
            "product_id": product_id,
            "lot_id": lot_id,
            "snapshot_start": snapshot_start,
            "snapshot_end": snapshot_end,
        },
    )
    df = fetch_data(q)
    if df.is_empty():
        return []
    return [row["snapshot_time"] for row in df.iter_rows(named=True)]


def _query_scope_lot_ids(
    product_id: Optional[str],
    lot_id: Optional[str],
    snapshot_start: Optional[datetime],
    snapshot_end: Optional[datetime],
) -> list[str]:
    if lot_id is not None:
        return [lot_id]

    q = render_sql(
        "get_scope_lot_ids.sql.j2",
        {
            "product_id": product_id,
            "lot_id": lot_id,
            "snapshot_start": snapshot_start,
            "snapshot_end": snapshot_end,
        },
    )
    df = fetch_data(q)
    if df.is_empty():
        return []
    return [row["lot_id"] for row in df.iter_rows(named=True)]


def _query_scope_latest_lot_progress_readiness(
    product_id: Optional[str],
    lot_id: Optional[str],
    snapshot_start: Optional[datetime],
    snapshot_end: Optional[datetime],
) -> tuple[int, list[LatestSnapshotCut]]:
    _ = snapshot_start
    q = render_sql(
        "get_scope_latest_lot_progress_readiness.sql.j2",
        {
            "product_id": product_id,
            "lot_id": lot_id,
            "snapshot_end": snapshot_end,
        },
    )
    df = fetch_data(q)
    if df.is_empty():
        return 0, []

    missing_df = df.filter(~df["is_ready"])
    missing_latest_snapshots: list[LatestSnapshotCut] = [
        {"lot_id": row["lot_id"], "snapshot_cut": row["snapshot_cut"]}
        for row in missing_df.iter_rows(named=True)
    ]
    return df.height, missing_latest_snapshots


def _query_lot_progress_snapshot_cuts_readiness(
    product_id: str,
    snapshot_cuts: list[LatestSnapshotCut],
) -> tuple[int, list[LatestSnapshotCut]]:
    snapshot_cuts = _dedupe_snapshot_cuts(snapshot_cuts)
    if not snapshot_cuts:
        return 0, []

    q = render_sql(
        "get_lot_progress_snapshot_cuts_readiness.sql.j2",
        {
            "product_id": product_id,
            "snapshot_cuts": snapshot_cuts,
        },
    )
    df = fetch_data(q)
    if df.is_empty():
        return 0, snapshot_cuts

    missing_df = df.filter(~df["is_ready"])
    missing_latest_snapshots: list[LatestSnapshotCut] = [
        {"lot_id": row["lot_id"], "snapshot_cut": row["snapshot_cut"]}
        for row in missing_df.iter_rows(named=True)
    ]
    return df.height, missing_latest_snapshots


def _run_lot_progress_recalc(
    lot_ids: list[str],
    snapshot_start: Optional[datetime],
    snapshot_end: Optional[datetime],
) -> list[str]:
    from wip_analysis.hwm.lot_level import get_lot_prog_over_snapshots

    failures: list[str] = []
    failures_lock = Lock()

    def _run_one(lid: str) -> None:
        try:
            get_lot_prog_over_snapshots(
                lid,
                refresh=False,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
            )
        except Exception as exc:
            logger.warning("Failed background lot progress calc for lot_id={}: {}", lid, exc)
            with failures_lock:
                failures.append(f"lot_id={lid}: {exc}")

    if not lot_ids:
        return []
    worker_count = _recalc_worker_count(len(lot_ids))
    if worker_count <= 1:
        for lid in lot_ids:
            _run_one(lid)
        return failures

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_run_one, str(lid)) for lid in lot_ids]
        for future in as_completed(futures):
            future.result()
    return failures


def _run_lot_progress_recalc_and_clear_scope(
    scope: RecalcScope,
    lot_ids: list[str],
    snapshot_start: Optional[datetime],
    snapshot_end: Optional[datetime],
) -> None:
    failures: list[str] = []
    try:
        failures = _run_lot_progress_recalc(lot_ids, snapshot_start, snapshot_end)
    finally:
        _finish_recalc_scope(scope, failures, drain_async_writes=True)


def _run_latest_lot_progress_recalc_and_clear_scope(
    scope: RecalcScope,
    missing_latest_snapshots: list[LatestSnapshotCut],
) -> None:
    from wip_analysis.hwm.lot_level import get_lot_prog_over_snapshots

    failures: list[str] = []
    failures_lock = Lock()

    def _run_one(row: LatestSnapshotCut) -> None:
        lid = str(row["lot_id"])
        snapshot_cut = row["snapshot_cut"]
        try:
            get_lot_prog_over_snapshots(
                lid,
                refresh=False,
                snapshot_start=snapshot_cut,
                snapshot_end=snapshot_cut,
            )
        except Exception as exc:
            logger.warning(
                "Failed background latest lot progress calc for lot_id={} snapshot_cut={}: {}",
                lid,
                snapshot_cut,
                exc,
            )
            with failures_lock:
                failures.append(f"lot_id={lid} snapshot_cut={snapshot_cut.isoformat()}: {exc}")

    try:
        if not missing_latest_snapshots:
            return
        worker_count = _recalc_worker_count(len(missing_latest_snapshots))
        if worker_count <= 1:
            for row in missing_latest_snapshots:
                _run_one(row)
            return

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_run_one, row) for row in missing_latest_snapshots]
            for future in as_completed(futures):
                future.result()
    finally:
        _finish_recalc_scope(scope, failures, drain_async_writes=True)


def _trigger_lot_progress_recalc_async(
    product_id: Optional[str],
    lot_id: Optional[str],
    snapshot_start: Optional[datetime],
    snapshot_end: Optional[datetime],
) -> tuple[int, bool]:
    scope = _scope_key(product_id, lot_id, snapshot_start, snapshot_end)
    with _INFLIGHT_RECALC_LOCK:
        already_running = scope in _INFLIGHT_RECALC_SCOPES
        if already_running:
            return 0, True
        _INFLIGHT_RECALC_SCOPES.add(scope)
        _clear_scope_failures(scope)

    lot_ids = _query_scope_lot_ids(product_id, lot_id, snapshot_start, snapshot_end)
    if not lot_ids:
        with _INFLIGHT_RECALC_LOCK:
            _INFLIGHT_RECALC_SCOPES.discard(scope)
        return 0, False

    t = Thread(
        target=_run_lot_progress_recalc_and_clear_scope,
        args=(scope, lot_ids, snapshot_start, snapshot_end),
        daemon=True,
    )
    t.start()
    return len(lot_ids), False


def _trigger_snapshot_cuts_lot_progress_recalc_async(
    product_id: str,
    missing_latest_snapshots: list[LatestSnapshotCut],
    scope: RecalcScope,
) -> tuple[int, bool]:
    _ = product_id
    with _INFLIGHT_RECALC_LOCK:
        already_running = scope in _INFLIGHT_RECALC_SCOPES
        if already_running:
            return 0, True
        _INFLIGHT_RECALC_SCOPES.add(scope)
        _clear_scope_failures(scope)

    if not missing_latest_snapshots:
        _finish_recalc_scope(scope, drain_async_writes=False)
        return 0, False

    t = Thread(
        target=_run_latest_lot_progress_recalc_and_clear_scope,
        args=(scope, missing_latest_snapshots),
        daemon=True,
    )
    t.start()
    return len(missing_latest_snapshots), False


def _query_product_date_out_lot_progress_readiness(
    product_id: str,
    target_days: list[date],
) -> tuple[int, list[LatestSnapshotCut]]:
    q = render_sql(
        "get_pid_date_out_lot_progress_readiness.sql.j2",
        {"product_id": product_id, "target_days": target_days},
    )
    df = fetch_data(q)
    if df.is_empty():
        return 0, []

    missing_df = df.filter(~df["is_ready"])
    missing_latest_snapshots: list[LatestSnapshotCut] = [
        {"lot_id": row["lot_id"], "snapshot_cut": row["snapshot_cut"]}
        for row in missing_df.iter_rows(named=True)
    ]
    return df.height, missing_latest_snapshots


def _query_product_date_out_prediction_readiness(
    product_id: str,
    target_days: list[date],
) -> tuple[int, dict[date, list[LatestSnapshotCut]]]:
    q = render_sql(
        "get_pid_date_out_prediction_readiness.sql.j2",
        {"product_id": product_id, "target_days": target_days},
    )
    df = fetch_data(q)
    if df.is_empty():
        return 0, {}

    missing_df = df.filter(~df["is_ready"])
    missing_by_target_day: dict[date, list[LatestSnapshotCut]] = defaultdict(list)
    for row in missing_df.iter_rows(named=True):
        target_day = row["target_day"]
        if isinstance(target_day, datetime):
            target_day = target_day.date()
        missing_by_target_day[target_day].append(
            {"lot_id": row["lot_id"], "snapshot_cut": row["snapshot_cut"]}
        )
    return df.height, dict(missing_by_target_day)


def _run_latest_completion_prediction_recalc_and_clear_scope(
    scope: RecalcScope,
    product_id: str,
    missing_latest_snapshots: list[LatestSnapshotCut],
) -> None:
    failures: list[str] = []
    try:
        compute_lot_completion_predictions(product_id, missing_latest_snapshots)
    except Exception as exc:
        failures.append(str(exc))
        logger.warning(
            "Failed background completion prediction calc for product_id={} missing_lots={}: {}",
            product_id,
            len(missing_latest_snapshots),
            exc,
        )
    finally:
        _finish_recalc_scope(scope, failures, drain_async_writes=False)


def _trigger_latest_completion_prediction_recalc_async(
    product_id: str,
    missing_by_target_day: dict[date, list[LatestSnapshotCut]],
) -> tuple[int, bool]:
    triggered_lot_count = 0
    already_running = False

    for target_day, missing_latest_snapshots in sorted(missing_by_target_day.items()):
        if not missing_latest_snapshots:
            continue

        scope = _target_day_scope_key(product_id, "__date_out_predictions__", target_day)
        with _INFLIGHT_RECALC_LOCK:
            if scope in _INFLIGHT_RECALC_SCOPES:
                already_running = True
                continue
            _INFLIGHT_RECALC_SCOPES.add(scope)
            _clear_scope_failures(scope)

        t = Thread(
            target=_run_latest_completion_prediction_recalc_and_clear_scope,
            args=(scope, product_id, missing_latest_snapshots),
            daemon=True,
        )
        t.start()
        triggered_lot_count += len(missing_latest_snapshots)

    return triggered_lot_count, already_running


def _trigger_latest_lot_progress_recalc_async(
    product_id: str,
    missing_latest_snapshots: list[LatestSnapshotCut],
    target_days: list[date],
) -> tuple[int, bool]:
    scope = _target_days_scope_key(product_id, "__date_out__", target_days)
    with _INFLIGHT_RECALC_LOCK:
        already_running = scope in _INFLIGHT_RECALC_SCOPES
        if already_running:
            return 0, True
        _INFLIGHT_RECALC_SCOPES.add(scope)
        _clear_scope_failures(scope)

    if not missing_latest_snapshots:
        _finish_recalc_scope(scope, drain_async_writes=False)
        return 0, False

    t = Thread(
        target=_run_latest_lot_progress_recalc_and_clear_scope,
        args=(scope, missing_latest_snapshots),
        daemon=True,
    )
    t.start()
    return len(missing_latest_snapshots), False


def _trigger_scope_latest_lot_progress_recalc_async(
    product_id: Optional[str],
    lot_id: Optional[str],
    snapshot_start: Optional[datetime],
    snapshot_end: Optional[datetime],
    missing_latest_snapshots: list[LatestSnapshotCut],
) -> tuple[int, bool]:
    scope = _scope_key(product_id, lot_id, snapshot_start, snapshot_end)
    with _INFLIGHT_RECALC_LOCK:
        already_running = scope in _INFLIGHT_RECALC_SCOPES
        if already_running:
            return 0, True
        _INFLIGHT_RECALC_SCOPES.add(scope)
        _clear_scope_failures(scope)

    if not missing_latest_snapshots:
        _finish_recalc_scope(scope, drain_async_writes=False)
        return 0, False

    t = Thread(
        target=_run_latest_lot_progress_recalc_and_clear_scope,
        args=(scope, missing_latest_snapshots),
        daemon=True,
    )
    t.start()
    return len(missing_latest_snapshots), False


def ensure_date_out_pred_populated(
    product_id: str,
    target_days: list[date],
    context: str = "latest product query",
    wait_if_missing: bool = True,
    max_wait_seconds: float | None = 30.0,
    poll_interval_seconds: float = 1.0,
) -> dict[str, int]:
    """
    Validate date-out completion prediction readiness for product-level queries.

    For each target day, it requires each lot's latest raw snapshot before the end of
    that target day to exist in lot_completion_predictions_table.
    """
    start_monotonic = time.monotonic()
    triggered_lot_count = 0
    triggered_once = False
    waited_after_trigger = False

    while True:
        expected_lot_count, missing_by_target_day = _query_product_date_out_prediction_readiness(
            product_id, target_days
        )
        missing_latest_snapshots_flat = [
            row for rows in missing_by_target_day.values() for row in rows
        ]
        missing_count = sum(
            len(missing_latest_snapshots)
            for missing_latest_snapshots in missing_by_target_day.values()
        )

        if missing_count == 0:
            return {
                "expected_snapshot_cuts": expected_lot_count,
                "ready_snapshot_cuts": expected_lot_count,
                "missing_snapshot_cuts": 0,
                "triggered_lot_count": triggered_lot_count,
            }

        if not triggered_once:
            triggered_lot_count, already_running = (
                _trigger_latest_completion_prediction_recalc_async(
                    product_id=product_id,
                    missing_by_target_day=missing_by_target_day,
                )
            )
            triggered_once = True
            logger.info(
                "Missing latest completion predictions for {} (product_id={}, missing_lots={}, triggered_lots={}, already_running={})",
                context,
                product_id,
                missing_count,
                triggered_lot_count,
                already_running,
            )

        scopes = [
            _target_day_scope_key(product_id, "__date_out_predictions__", target_day)
            for target_day in missing_by_target_day
        ]
        _raise_if_recalc_finished_but_missing(
            scopes=scopes,
            context=context,
            missing_count=missing_count,
            missing_latest_snapshots=missing_latest_snapshots_flat,
            waited_after_trigger=waited_after_trigger,
            detail=f"product_id={product_id}",
        )

        if not wait_if_missing:
            logger.warning(
                "wait_if_missing=False is ignored for {} (product_id={}); readiness is always blocking",
                context,
                product_id,
            )

        elapsed = time.monotonic() - start_monotonic
        logger.info(
            "Waiting for completion prediction readiness for {} (product_id={}, missing_lots={}, elapsed={:.1f}s)",
            context,
            product_id,
            missing_count,
            elapsed,
        )

        if max_wait_seconds is not None:
            if elapsed >= max_wait_seconds:
                raise TimeoutError(
                    "Timed out waiting for lot_completion_predictions data to be ready for "
                    f"{context} after {elapsed:.1f}s "
                    f"(product_id={product_id}, missing_latest_lots={missing_count}, "
                    f"sample={_build_missing_sample(missing_latest_snapshots_flat)})."
                )

            remaining = max_wait_seconds - elapsed
            time.sleep(min(poll_interval_seconds, remaining))
        else:
            time.sleep(poll_interval_seconds)
        waited_after_trigger = True


def ensure_date_out_lot_progress_populated_for_product(
    product_id: str,
    target_days: list[date],
    context: str = "latest product query",
    wait_if_missing: bool = True,
    max_wait_seconds: float | None = 30.0,
    poll_interval_seconds: float = 1.0,
) -> dict[str, int]:
    """
    Validate date-out lot_progress readiness for product-level queries.

    This is intentionally narrower than ensure_lot_progress_populated(product_id=...).
    For each target day, it requires each lot's latest raw snapshot before the end of
    that target day to exist in lot_progress_table.
    """
    start_monotonic = time.monotonic()
    triggered_lot_count = 0
    triggered_once = False
    waited_after_trigger = False
    scope = _target_days_scope_key(product_id, "__date_out__", target_days)

    while True:
        expected_lot_count, missing_latest_snapshots = (
            _query_product_date_out_lot_progress_readiness(product_id, target_days)
        )
        missing_count = len(missing_latest_snapshots)

        if missing_count == 0:
            return {
                "expected_snapshot_cuts": expected_lot_count,
                "ready_snapshot_cuts": expected_lot_count,
                "missing_snapshot_cuts": 0,
                "triggered_lot_count": triggered_lot_count,
            }

        if not triggered_once:
            triggered_lot_count, already_running = _trigger_latest_lot_progress_recalc_async(
                product_id=product_id,
                missing_latest_snapshots=missing_latest_snapshots,
                target_days=target_days,
            )
            triggered_once = True
            logger.info(
                "Missing latest lot_progress snapshots for {} (product_id={}, missing_lots={}, triggered_lots={}, already_running={})",
                context,
                product_id,
                missing_count,
                triggered_lot_count,
                already_running,
            )

        _raise_if_recalc_finished_but_missing(
            scopes=[scope],
            context=context,
            missing_count=missing_count,
            missing_latest_snapshots=missing_latest_snapshots,
            waited_after_trigger=waited_after_trigger,
            detail=f"product_id={product_id}",
        )

        if not wait_if_missing:
            logger.warning(
                "wait_if_missing=False is ignored for {} (product_id={}); readiness is always blocking",
                context,
                product_id,
            )

        elapsed = time.monotonic() - start_monotonic
        logger.info(
            "Waiting for latest lot_progress readiness for {} (product_id={}, missing_lots={}, elapsed={:.1f}s)",
            context,
            product_id,
            missing_count,
            elapsed,
        )

        if max_wait_seconds is not None:
            if elapsed >= max_wait_seconds:
                raise TimeoutError(
                    "Timed out waiting for latest lot_progress_table data to be ready for "
                    f"{context} after {elapsed:.1f}s "
                    f"(product_id={product_id}, missing_latest_lots={missing_count}, "
                    f"sample={_build_missing_sample(missing_latest_snapshots)})."
                )

            remaining = max_wait_seconds - elapsed
            time.sleep(min(poll_interval_seconds, remaining))
        else:
            time.sleep(poll_interval_seconds)
        waited_after_trigger = True


def ensure_lot_progress_snapshot_cuts_populated(
    product_id: str,
    snapshot_cuts: list[LatestSnapshotCut],
    context: str = "query",
    wait_if_missing: bool = True,
    max_wait_seconds: float | None = None,
    poll_interval_seconds: float = 1.0,
) -> dict[str, int]:
    """Validate lot_progress readiness for an explicit lot+snapshot_cut set."""
    snapshot_cuts = _dedupe_snapshot_cuts(snapshot_cuts)
    if not snapshot_cuts:
        return {
            "expected_snapshot_cuts": 0,
            "ready_snapshot_cuts": 0,
            "missing_snapshot_cuts": 0,
            "triggered_lot_count": 0,
        }

    start_monotonic = time.monotonic()
    triggered_lot_count = 0
    triggered_once = False
    waited_after_trigger = False
    scope = _snapshot_cuts_scope_key(product_id, "__explicit_snapshot_cuts__", snapshot_cuts)

    while True:
        expected_lot_count, missing_latest_snapshots = _query_lot_progress_snapshot_cuts_readiness(
            product_id, snapshot_cuts
        )
        missing_count = len(missing_latest_snapshots)

        if missing_count == 0:
            return {
                "expected_snapshot_cuts": expected_lot_count,
                "ready_snapshot_cuts": expected_lot_count,
                "missing_snapshot_cuts": 0,
                "triggered_lot_count": triggered_lot_count,
            }

        if not triggered_once:
            triggered_lot_count, already_running = _trigger_snapshot_cuts_lot_progress_recalc_async(
                product_id=product_id,
                missing_latest_snapshots=missing_latest_snapshots,
                scope=scope,
            )
            triggered_once = True
            logger.info(
                "Missing explicit lot_progress snapshots for {} (product_id={}, missing_lots={}, triggered_lots={}, already_running={})",
                context,
                product_id,
                missing_count,
                triggered_lot_count,
                already_running,
            )

        _raise_if_recalc_finished_but_missing(
            scopes=[scope],
            context=context,
            missing_count=missing_count,
            missing_latest_snapshots=missing_latest_snapshots,
            waited_after_trigger=waited_after_trigger,
            detail=f"product_id={product_id}",
        )

        if not wait_if_missing:
            logger.warning(
                "wait_if_missing=False is ignored for {} (product_id={}); readiness is always blocking",
                context,
                product_id,
            )

        elapsed = time.monotonic() - start_monotonic
        logger.info(
            "Waiting for explicit lot_progress readiness for {} (product_id={}, missing_lots={}, elapsed={:.1f}s)",
            context,
            product_id,
            missing_count,
            elapsed,
        )

        if max_wait_seconds is not None:
            if elapsed >= max_wait_seconds:
                raise TimeoutError(
                    "Timed out waiting for explicit lot_progress_table data to be ready for "
                    f"{context} after {elapsed:.1f}s "
                    f"(product_id={product_id}, missing_latest_lots={missing_count}, "
                    f"sample={_build_missing_sample(missing_latest_snapshots)})."
                )

            remaining = max_wait_seconds - elapsed
            time.sleep(min(poll_interval_seconds, remaining))
        else:
            time.sleep(poll_interval_seconds)
        waited_after_trigger = True


def ensure_lot_progress_populated(
    product_id: Optional[str] = None,
    lot_id: Optional[str] = None,
    stage_names: Optional[str | list[str]] = None,
    snapshot_start: Optional[datetime] = None,
    snapshot_end: Optional[datetime] = None,
    stage_time_start: Optional[datetime] = None,
    stage_time_end: Optional[datetime] = None,
    require_duration: bool = False,
    context: str = "query",
    wait_if_missing: bool = True,
    max_wait_seconds: float | None = None,
    poll_interval_seconds: float = 1.0,
) -> dict[str, int]:
    """
    Validate lot_progress readiness by lot latest-snapshot coverage.

    Readiness rule:
      1) For every lot in scope, find its latest raw snapshot at or before
         snapshot_end.
      2) Require a lot_progress row for each required (lot_id, latest snapshot).
      3) If any required row is missing, trigger background lot-progress
         calculation for those exact snapshots and block until they are ready.

    stage_names, stage_time_start, stage_time_end, and require_duration are accepted
    for API compatibility but not used in the snapshot-time readiness rule.

    When missing snapshot cuts are detected, this function triggers one background
    recalculation per scope and blocks until readiness is satisfied or timeout is reached.
    The wait_if_missing argument is retained for compatibility and ignored.
    """
    _ = (stage_names, stage_time_start, stage_time_end, require_duration)

    start_monotonic = time.monotonic()
    triggered_lot_count = 0
    triggered_once = False
    waited_after_trigger = False
    scope = _scope_key(product_id, lot_id, snapshot_start, snapshot_end)

    while True:
        expected_lot_count, missing_latest_snapshots = _query_scope_latest_lot_progress_readiness(
            product_id, lot_id, snapshot_start, snapshot_end
        )
        missing_count = len(missing_latest_snapshots)
        if expected_lot_count == 0:
            return {
                "expected_snapshot_cuts": 0,
                "ready_snapshot_cuts": 0,
                "missing_snapshot_cuts": 0,
                "triggered_lot_count": 0,
            }

        if missing_count == 0:
            return {
                "expected_snapshot_cuts": expected_lot_count,
                "ready_snapshot_cuts": expected_lot_count,
                "missing_snapshot_cuts": 0,
                "triggered_lot_count": triggered_lot_count,
            }

        if not triggered_once:
            triggered_lot_count, already_running = _trigger_scope_latest_lot_progress_recalc_async(
                product_id=product_id,
                lot_id=lot_id,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
                missing_latest_snapshots=missing_latest_snapshots,
            )
            triggered_once = True
            logger.info(
                "Missing latest lot_progress snapshots for {} (product_id={}, lot_id={}, missing_lots={}, triggered_lots={}, already_running={})",
                context,
                product_id,
                lot_id,
                missing_count,
                triggered_lot_count,
                already_running,
            )

        _raise_if_recalc_finished_but_missing(
            scopes=[scope],
            context=context,
            missing_count=missing_count,
            missing_latest_snapshots=missing_latest_snapshots,
            waited_after_trigger=waited_after_trigger,
            detail=(
                f"product_id={product_id}, lot_id={lot_id}, "
                f"snapshot_start={snapshot_start}, snapshot_end={snapshot_end}"
            ),
        )

        if not wait_if_missing:
            logger.warning(
                "wait_if_missing=False is ignored for {} (product_id={}, lot_id={}); "
                "readiness is always blocking",
                context,
                product_id,
                lot_id,
            )

        elapsed = time.monotonic() - start_monotonic
        logger.info(
            "Waiting for latest lot_progress readiness for {} (product_id={}, lot_id={}, missing_lots={}, elapsed={:.1f}s)",
            context,
            product_id,
            lot_id,
            missing_count,
            elapsed,
        )

        if max_wait_seconds is not None:
            if elapsed >= max_wait_seconds:
                raise TimeoutError(
                    "Timed out waiting for lot_progress_table to be ready for "
                    f"{context} after {elapsed:.1f}s "
                    f"(product_id={product_id}, lot_id={lot_id}, "
                    f"snapshot_start={snapshot_start}, snapshot_end={snapshot_end}, "
                    f"missing_latest_lots={missing_count}, "
                    f"sample={_build_missing_sample(missing_latest_snapshots)})."
                )

            remaining = max_wait_seconds - elapsed
            time.sleep(min(poll_interval_seconds, remaining))
        else:
            time.sleep(poll_interval_seconds)
        waited_after_trigger = True
