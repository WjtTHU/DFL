"""Experimental advance order book, separate from snapshots and true outcomes.

No actual completion timestamp is copied to the order book. Promises use only
unique labelled training Lots. The first observation date/quantity are assumed
to be planned entry/contract quantity known 28 days earlier, not enterprise facts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_promised_cycles(training_index, cutoff):
    cutoff = pd.Timestamp(cutoff)
    train = training_index.loc[training_index.split == "train"].copy()
    if train.empty:
        raise ValueError("No training Lots for promised-cycle calibration.")
    for key in ("forecast_date", "first_snapshot_at", "completion_at", "label_available_at"):
        train[key] = pd.to_datetime(train[key], errors="raise")
    if not ((train.forecast_date < cutoff) & (train.label_available_at <= cutoff)).all():
        raise ValueError("Promise calibration contains unavailable training labels.")
    if (train.groupby("lot_id")[["product_id", "first_snapshot_at", "completion_at"]]
            .nunique().gt(1).any().any()):
        raise ValueError("Inconsistent product or life-cycle dates for a training Lot.")
    lots = train.drop_duplicates("lot_id").copy()
    lots["cycle_days"] = (lots.completion_at.dt.normalize()
                          - lots.first_snapshot_at.dt.normalize()).dt.days
    if (lots.cycle_days <= 0).any():
        raise ValueError("Observed training cycles must be positive.")
    means = lots.groupby("product_id").cycle_days.agg(["mean", "count"])
    return {
        "training_label_cutoff": str(cutoff),
        "training_lots": len(lots),
        "global_mean_days": float(lots.cycle_days.mean()),
        "product_mean_days": means["mean"].to_dict(),
        "product_training_lots": means["count"].astype(int).to_dict(),
    }


def build_order_book(lot_catalog, fitted, advance_notice_days=28):
    if type(advance_notice_days) is not int or advance_notice_days < 1:
        raise ValueError("Advance notice must be positive integer days.")
    columns = ["lot_id", "woItem", "material_id", "supplier_id", "fixed_quantity",
               "quantity_unit", "first_seen_at"]
    # Deliberate whitelist: final state, actual completion and future snapshots excluded.
    book = lot_catalog[columns].copy()
    if book.lot_id.duplicated().any() or book[columns].isna().any().any():
        raise ValueError("Each Lot needs one complete order record.")
    if not (np.isfinite(book.fixed_quantity) & (book.fixed_quantity > 0)).all():
        raise ValueError("Order quantities must be finite and positive.")
    book["planned_entry_date"] = pd.to_datetime(book.pop("first_seen_at")).dt.normalize()
    book["announced_at"] = book.planned_entry_date - pd.Timedelta(days=advance_notice_days)
    mean = book.woItem.map(fitted["product_mean_days"])
    book["promise_cycle_source"] = np.where(mean.notna(), "training_product_mean", "training_global_mean")
    book["promise_cycle_days"] = np.ceil(mean.fillna(fitted["global_mean_days"])).astype(int)
    book["promised_completion_date"] = book.planned_entry_date + pd.to_timedelta(book.promise_cycle_days, unit="D")
    book["information_origin"] = "synthetic_advance_notice_from_real_lot_first_snapshot"
    return book.sort_values(["announced_at", "lot_id"]).reset_index(drop=True)


def visible_orders(book, decision_at, known_completed_lots=()):
    """Past-due promises do not remove pending orders. Caller supplies known receipts."""
    decision_at = pd.Timestamp(decision_at)
    result = book.loc[(book.announced_at <= decision_at)
                      & ~book.lot_id.isin(set(known_completed_lots))].copy()
    return result.reset_index(drop=True)
