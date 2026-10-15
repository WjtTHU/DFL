import os

import pandas as pd

from wip_pred.utils.visualize import plot_true_vs_pred


def _baseline(
    pred: pd.Series,
    true: pd.Series,
    id_groups: dict | None = None,
    folder_path: str | None = None,
    fig_name: str | None = None,
    fig_title: str = "",
):
    all_cfg = {"x": true.values, "y": pred.values, "alpha": 0, "label": "All"}
    cfgs = [all_cfg]
    if id_groups:
        for name, ids in id_groups.items():
            true_vals = true[true.index.isin(ids)].values
            pred_vals = pred[pred.index.isin(ids)].values

            new_cfg = {"x": true_vals, "y": pred_vals, "alpha": 0.5, "label": name}
            cfgs.append(new_cfg)
    else:
        cfgs[0]["alpha"] = 0.5

    if folder_path and fig_name:
        plot_true_vs_pred(
            true.values, cfgs, title=fig_title, save_path=os.path.join(folder_path, fig_name)
        )

    return pred


def baseline_1(
    df_of_completed_lots: pd.DataFrame,
    split_time: pd.Timestamp,
    val_ids: set,
    folder_path: str | None = None,
):
    # baseline 1: LT - split time (Note: LT must based on trained data)
    # recalculate LT based on data before split time
    LT = (
        df_of_completed_lots[df_of_completed_lots["end_time"] <= split_time]
        .groupby("product_id")
        .agg(LT=("duration", "mean"))
        .reset_index()
    )
    test_df = df_of_completed_lots[df_of_completed_lots.index.isin(val_ids)].copy()
    test_df = test_df.drop(columns=["LT"])
    test_df = pd.merge(test_df, LT, left_on="product_id", right_on="product_id", how="left")

    true = (test_df["end_time"] - split_time).dt.total_seconds() / (24 * 3600)  # type: ignore
    pred = (
        (
            test_df["start_time"] + pd.to_timedelta(test_df["LT"], unit="D") - split_time
        ).dt.total_seconds()
        / (24 * 3600)
    ).clip(lower=0)  # type: ignore

    _baseline(pred, true, None, folder_path, "Baseline1.png", "Validation Predictions (Baseline1)")

    return pred


def baseline_2(
    df: pd.DataFrame,
    pred_col: str,
    df_of_completed_lots: pd.DataFrame,
    split_time: pd.Timestamp,
    val_ids: set,
    id_groups: dict | None = None,
    folder_path: str | None = None,
):
    # baseline 2: Wafer Out Date
    test_df = df[df["lot_id"].isin(val_ids) & (df["created_at"] <= split_time)]
    test_df = test_df.drop_duplicates(["lot_id"], keep="last").copy()
    test_df["pred"] = pd.to_datetime(test_df[pred_col], errors="coerce")
    test_df = test_df[~test_df["pred"].isnull()]
    test_df = test_df[(test_df["pred"] - split_time).dt.days < 500]  # type: ignore
    pred = test_df.set_index("lot_id")["pred"]

    true = (
        df_of_completed_lots[df_of_completed_lots.index.isin(test_df["lot_id"])]["end_time"]
        - split_time
    ).dt.total_seconds() / (24 * 3600)  # type: ignore
    pred = ((true.index.to_series().map(pred) - split_time).dt.total_seconds() / (24 * 3600)).clip(
        lower=0
    )

    _baseline(
        pred, true, id_groups, folder_path, "Baseline2.png", "Validation Predictions (Baseline2)"
    )

    return pred
