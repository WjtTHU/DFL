import pandas as pd
from pandas import DataFrame


def _extract_completed_root_lots(df: DataFrame, margin_in_days: float) -> DataFrame:
    start_within_margin = df["created_at"] < df["created_at"].min() + pd.Timedelta(
        days=margin_in_days
    )
    end_within_margin = df["created_at"] > df["created_at"].max() - pd.Timedelta(
        days=margin_in_days
    )
    del_lots = df[start_within_margin | end_within_margin]["lot_id"].unique()
    return df[~df["lot_id"].isin(del_lots)].reset_index(drop=True)


def get_df_of_completed_lots(df: DataFrame, margin_in_days: float = 3.0) -> DataFrame:
    """
    Get start time and end time of completed lots.

    Args:
        df (DataFrame): snapshot data
        margin_in_days (float): the safe margin from the earlist record date and the
            latest record date to determine whether a lot is completed

    Returns:
        DataFrame: df with columns `product_id`, `lot_id`, `start_time`, `end_time`, `duration`, `LT`, use lot id as row index
    """
    df = df[df["lot_id"] == df["parent_lot_id"]]  # only keep root lots

    # delete incomplete root lots; if strict margin removes all lots, reduce the margin
    # adaptively so short snapshot windows can still produce training labels.
    margins = [max(float(margin_in_days), 0.0)]
    while margins[-1] > 0:
        next_margin = margins[-1] / 2
        if next_margin < 1e-6:
            break
        margins.append(next_margin)
    margins.append(0.0)

    df_filtered = DataFrame()
    for margin in margins:
        df_filtered = _extract_completed_root_lots(df, margin)
        if not df_filtered.empty:
            break

    df = df_filtered

    df_of_completed_lots = (
        df.groupby(["product_id", "lot_id"])
        .agg(start_time=("created_at", "min"), end_time=("created_at", "max"))
        .reset_index()
        .copy()
    )

    df_of_completed_lots["duration"] = (
        df_of_completed_lots["end_time"] - df_of_completed_lots["start_time"]
    ).dt.total_seconds() / (24 * 3600)  # time delta in days

    LT = df_of_completed_lots.groupby("product_id").agg(LT=("duration", "mean")).reset_index()

    df_of_completed_lots = pd.merge(
        df_of_completed_lots, LT, left_on="product_id", right_on="product_id", how="left"
    )

    df_of_completed_lots = df_of_completed_lots.set_index("lot_id", drop=False)
    df_of_completed_lots["parent_lot_id"] = df_of_completed_lots["lot_id"]

    return df_of_completed_lots
