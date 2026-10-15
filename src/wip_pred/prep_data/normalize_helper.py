import numpy as np
import pandas as pd
from pandas import DataFrame

from wip_pred.prep_data.check_data import check_data_normativity


def truncate_meaningless_stages(
    df: DataFrame, expr: str = "", end_stages: list[str] | None = None
) -> DataFrame:
    end_stages = end_stages or []
    # ensure data is sorted by timestamp
    df = df.sort_values("created_at")

    # assign index in the group
    df = df.assign(group_idx=df.groupby("lot_id").cumcount())

    if not expr:
        expr = "|".join(end_stages)

    mask = df["stage"].str.fullmatch(expr, na=False)

    first_occurrences = (
        df[mask]
        .groupby("lot_id")["group_idx"]
        .first()
        .reindex(df["lot_id"].unique(), fill_value=np.inf)
    )

    first_occurrence_mapped = df["lot_id"].map(first_occurrences)

    result = df[df["group_idx"] <= first_occurrence_mapped]

    return result.drop(columns=["group_idx"])


def fill_missing_datapoints(
    df: DataFrame, group_key: str, datetime_col: str, freq: str = "D"
) -> DataFrame:
    assert "is_given" not in df.columns

    df = df.sort_values([group_key, datetime_col]).reset_index(drop=True)

    offset = pd.tseries.frequencies.to_offset(freq)
    delta_ns = offset.nanos

    datetime_values_ns = df[datetime_col].astype("datetime64[ns]").astype(np.int64)

    groups = df.groupby(group_key, observed=True)[datetime_col]
    limits = groups.agg(["min", "max"])
    limits_min_ns = limits["min"].astype("datetime64[ns]").astype(np.int64)
    limits_max_ns = limits["max"].astype("datetime64[ns]").astype(np.int64)

    counts = ((limits_max_ns - limits_min_ns) // delta_ns + 1).astype(int)

    total_rows = counts.sum()
    repeated_keys = np.repeat(limits.index.values, counts)

    start_indices = np.zeros(len(counts), dtype=int)
    np.cumsum(counts.values[:-1], out=start_indices[1:])

    intra_offsets = np.arange(total_rows) - np.repeat(start_indices, counts)
    all_dates = np.repeat(limits_min_ns.values, counts) + (intra_offsets * delta_ns)

    out = pd.DataFrame(
        {group_key: repeated_keys, datetime_col: pd.to_datetime(all_dates, unit="ns")}
    )

    df_group_starts = np.repeat(start_indices, groups.size())
    df_mins = np.repeat(limits_min_ns.values, groups.size())
    target_indices = df_group_starts + (datetime_values_ns.values - df_mins) // delta_ns

    for col in df.columns:
        if col not in [group_key, datetime_col]:
            if df[col].dtype == object or isinstance(df[col].dtype, pd.CategoricalDtype):
                out[col] = None
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                out[col] = pd.NaT
            else:
                out[col] = np.nan

            out.loc[target_indices, col] = df[col].values

    out["is_given"] = False
    out.loc[target_indices, "is_given"] = True

    cols_to_fill = df.columns.difference([group_key, datetime_col])
    out[cols_to_fill] = out.groupby(group_key, observed=True)[cols_to_fill].ffill()

    return out


def get_dead_lots(
    df: DataFrame, datetime_col: str = "created_at", duration_thres_in_days: float = 30.0
):
    """
    Lots with duration in a single stage longer than duration_thres_in_days will be regarded as dead.

    """
    df = df.sort_values(["lot_id", datetime_col]).copy()
    df["stage_change"] = (df["stage"] != df.groupby("lot_id")["stage"].shift()).astype(int)
    df["segment_id"] = df.groupby("lot_id")["stage_change"].cumsum()

    segments = (
        df.groupby(["lot_id", "stage", "segment_id"])
        .agg(enter_time=(datetime_col, "min"), exit_time=(datetime_col, "max"))
        .reset_index()
    )

    segments["stay_duration"] = segments["exit_time"] - segments["enter_time"]
    segments["stay_duration_in_days"] = segments["stay_duration"].dt.total_seconds() / (24 * 3600)

    return segments[segments["stay_duration_in_days"] >= duration_thres_in_days]["lot_id"].unique()


def remove_dead_lots(
    df: DataFrame, datetime_col: str = "created_at", duration_thres_in_days: float = 30.0
):
    dead_lots = get_dead_lots(df, datetime_col, duration_thres_in_days)
    df = df[~df["lot_id"].isin(dead_lots)]

    return df


def remove_problematic_lots(
    df: DataFrame,
    lot_id_col: str,
    product_id_col: str,
    parent_lot_id_col: str,
    print_details: bool = False,
) -> DataFrame:
    """
    Remove rows of problematic lots which do not satisfy data normativity.

    Args:
        df (DataFrame): data to be checked and normalized
        lot_id_col (str): Column name of lot id
        product_id_col (str): Column name of product id
        parent_lot_id_col (str): Column name of parent lot id
        print_details (bool): whether to print invalid details or not

    Returns:
        DataFrame: data without problematic lots
    """

    lots_to_remove = check_data_normativity(
        df, lot_id_col, product_id_col, parent_lot_id_col, print_invalid_details=print_details
    )

    return df[~df[lot_id_col].isin(lots_to_remove)].reset_index(drop=True)
