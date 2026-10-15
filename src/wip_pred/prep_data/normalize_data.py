import pandas as pd
from loguru import logger
from pandas import DataFrame

from wip_pred.prep_data.normalize_helper import (
    fill_missing_datapoints,
    remove_dead_lots,
    remove_problematic_lots,
    truncate_meaningless_stages,
)
from wip_pred.prep_data.preprocessor import Preprocessor


def normalize_df(
    df: DataFrame,
    lot_id_col: str,
    parent_lot_id_col: str,
    product_id_col: str,
    stage_col: str,
    status_col: str,
    wafer_qty_col: str,
    timestamp_col: str,
    do_time_binning: bool = False,
    time_binning_freq: str = "10min",
    truncate_expr: str = "",
    truncate_stages: list[str] | None = None,
    duration_thers_of_dead_lots_in_days: float | None = None,
    normalize_by_date: bool = False,
    print_details: bool = False,
) -> DataFrame:
    """
    Normalize data (start from raw data).
    Pipeline:
    1. Check null values in each column
    2. Rename columns to standard column names.
    3. Time binning (Optional).
    4. Drop duplicates.
    5. Remove problematic lots (with more than one product ID).
    6. Sort by timestamp.
    7. Truncate meaningless stages (Optional).
    8. Remove dead lots (Optional).
    9. If normalize_by_date: collapse to one record per Lot per day and fill
       missing calendar days (adds is_given column). Otherwise keep irregular
       timestamps as observed snapshots.
    10. Sort data by lot_id and timestamp.

    """

    truncate_stages = truncate_stages or []

    # Check null values
    logger.debug("Check if there are any null values...")
    if df.isnull().values.any():
        null_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(
            f"Data contains missing values in columns: {null_cols}. "
            "Please verify whether these columns will be input to the model. "
            "If not, remove them before passing the data. "
            "Otherwise, fill or impute the missing values beforehand."
        )

    # Rename columns
    logger.debug("Rename columns...")
    df = df.rename(
        columns={
            lot_id_col: "lot_id",
            parent_lot_id_col: "parent_lot_id",
            product_id_col: "product_id",
            stage_col: "stage",
            status_col: "lot_status",
            wafer_qty_col: "current_wafer_qty",
            timestamp_col: "created_at",
        }
    )

    logger.debug("Convert column {} from string type to timestamp...", timestamp_col)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.sort_values("created_at")  # prepare for dropping duplicates

    # Time binning
    if do_time_binning:
        logger.debug("Do time binning at a frequency of {}...", time_binning_freq)
        df["created_at"] = df["created_at"].dt.round(time_binning_freq)
    else:
        logger.debug("Skip time binning")

    # Drop duplicates
    logger.debug("Drop duplicates...")
    old_row_num = len(df)
    df = df.drop_duplicates(["lot_id", "created_at"], keep="last")
    new_row_num = len(df)
    if print_details:
        print(f"Row num: {old_row_num} -> {new_row_num} after dropping duplicates")

    # Remove problematic lots
    logger.debug("Remove lots with multiple product IDs...")
    old_row_num = len(df)
    df = remove_problematic_lots(
        df,
        lot_id_col="lot_id",
        product_id_col="product_id",
        parent_lot_id_col="parent_lot_id",
        print_details=print_details,
    )
    new_row_num = len(df)
    if print_details:
        print(
            f"Row num: {old_row_num} -> {new_row_num} after removing lots with multiple product IDs"
        )

    # Sort by timestamp
    logger.debug("Order the data ascending by timestamp...")
    df = df.sort_values("created_at").reset_index(drop=True)

    # Truncate meaningless stages at the end
    if truncate_expr or truncate_stages:
        logger.debug("Truncate meaningless stages at the end...")
        df = truncate_meaningless_stages(df, expr=truncate_expr, end_stages=truncate_stages)
    else:
        logger.debug("Skip truncating meaningless stages at the end")

    # Remove dead lots
    if duration_thers_of_dead_lots_in_days:
        logger.debug(
            "Remove lots with duration longer than {} days in a single stage...",
            duration_thers_of_dead_lots_in_days,
        )
        df = remove_dead_lots(df, duration_thres_in_days=duration_thers_of_dead_lots_in_days)
    else:
        logger.debug("Skip removing dead lots")

    if normalize_by_date:
        logger.debug("Normalize timestamp to ensure a maximum of one record per Lot per day...")
        df["created_at"] = df["created_at"].dt.normalize()
        df = df.drop_duplicates(["lot_id", "created_at"], keep="last")

        logger.debug("Fill missing data points of each lot...")
        df = fill_missing_datapoints(df, "lot_id", "created_at", freq="D")
    else:
        logger.debug("Skip calendar-day normalization and gap filling")

    # sort to ensure the data is in order
    logger.debug("Sort records by lot_id and timestamp...")
    df = df.sort_values(["lot_id", "created_at"])

    logger.debug("Finish data normalization!")

    return df


def transform_df(
    df: DataFrame,
    embed_config: dict,
    split_time: pd.Timestamp | None = None,
    preprocessor: Preprocessor | None = None,
):
    if split_time:
        df = df[df["created_at"] <= split_time].copy()
    else:
        df = df.copy()

    # create scalers and label encoders
    numeric_feat_cols = embed_config["numeric_feat_cols"]
    categorical_feat_cols = [col["name"] for col in embed_config["categorical_feat_cols"]]
    temporal_feat_cols = embed_config["temporal_feat_cols"]

    if not preprocessor:
        pre = Preprocessor(
            datetime_col="created_at",
            temporal_features=temporal_feat_cols,
            numeric_cols=numeric_feat_cols,
            categorical_cols=categorical_feat_cols,
        )

        # Note: only non-padded data before split_time should be used for fitting
        df = pre.create_temporal_features(df)
        pre.fit(df[df["is_given"]] if "is_given" in df.columns else df)
    else:
        pre = preprocessor
        df = pre.create_temporal_features(df)

    df = pre.transform(df)

    if preprocessor:
        return df

    for col in embed_config["categorical_feat_cols"]:
        col["num_classes"] = int(df[col["name"]].max() + 1)
        if col["encoder"] == "embed":
            col["dim"] = col.get("dim", min(50, round(col["num_classes"] ** 0.5)))

    return df, pre
