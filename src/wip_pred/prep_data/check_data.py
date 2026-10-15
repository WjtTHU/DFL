import pandas as pd
from pandas import DataFrame

"""
This script is for checking some assumed property of data.

"""


def _is_one_way_unique(
    df: DataFrame, from_col: str, to_col: str, print_invalid_vals: bool = False
) -> tuple[bool, list]:
    """
    Check if there is a unique one-way mapping from the source column to the target column.
    This means: each value in the source column should map to only one value in the target column.

    Args:
        df (DataFrame): The dataframe to be checked
        from_col (str): Source column name
        to_col (str): Target column name
        print_invalid_vals (bool): whether to print problematic values or not

    Returns:
        tuple: (is_valid, problematic_values)
            - is_valid (bool): True if one-way mapping is maintained for all values in source column
            - problematic_values (list): Values in `from_col` that map to multiple values in `to_col`
    """
    if from_col not in df.columns or to_col not in df.columns:
        raise ValueError("Specified columns do not exits in the input dataframe!")

    unique_pairs = df[[from_col, to_col]].drop_duplicates()
    unique_from = unique_pairs[from_col].nunique()

    if unique_from == len(unique_pairs):
        return True, []

    mapping_cnts = unique_pairs.groupby(from_col).agg(map_cnts=(to_col, "size"))
    mappng_cnts_filtered = mapping_cnts[mapping_cnts["map_cnts"] > 1]

    if print_invalid_vals:
        # pd.set_option('display.max_rows', None)
        print(mappng_cnts_filtered)

        print(
            f"Abnormal Records: {len(mappng_cnts_filtered)} / {df[from_col].nunique()} ({round(100 * len(mappng_cnts_filtered) / df[from_col].nunique(), 2)}%)"
        )

    return False, mappng_cnts_filtered.index.tolist()


def check_data_normativity(
    df: DataFrame,
    lot_id_col: str,
    product_id_col: str,
    parent_lot_id_col: str,
    print_invalid_details=False,
) -> set:
    """
    Check whether the following assumptions are satisfied:
    1. Every lot id maps to a unique product id
    2. Every lot id maps to a unique parent lot id

    Args:
        df (DataFrame): The dataframe to be checked
        lot_id_col (str): Column name of lot id
        product_id_col (str): Column name of product id
        parent_lot_id_col (str): Column name of parent lot id
        print_invalid_details (bool): whether to print problematic values or not

    Returns:
        set: set of problematic lot ids
    """

    if print_invalid_details:
        print("-" * 60)

    if lot_id_col not in df.columns or parent_lot_id_col not in df.columns:
        raise ValueError("Specified columns do not exits in the input dataframe!")

    # Check: Every lot id maps to a unique product id
    is_valid, problematic_vals_1 = _is_one_way_unique(
        df, lot_id_col, product_id_col, print_invalid_details
    )

    if print_invalid_details:
        print(f"\nEvery lot id maps to a unique product id: {is_valid}")
        print("-" * 60)

    # Check: Every lot id maps to a unique parent lot id
    is_valid, problematic_vals_2 = _is_one_way_unique(
        df, lot_id_col, parent_lot_id_col, print_invalid_details
    )

    if print_invalid_details:
        print(f"\nEvery lot id maps to a unique parent lot id: {is_valid}")
        print("-" * 60)

    return set(problematic_vals_1 + problematic_vals_2)


def child_lots_ends_before_root_lots(df: DataFrame, print_invalid_details: bool = False) -> bool:
    """
    Check if all child lots in df end before their root lots.

    """
    df_of_root_lots = df[df["lot_id"].isin(df["parent_lot_id"])]

    max_by_parent = df.groupby("parent_lot_id")["created_at"].max().rename("max_by_parent_lot_id")
    max_by_root_lot = df_of_root_lots.groupby("lot_id")["created_at"].max().rename("max_by_lot_id")

    if max_by_parent.equals(max_by_root_lot.reindex(max_by_parent.index)):
        return True

    if print_invalid_details:
        comparison = pd.merge(
            max_by_parent, max_by_root_lot, how="right", left_index=True, right_index=True
        )
        differences = comparison[comparison["max_by_parent_lot_id"] != comparison["max_by_lot_id"]]
        # pd.set_option('display.max_rows', None)
        print(differences)
        print(
            f"Abnormal Records: {len(differences)} / {len(comparison)} ({round(100 * len(differences) / len(comparison), 2)}%)"
        )

    return False


def is_root_lot_with_largest_quantity(df: DataFrame, print_invalid_details: bool = False):
    """
    Check if root lot has the largest wafer quantity compared with child lots.

    """
    df_of_root_lots = df[df["lot_id"].isin(df["parent_lot_id"])][
        ["parent_lot_id", "created_at", "current_wafer_qty"]
    ]

    max_by_parent = (
        df.groupby(["parent_lot_id", "created_at"])["current_wafer_qty"].max().reset_index()
    )

    comparison = pd.merge(
        max_by_parent, df_of_root_lots, how="right", on=["parent_lot_id", "created_at"]
    )
    comparison = comparison.rename(
        columns={
            "current_wafer_qty_x": "Max Wafer Qty",
            "current_wafer_qty_y": "Wafer Qty of Root Lot",
        }
    )
    differences = comparison[comparison["Max Wafer Qty"] != comparison["Wafer Qty of Root Lot"]]
    if print_invalid_details:
        # pd.set_option('display.max_rows', None)
        print(differences)
        print(
            f"Abnormal Records: {len(differences)} / {len(comparison)} ({round(100 * len(differences) / len(comparison), 2)}%)"
        )

    return len(differences) == 0
