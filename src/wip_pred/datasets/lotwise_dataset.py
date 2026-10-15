import warnings

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Timedelta, Timestamp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from wip_pred.datasets.helper import default_collate, noise_mask


def create_indices_in_sliding_windows(
    df: DataFrame,
    min_pred_time: Timestamp,
    max_pred_time: Timestamp,
    window_days: int,
    step_days: int,
    sample_weight: str | None = None,
    sample_prop: float = 0.8,
    alpha: float = 0.9,
    **kwargs,
) -> list:
    """
    Create sliding windows:
    min_pred_time <= the end of the input window <= max_pred_time
    If the length of input is shorter than window_days, it won't be truncated.

    Args:
        df (DataFrame): data
        min_pred_time (Timestamp): the earliest time to make a prediction
        max_pred_time (Timestamp): the latest time to make a prediction
        window_days (int): length of the window
        step_days (int): length of sliding step
        sample_weight (str|None=None): the way to assign sampling weight when sampling, can be `exp` or `linear` or
        `reciprocal`. If it's None, then sampling will not be used
        sample_prop (float): the proportion to sample from  all windows constructed (only used when sample_prop is not None)
        alpha (float): additional param when sample_weight == 'exp'

    Returns:
        list: a list of windows, each window is represented by pd.Index of corresponding rows

    """
    # max_pred_time = max_pred_time.normalize()  # to be consistent with fabwise dataset
    windows = []
    i = 0
    while True:
        window_end = max_pred_time - Timedelta(days=i * step_days)
        window_start = window_end - Timedelta(days=window_days)

        if window_end < min_pred_time:
            break

        window_df = df[(df["created_at"] > window_start) & (df["created_at"] <= window_end)]
        if not window_df.empty:
            windows.append(window_df.index)

        i += 1

        if step_days == 0:
            break

    if sample_weight is None:
        return windows

    n = len(windows)
    k = int(sample_prop * n)

    if k == 0:
        return []

    # sampling: avoid target being too small (this will change the distribution of targets)
    idx = np.arange(n)
    if sample_weight == "exp":
        weights = alpha**idx
    elif sample_weight == "linear":
        weights = n - idx
    elif sample_weight == "reciprocal":
        weights = 1.0 / (idx + 1)

    chosen_idx = np.random.choice(n, size=k, replace=False, p=weights / weights.sum())

    return [windows[i] for i in chosen_idx]


class LotwiseIdxRegDataset(Dataset):
    """
    Note: Since each record may be transformed into a token and used in many samples, we use index in
    the dataframe instead of raw data to save space.

    """

    def __init__(
        self,
        df: DataFrame,
        df_of_completed_lots: DataFrame | None,
        lots_for_test: list[str] | None = None,
        predict_mode: bool = False,
        step_days: int = 3,
        window_days: int = 30,
        target_scaler: tuple[float, float] | None = None,
        group_id: str = "lot_id",
        **kwargs,
    ) -> None:
        """
        Only complete lots with completion_time_lb < completion time <= completion_time_ub will be used to make dataset.

        Args:
            df (DataFrame): data to make dataset (do not use data in the future).
            df_of_completed_lots (DataFrame|None): dataframe of completed lots with following columns (the index of each row is id of all completed lots). Lots with end_time <= latest_time and min_pred_time <= the end of the input window <= max_pred_time will be used to construct samples. For test dataset, you can pass None.
                - product_id
                - parent_lot_id
                - lot_id
                - start_time
                - end_time
                - min_pred_time
                - max_pred_time
                - ...
            lots_for_test (list[str]): when the dataset is used as test dataset, this param specifies which lots you want to predict. For train dataset and validation dataset, simply use an empty list.
            predict_mode (bool): if true, only create one sequence per time series, otherwise create subsequences by sliding a window over the data samples.
            step_days (int): only matters when predict_mode=False. This specifies the step length between windows.
            window_days (int): This specifies the window lengths.
            target_scaler (tuple[float, float]|None): it must be passed as a tuple of two floats for min-max scaler, otherwise it will be fitted.
            group_id (str): the unit ID to construct each sample (sequence), can be `lot_id` or `parent_lot_id` (when the child lots will be used to construct the "sequence" as well, in this case there might be multiple tokens in a time step, if positional encoding is used, the code will be the same for tokens at same time step).
            kwargs: for specifying window sliding strategy (see params in create_indices_in_sliding_windows).

        """
        lots_for_test = lots_for_test or []
        latest_time = df["created_at"].max()

        self.df = df
        self.samples = []
        self.lot_ids = set()
        self.max_seq_len = 0
        self.predict_mode = predict_mode

        if target_scaler:
            assert target_scaler[0] < target_scaler[1]
            self.min_target = target_scaler[0]
            self.max_target = target_scaler[1]
            self.fit_scaler = False
        else:
            self.max_target = 0.0
            self.min_target = float("inf")
            self.fit_scaler = True

        if not predict_mode:
            assert df_of_completed_lots is not None
            if not self.fit_scaler:
                warnings.warn(
                    f"target_scaler={target_scaler} will be used instead of fitting from data. This dataset will be treated as a validation dataset.",
                    UserWarning,
                    stacklevel=2,
                )

            lot_candidates = df_of_completed_lots[
                df_of_completed_lots["end_time"] <= latest_time
            ].index
            required_group_ids = df_of_completed_lots.loc[lot_candidates, group_id].unique()
            df_filtered = df[df[group_id].isin(required_group_ids)]
            group_dict = dict(list(df_filtered.groupby(group_id)))

            for lot_id in lot_candidates:
                target_val = df_of_completed_lots.at[lot_id, group_id]

                lot_data = group_dict[target_val]

                min_pred_time = df_of_completed_lots.at[lot_id, "min_pred_time"]
                max_pred_time = df_of_completed_lots.at[lot_id, "max_pred_time"]  # contain
                windows = create_indices_in_sliding_windows(
                    lot_data,
                    min_pred_time,
                    max_pred_time,
                    window_days,
                    step_days=step_days,
                    **kwargs,
                )
                completion_time = df_of_completed_lots.at[lot_id, "end_time"]

                for indices in windows:
                    pred_time = df.loc[indices]["created_at"].max()
                    target = (completion_time - pred_time).total_seconds() / (24 * 3600)
                    self.max_target = max(self.max_target, target)
                    self.min_target = min(self.min_target, target)
                    sample = {"lot_id": lot_id, "row_indices": indices, "RCT_in_days": target}
                    self.samples.append(sample)
                    self.lot_ids.add(lot_id)
                    self.max_seq_len = max(self.max_seq_len, len(indices))

        else:
            assert (df_of_completed_lots is not None and not lots_for_test) or (
                df_of_completed_lots is None and lots_for_test
            )
            if target_scaler is None:
                raise ValueError("target_scaler must be given when predict_mode=True")

            if df_of_completed_lots is not None:
                lot_candidates = df_of_completed_lots[
                    (df_of_completed_lots["max_pred_time"] >= latest_time)
                    & (latest_time >= df_of_completed_lots["min_pred_time"])
                ].index
                required_group_ids = df_of_completed_lots.loc[lot_candidates, group_id].unique()
            else:
                lot_candidates = lots_for_test
                required_group_ids = df[df["lot_id"].isin(lot_candidates)][group_id].unique()

            df_filtered = df[df[group_id].isin(required_group_ids)]
            group_dict = dict(list(df_filtered.groupby(group_id)))

            for lot_id in lot_candidates:
                if df_of_completed_lots is not None:
                    target_val = df_of_completed_lots.at[lot_id, group_id]
                else:
                    target_val = df[df["lot_id"] == lot_id][group_id].iloc[0]
                lot_data = group_dict.get(target_val, DataFrame())

                min_pred_time = latest_time
                max_pred_time = latest_time
                windows = create_indices_in_sliding_windows(
                    lot_data, min_pred_time, max_pred_time, window_days, step_days
                )

                if df_of_completed_lots is not None:
                    completion_time = df_of_completed_lots.at[lot_id, "end_time"]
                else:
                    completion_time = latest_time  # Placeholder target; ignored in predict mode.

                for indices in windows:
                    pred_time = df.loc[indices]["created_at"].max()
                    sample = {
                        "lot_id": lot_id,
                        "row_indices": indices,
                        "RCT_in_days": (completion_time - pred_time).total_seconds() / (24 * 3600),
                    }
                    self.samples.append(sample)
                    self.lot_ids.add(lot_id)
                    self.max_seq_len = max(self.max_seq_len, len(indices))

        if len(self.samples) == 0:
            raise ValueError("Nothing to predict!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        row_indices = sample["row_indices"]
        lot_id = sample["lot_id"]
        pred = sample["RCT_in_days"]

        return lot_id, row_indices, pred

    def collate_fn(self, batch):
        """
        batch: list of (lot_id, row_indices, pred)

        """
        lot_id_list, row_indices_list, pred_list = zip(*batch, strict=True)

        return lot_id_list, row_indices_list, torch.tensor(pred_list, dtype=torch.float32)

    def to_dataloader(self, batch_size=512, shuffle=True, num_workers=0):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )


class LotwiseDfRegDataset(LotwiseIdxRegDataset):
    def __init__(
        self,
        df: DataFrame,
        df_of_completed_lots: DataFrame | None,
        lots_for_test: list[str] | None = None,
        predict_mode: bool = False,
        step_days: int = 3,
        window_days: int = 30,
        target_scaler: tuple[float, float] | None = None,
        group_id: str = "lot_id",
        **kwargs,
    ) -> None:
        super().__init__(
            df,
            df_of_completed_lots,
            lots_for_test,
            predict_mode,
            step_days,
            window_days,
            target_scaler,
            group_id,
            **kwargs,
        )

        for sample in self.samples:
            row_indices = sample["row_indices"]
            del sample["row_indices"]
            sample["subdf"] = self.df.loc[row_indices]

    def __getitem__(self, index):
        sample = self.samples[index]
        lot_id = sample["lot_id"]
        subdf = sample["subdf"]
        pred = sample["RCT_in_days"]

        return lot_id, subdf, pred

    def collate_fn(self, batch):
        """
        batch: list of (lot_id, df, pred)

        """
        lot_id_list, df_list, pred_list = zip(*batch, strict=True)

        return lot_id_list, df_list, torch.tensor(pred_list, dtype=torch.float32)


class LotwiseTensorRegDataset(LotwiseIdxRegDataset):
    def __init__(
        self,
        df: DataFrame,
        df_of_completed_lots: DataFrame | None = None,
        lots_for_test: list[str] | None = None,
        numeric_feat_cols: list[str] | None = None,
        categorical_feat_cols: list[str] | None = None,
        temporal_feat_cols: list[str] | None = None,
        position_col: str | None = None,
        is_given_col: str | None = None,
        predict_mode: bool = False,
        step_days: int = 3,
        window_days: int = 30,
        target_scaler: tuple[float, float] | None = None,
        group_id: str = "lot_id",
        **kwargs,
    ) -> None:
        """
        Only complete lots with completion_time_lb < completion time <= completion_time_ub will be used to make dataset.

        Args:
            df (DataFrame): data to make dataset (do not use data in the future).
            df_of_completed_lots (DataFrame|None): dataframe of completed lots with following columns (the index of each row is id of all completed lots). Lots with end_time <= latest_time and min_pred_time <= the end of the input window <= max_pred_time will be used to construct samples. For test dataset, you can pass None.
                - product_id
                - parent_lot_id
                - lot_id
                - start_time
                - end_time
                - min_pred_time
                - max_pred_time
                - ...
            lots_for_test (list[str]): when the dataset is used as test dataset, this param specifies which lots you want to predict. For train dataset and validation dataset, simply use an empty list.
            numeric_feat_cols (list[str]):  a list of numeric column names which will be used as features.
            categorical_feat_cols (list[str]): a list of categorical column names which will be used as features.
            temporal_feat_cols (list[str]): a list of temporal column names which will be used as features.
            position_col (str|None): column for constructing position indices, which might be used later for positional encoding (each unique value will be mapped to a unique position in a sorted way).
            is_given_col (str|None): the column indicating whether the row is given or obtained by imputation. This might can be a feature.

            predict_mode (bool): if true, only create one sequence per time series, otherwise create subsequences by sliding a window over the data samples.
            step_days (int): only matters when predict_mode=False. This specifies the step length between windows.
            window_days (int): This specifies the window lengths.

            target_scaler (tuple[float, float]|None): it must be passed as a tuple of two floats for min-max scaler, otherwise it will be fitted.

            group_id (str): the unit ID to construct each sample (sequence), can be `lot_id` or `parent_lot_id` (when the child lots will be used to construct the "sequence" as well, in this case there might be multiple tokens in a time step, if positional encoding is used, the code will be the same for tokens at same time step).
            kwargs: for specifying window sliding strategy (see params in create_indices_in_sliding_windows).

        """

        numeric_feat_cols = numeric_feat_cols or []
        categorical_feat_cols = categorical_feat_cols or []
        temporal_feat_cols = temporal_feat_cols or [
            "year",
            "month",
            "day",
            "weekday",
            "hour",
            "minute",
        ]

        super().__init__(
            df,
            df_of_completed_lots,
            lots_for_test,
            predict_mode,
            step_days,
            window_days,
            target_scaler,
            group_id,
            **kwargs,
        )

        for sample in tqdm(
            self.samples,
            ncols=80,
            desc="Generating samples",
            leave=False,
            disable=predict_mode,
        ):
            row_indices = sample["row_indices"]
            df = self.df.loc[row_indices]
            del sample["row_indices"]

            if numeric_feat_cols:
                sample["numeric"] = torch.tensor(df[numeric_feat_cols].values, dtype=torch.float32)

            sample["categorical"] = dict()
            for col in categorical_feat_cols:
                sample["categorical"][col] = torch.tensor(df[col].values, dtype=torch.long)

            sample["temporal"] = torch.tensor(df[temporal_feat_cols].values, dtype=torch.float32)

            if position_col is not None:
                sample["position"] = torch.tensor(
                    (df[position_col].rank(method="dense", ascending=True).astype(int) - 1).values,
                    dtype=torch.long,
                )

            if is_given_col is not None:
                sample["is_given"] = torch.tensor(df[is_given_col].values, dtype=torch.long)

            sample["length"] = len(df)

    def __getitem__(self, index):
        return self.samples[index]

    def collate_superv_fn(self, batch):
        collated = default_collate(batch, fixed_length=False)
        collated["length"] = torch.tensor(collated["length"], dtype=torch.long)

        lot_id_list = collated.pop("lot_id")
        pred_list = collated.pop("RCT_in_days")

        targets = torch.tensor(pred_list, dtype=torch.float32)
        scaled_targets = (targets - self.min_target) / (self.max_target - self.min_target)

        # the last element will be used as mask (1 means this sample has a label)
        return (
            pd.Series(lot_id_list),
            collated,
            scaled_targets,
            torch.ones(len(lot_id_list), dtype=torch.bool),
        )

    def collate_unsuperv_fn(self, batch):
        # the collate function for the unsupervised task
        # now only support adding noise to categorical features
        IDs, inputs, _, _ = self.collate_superv_fn(batch)
        targets = dict()
        masks = dict()
        dict_of_noise_mask = noise_mask(
            self.noise_cols, len(IDs), self.max_seq_len, self.noise_ratio
        )

        for col, batch_mask in dict_of_noise_mask.items():
            # batch_mask: [batch_size, seq_len]
            # inputs['is_given']: [batch_size, seq_Len]
            # inputs which are meaningless should not be masked because it will be meaning less to predict these
            if "is_given" in inputs:
                batch_mask[inputs["is_given"] == 0] = 1
            else:  # in this case, only inputs outside 'length' will not be masked
                batch_size, max_len = batch_mask.shape
                lengths = inputs["length"]
                range_tensor = (
                    torch.arange(max_len, device=batch_mask.device)
                    .unsqueeze(0)
                    .expand(batch_size, max_len)
                )
                mask_outside = range_tensor >= lengths.unsqueeze(1)
                batch_mask[mask_outside] = 1

            targets[col] = inputs["categorical"][col]
            masks[col] = batch_mask

            # inputs['categorical'][col]: [batch_size, seq_len]
            inputs["categorical"][col] = inputs["categorical"][col] * batch_mask

        return IDs, inputs, targets, masks

    def to_dataloader(
        self,
        batch_size=512,
        shuffle=True,
        num_workers=0,
        superv=True,
        noise_cols: list[str] | None = None,
        noise_ratio: float = 0.2,
    ):
        """
        noise_cols (list[str]): columns to add noise mask. This takes effect only when you are constructing a dataset for pretrain.
        noise_ratio (float): the ratio to add noise mask in a sequence.
        """
        self.noise_cols = noise_cols or []
        self.noise_ratio = noise_ratio
        # to avoid the case in which the last batch_size == 1 (this will raise error in BatchNorm))
        if self.fit_scaler and len(self.samples) % batch_size == 1:
            drop_last = True
        else:
            drop_last = False

        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_superv_fn if superv else self.collate_unsuperv_fn,
            drop_last=drop_last,
        )
