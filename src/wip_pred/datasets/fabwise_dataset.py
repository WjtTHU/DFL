import math
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pandas import DataFrame, MultiIndex, Timedelta, Timestamp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from wip_pred.datasets.helper import default_collate, noise_mask


def log_uniform_int(low: int, high: int):
    """
    A function for determining the sample number.
    Args:
        low (int): lower bound of sampling number
        high (int): upper bound of sampling number

    Returns:
        int: random sampling num
    """

    u = random.random()
    return int(math.exp(math.log(low) + u * (math.log(high) - math.log(low))))


def get_random_indices(N, K):
    indices = []
    segment_len = N / K
    for i in range(K):
        start = int(i * segment_len)
        end = int((i + 1) * segment_len)
        idx = np.random.randint(start, end)
        indices.append(idx)

    return indices


"""
Note: the methods to generate position index in FabwiseRegDataset1 and FabwiseRegDataset2 are a little different, but this may not have a large impact on the result and if pos_enc=None, the result will be the same.

"""


class BaseFabwiseDataset(Dataset):
    def __init__(
        self,
        df: DataFrame,
        df_of_completed_lots: DataFrame | None,
        latest_time: Timestamp,
        lots_for_test: list[str] | None = None,
        predict_mode: bool = False,
        step_days: int = 3,
        window_days: int = 30,
        gamma=0.08,
        target_scaler: tuple[float, float] | None = None,
        random_sampling=False,
        random_sample_num: int | None = None,
        min_batch_size: int = 2,
    ) -> None:
        """
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
            latest_time (Timestamp): the latest timestamp in df.

            lots_for_test (list[str]): when the dataset is used as test dataset, this param specifies which lots you want to predict. For train dataset and validation dataset, simply use an empty list.

            predict_mode (bool): if true, only create one sequence per time series, otherwise create subsequences by sliding a window over the data samples.
            step_days (int): only matters when predict_mode=False. This specifies the step length between windows.
            window_days (int): This specifies the window lengths.

            gamma (float): param for sampling in a batch.
            target_scaler (tuple[float, float]|None): it must be passed as a tuple of two floats for min-max scaler, otherwise it will be fitted.

            random_sampling (bool): whether use random sliding windows or not.
            min_batch_size (int): when the number of samples in the batch is less than this, this batch will be discarded (the batch size must be greater than 1 for BatchNorm layer).

        """

        lots_for_test = lots_for_test or []

        self.lot_ids = set()
        self.batches = []
        self.gamma = gamma
        self.max_seq_len = 0
        self.min_seq_len = float("inf")
        self.predict_mode = predict_mode
        self.random_sampling = random_sampling
        self.min_batch_size = min_batch_size

        if target_scaler:
            assert target_scaler[0] < target_scaler[1]
            self.min_target = target_scaler[0]
            self.max_target = target_scaler[1]
            self.fit_scaler = False
        else:
            self.max_target = 0.0
            self.min_target = float("inf")
            self.fit_scaler = True

        if random_sampling:
            assert random_sample_num is not None
        else:
            random_sample_num = None

        if not predict_mode:
            assert df_of_completed_lots is not None
            if not self.fit_scaler:
                warnings.warn(
                    f"target_scaler={target_scaler} will be used instead of fitting from data.",
                    UserWarning,
                    stacklevel=2,
                )

        else:
            assert (df_of_completed_lots is not None and not lots_for_test) or (
                df_of_completed_lots is None and lots_for_test
            )
            if target_scaler is None:
                raise ValueError("target_scaler must be given when predict_mode=True")

        windows = self._generate_windows(
            df,
            df_of_completed_lots,
            latest_time,
            lots_for_test,
            predict_mode,
            step_days,
            window_days,
        )

        # process window_df if necessary
        windows = self._process_window_df(windows)

        if random_sample_num:
            self.window_num = min(len(windows), random_sample_num)
        else:
            self.window_num = len(windows)

        self._prepare_for_batching()

        self.batches = []
        for window in tqdm(
            reversed(windows),
            ncols=80,
            desc="Generating batches",
            leave=False,
            disable=predict_mode,
        ):
            batch = self.make_batch(window, df_of_completed_lots)
            self.batches.append(batch)

        if self.max_seq_len != self.min_seq_len:
            raise ValueError(
                f"The lengths of samples are inconsistent (max_len={self.max_seq_len}; min_len={self.min_seq_len}). Please check the data!"
            )

        if len(self.lot_ids) == 0:
            raise ValueError("Nothing to predict!")

    def __len__(self):
        return self.window_num

    def __getitem__(self, idx):
        if self.random_sampling:
            idx = get_random_indices(len(self.batches), self.window_num)[idx]
        return self.batches[idx]

    def collate_superv_fn(self, batch):
        # the collate function for the supervised task
        batch = batch[0]

        seqs_with_label, seqs_without_label = batch["seqs_with_label"], batch["seqs_without_label"]

        M = len(seqs_with_label) + len(seqs_without_label)

        # sampling
        if not self.predict_mode:
            D = log_uniform_int(1, M) if random.random() < self.gamma else M
        else:
            D = M

        if len(seqs_with_label) >= D:
            selected_seqs = random.sample(seqs_with_label, D)
        else:
            selected_seqs = seqs_with_label + random.sample(
                seqs_without_label, D - len(seqs_with_label)
            )

        collated = default_collate(selected_seqs)

        lot_id_list = collated.pop("lot_id")
        target_list = collated.pop("RCT_in_days")
        target_mask = collated.pop("with_label")

        target = torch.tensor(target_list, dtype=torch.float32)
        scaled_target = (target - self.min_target) / (self.max_target - self.min_target)

        target_mask = torch.tensor(target_mask, dtype=torch.bool)

        return pd.Series(lot_id_list), collated, scaled_target, target_mask

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
            # we only have correct labels when is_given=True, that is 1 in inputs['is_given'] and batch_mask can be 0 in these places
            batch_mask[inputs["is_given"] == 0] = 1

            targets[col] = inputs["categorical"][col]
            masks[col] = batch_mask

            # inputs['categorical'][col]: [batch_size, seq_len]
            inputs["categorical"][col] = inputs["categorical"][col] * batch_mask

        return IDs, inputs, targets, masks

    def _process_window_df(self, windows):
        return windows

    def _generate_windows(
        self,
        df: DataFrame,
        df_of_completed_lots: DataFrame | None,
        latest_time: Timestamp,
        lots_for_test: list[str],
        predict_mode: bool,
        step_days: int,
        window_days: int,
    ):
        # generate each window
        windows = []

        if not predict_mode and df_of_completed_lots is not None:
            min_time = df["created_at"].min()
            if self.random_sampling:
                step_days = 1
            i = 0
            while True:
                window_end = latest_time - Timedelta(days=i * step_days)
                window_start = window_end - Timedelta(days=window_days)

                if window_start < min_time:
                    break

                window_df = df[(df["created_at"] > window_start) & (df["created_at"] <= window_end)]
                lots_to_predict = df_of_completed_lots[
                    (df_of_completed_lots["end_time"] <= latest_time)
                    & (df_of_completed_lots["min_pred_time"] <= window_end)
                    & (df_of_completed_lots["max_pred_time"] >= window_end)
                ].index

                lots_with_label = window_df[window_df["lot_id"].isin(lots_to_predict)][
                    "lot_id"
                ].unique()

                if len(lots_with_label) >= self.min_batch_size:
                    self.lot_ids.update(lots_with_label)
                    pred_time = window_df["created_at"].max()
                    windows.append(
                        {
                            "window_df": window_df,
                            "lots_to_predict": lots_to_predict,
                            "pred_time": pred_time,
                        }
                    )

                i += 1
        else:
            window_end = latest_time
            window_start = window_end - Timedelta(days=window_days)
            window_df = df[(df["created_at"] > window_start) & (df["created_at"] <= window_end)]
            if df_of_completed_lots is None:
                lots_to_predict = lots_for_test
            else:
                lots_to_predict = df_of_completed_lots[
                    (df_of_completed_lots["max_pred_time"] >= latest_time)
                    & (latest_time >= df_of_completed_lots["min_pred_time"])
                ].index

            self.lot_ids.update(
                window_df[window_df["lot_id"].isin(lots_to_predict)]["lot_id"].unique()
            )

            pred_time = window_df["created_at"].max()

            windows.append(
                {"window_df": window_df, "lots_to_predict": lots_to_predict, "pred_time": pred_time}
            )

        return windows

    def to_dataloader(
        self,
        shuffle=True,
        batch_size=1,
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
        if batch_size != 1:
            warnings.warn(
                f"batch_size={batch_size} is not supported for fabwise dataset. batch_size=1 will be used instead.",
                UserWarning,
                stacklevel=2,
            )
        return DataLoader(
            dataset=self,
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_superv_fn if superv else self.collate_unsuperv_fn,
        )

    def _prepare_for_batching(self):
        pass

    def make_batch(self, window: dict, df_of_completed_lots: DataFrame | None):
        pass


class FabwiseRegDataset1(BaseFabwiseDataset):
    def __init__(
        self,
        df: DataFrame,
        position_col: str,
        df_of_completed_lots: DataFrame | None = None,
        lots_for_test: list[str] | None = None,
        numeric_feat_cols: list[str] | None = None,
        categorical_feat_cols: list[str] | None = None,
        temporal_feat_cols: list[str] | None = None,
        is_given_col: str | None = None,
        predict_mode: bool = False,
        step_days: int = 3,
        window_days: int = 30,
        gamma=0.08,
        constructed_seqs: dict | None = None,
        target_scaler: tuple[float, float] | None = None,
        random_sampling=False,
        random_sample_num: int | None = None,
        min_batch_size: int = 2,
        **kwargs,
    ) -> None:
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

        # assumption: every lot can only have one record at one time
        assert not df.duplicated(subset=["lot_id", "created_at"]).any()
        assert "_abs_pos" not in df.columns

        latest_time = df["created_at"].max()
        df = df.copy()  # only data before split time can be used

        # generate absolute position idx
        df["_abs_pos"] = df[position_col].rank(method="dense", ascending=True).astype(int) - 1

        self.position_col = position_col
        self.numeric_feat_cols = numeric_feat_cols
        self.categorical_feat_cols = categorical_feat_cols
        self.temporal_feat_cols = temporal_feat_cols
        self.is_given_col = is_given_col

        if not predict_mode:
            self.seqs = self._make_tensor_of_seqs(df, constructed_seqs)
        else:
            self.seqs = self._make_tensor_of_seqs(
                df[
                    (df["created_at"] > latest_time - Timedelta(days=window_days))
                    & (df["created_at"] <= latest_time)
                ],
                constructed_seqs,
            )  # to speed up tensor making

        super().__init__(
            df,
            df_of_completed_lots,
            latest_time,
            lots_for_test,
            predict_mode,
            step_days,
            window_days,
            gamma,
            target_scaler,
            random_sampling,
            random_sample_num,
            min_batch_size,
        )

    def _make_tensor_of_seqs(self, df, constructed_seqs):
        # turn features of each sequence into tensors
        if constructed_seqs:
            seqs = constructed_seqs
        else:
            seqs = dict()
            for lot_id, lot_data in df.groupby("lot_id"):
                seq = dict()

                seq["lot_id"] = lot_id

                if self.numeric_feat_cols:
                    seq["numeric"] = torch.tensor(
                        lot_data[self.numeric_feat_cols].values, dtype=torch.float32
                    )

                seq["categorical"] = dict()
                for col in self.categorical_feat_cols:
                    seq["categorical"][col] = torch.tensor(lot_data[col].values, dtype=torch.long)

                seq["temporal"] = torch.tensor(
                    lot_data[self.temporal_feat_cols].values, dtype=torch.float32
                )

                seq["_abs_pos"] = torch.tensor(
                    lot_data["_abs_pos"].values, dtype=torch.long
                )  # auxiliary attributes (start with `_`)

                if self.is_given_col is not None:
                    seq["is_given"] = torch.tensor(
                        lot_data[self.is_given_col].values.astype(bool), dtype=torch.long
                    )

                seqs[lot_id] = seq

        return seqs

    def make_batch(self, window: dict, df_of_completed_lots: DataFrame | None):
        window_df = window["window_df"]
        lots_to_predict = window["lots_to_predict"]
        pred_time = window["pred_time"]
        input_len = window_df["_abs_pos"].nunique()
        self.max_seq_len = max(self.max_seq_len, input_len)
        self.min_seq_len = min(self.min_seq_len, input_len)

        # for each batch, record min_abs_pos, max_abs_pos, lot ids contained
        min_abs_pos = window_df["_abs_pos"].min()
        max_abs_pos = window_df["_abs_pos"].max()
        lots_with_label = dict()
        lots_without_label = dict()
        for lot_id in window_df["lot_id"].unique().tolist():
            if lot_id in lots_to_predict:
                if df_of_completed_lots is not None:  # train or validation set
                    target = (
                        df_of_completed_lots.at[lot_id, "end_time"] - pred_time
                    ).total_seconds() / (24 * 3600)
                else:  # test set
                    target = 0.0  # Placeholder target; masked/ignored in predict mode.
                lots_with_label[lot_id] = {"with_label": 1, "RCT_in_days": target}
                if self.fit_scaler:
                    self.min_target = min(target, self.min_target)
                    self.max_target = max(target, self.max_target)
            else:
                lots_without_label[lot_id] = {"with_label": 0, "RCT_in_days": 0.0}

        seqs_with_label = []
        seqs_without_label = []
        # get padded tensor slices to construct full batch
        for lot_id, labels in lots_with_label.items():
            seq = FabwiseRegDataset1.get_seq_slice(self.seqs[lot_id], min_abs_pos, max_abs_pos)
            for label_key in labels:
                seq[label_key] = labels[label_key]

            seqs_with_label.append(seq)

        for lot_id, labels in lots_without_label.items():
            seq = FabwiseRegDataset1.get_seq_slice(self.seqs[lot_id], min_abs_pos, max_abs_pos)
            for label_key in labels:
                seq[label_key] = labels[label_key]

            seqs_without_label.append(seq)

        return {"seqs_with_label": seqs_with_label, "seqs_without_label": seqs_without_label}

    @staticmethod
    def pad_first_dim(x, pad_left: int, pad_right: int, value=0):
        pad = [0] * (2 * (x.ndim - 1)) + [pad_left, pad_right]
        return F.pad(x, tuple(pad), value=value)

    @staticmethod
    def get_seq_slice(seq: dict, min_abs_pos: int, max_abs_pos: int):
        seq_slice = dict()

        seq_slice["lot_id"] = seq["lot_id"]
        min_local_pos = seq["_abs_pos"].min().item()
        max_local_pos = seq["_abs_pos"].max().item()

        left_pad = max(0, min_local_pos - min_abs_pos)
        right_pad = max(0, max_abs_pos - max_local_pos)
        start = max(0, min_abs_pos - min_local_pos)
        end = min(max_abs_pos - min_local_pos + 1, max_local_pos - min_local_pos + 1)

        # ----- adjust pos -----
        # method 1: use relative position in window for valid tokens only (typical way)
        pos_slice = seq["_abs_pos"][start:end]
        seq_slice["position"] = FabwiseRegDataset1.pad_first_dim(
            pos_slice - pos_slice.min(), pad_left=left_pad, pad_right=right_pad
        )

        # method 2: use relative position in window for all tokens (consistent with method in SetSeqSlidingDataset) -- best
        # seq_slice['position'] = torch.arange(max_abs_pos - min_abs_pos + 1)

        # method 3: use relative position in sequence
        # seq_slice['position'] = FabwiseRegDataset.pad_first_dim((seq['_abs_pos'] - seq['_abs_pos'].min())[start:end], pad_left=left_pad, pad_right=right_pad)

        # method 4: not use position at all
        # ----------------------

        for key in seq.keys():
            if key != "categorical" and key != "lot_id" and not key.startswith("_"):
                seq_slice[key] = FabwiseRegDataset1.pad_first_dim(
                    seq[key][start:end], left_pad, right_pad
                )

        seq_slice["categorical"] = dict()
        for cat_key in seq["categorical"]:
            seq_slice["categorical"][cat_key] = FabwiseRegDataset1.pad_first_dim(
                seq["categorical"][cat_key][start:end], left_pad, right_pad
            )

        # add mask
        mask = torch.zeros(max_abs_pos - min_abs_pos + 1)
        mask[left_pad : left_pad + end - start] = 1
        seq_slice["mask"] = mask

        return seq_slice


class FabwiseRegDataset2(BaseFabwiseDataset):
    def __init__(
        self,
        df: DataFrame,
        position_col: str,
        df_of_completed_lots: DataFrame | None = None,
        lots_for_test: list[str] | None = None,
        numeric_feat_cols: list[str] | None = None,
        categorical_feat_cols: list[str] | None = None,
        temporal_feat_cols: list[str] | None = None,
        is_given_col: str | None = None,
        predict_mode: bool = False,
        step_days: int = 3,
        window_days: int = 30,
        gamma=0.08,
        target_scaler: tuple[float, float] | None = None,
        random_sampling=False,
        random_sample_num: int | None = None,
        min_batch_size: int = 2,
        **kwargs,
    ) -> None:
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

        warnings.warn(
            f"{self.__class__.__name__} is deprecated. Use FabwiseRegDataset1 instead, which performs calculations more efficiently.",
            DeprecationWarning,
            stacklevel=2,
        )

        # assumption: every lot can only have one record at one time
        assert not df.duplicated(subset=["lot_id", "created_at"]).any()
        assert "_pos" not in df.columns

        latest_time = df["created_at"].max()

        self.position_col = position_col
        self.numeric_feat_cols = numeric_feat_cols
        self.categorical_feat_cols = categorical_feat_cols
        self.temporal_feat_cols = temporal_feat_cols
        self.is_given_col = is_given_col

        super().__init__(
            df,
            df_of_completed_lots,
            latest_time,
            lots_for_test,
            predict_mode,
            step_days,
            window_days,
            gamma,
            target_scaler,
            random_sampling,
            random_sample_num,
            min_batch_size,
        )

    def _process_window_df(self, windows):
        for window in windows:
            window_df = window["window_df"].copy()
            window_df["_pos"] = (
                window_df[self.position_col].rank(method="dense", ascending=True).astype(int) - 1
            )
            window_df = FabwiseRegDataset2.padding(window_df, "_pos")
            window_df = window_df.fillna(0)
            window_df = window_df.sort_values(["lot_id", "_pos"])
            window["window_df"] = window_df

        return windows

    @staticmethod
    def padding(df: DataFrame, rank_col: str):
        all_ids = df["lot_id"].unique()
        all_ranks = df[rank_col].unique()
        full_index = MultiIndex.from_product(
            [all_ids, all_ranks],
            names=["lot_id", rank_col],
        )
        full_df = DataFrame(index=full_index).reset_index()

        # merge with original data
        merged_df = full_df.merge(df, on=["lot_id", rank_col], how="left", indicator=True)

        # mark padding values
        merged_df["is_padding"] = merged_df["_merge"].eq("left_only")
        merged_df = merged_df.drop(columns=["_merge"])

        return merged_df

    def make_batch(self, window: dict, df_of_completed_lots: DataFrame | None):
        window_df = window["window_df"]
        lots_to_predict = window["lots_to_predict"]
        pred_time = window["pred_time"]
        input_len = window_df["_pos"].nunique()
        self.max_seq_len = max(self.max_seq_len, input_len)
        self.min_seq_len = min(self.min_seq_len, input_len)
        seqs_with_label = []
        seqs_without_label = []
        for lot_id, lot_data in window_df.groupby("lot_id"):
            seq = dict()
            if lot_id in lots_to_predict:
                seq["with_label"] = 1
                if df_of_completed_lots is not None:
                    target = (
                        df_of_completed_lots.at[lot_id, "end_time"] - pred_time
                    ).total_seconds() / (24 * 3600)
                else:
                    target = 0.0
                seq["RCT_in_days"] = target
                if self.fit_scaler:
                    self.min_target = min(target, self.min_target)
                    self.max_target = max(target, self.max_target)
            else:
                seq["with_label"] = 0
                seq["RCT_in_days"] = 0

            seq["lot_id"] = lot_id

            if self.numeric_feat_cols:
                seq["numeric"] = torch.tensor(
                    lot_data[self.numeric_feat_cols].values, dtype=torch.float32
                )

            seq["categorical"] = dict()
            for col in self.categorical_feat_cols:
                seq["categorical"][col] = torch.tensor(lot_data[col].values, dtype=torch.long)

            seq["temporal"] = torch.tensor(
                lot_data[self.temporal_feat_cols].values, dtype=torch.float32
            )

            seq["position"] = torch.tensor(lot_data["_pos"].values, dtype=torch.long)

            if self.is_given_col is not None:
                seq["is_given"] = torch.tensor(
                    lot_data[self.is_given_col].values.astype(bool), dtype=torch.long
                )

            seq["mask"] = torch.tensor(
                (~lot_data["is_padding"]).values, dtype=torch.long
            )  # 1 for not masking and 0 for masking

            if seq["with_label"] == 1:
                seqs_with_label.append(seq)
            else:
                seqs_without_label.append(seq)

        return {"seqs_with_label": seqs_with_label, "seqs_without_label": seqs_without_label}
