import time
from functools import lru_cache
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from typing import Iterable, Sequence, cast

import pandas as pd
import polars as pl
import torch
import yaml
from loguru import logger

from wip_analysis.utils.env_num import env_num
from wip_pred.data_loader.fetch_data import fetch_raw_data
from wip_pred.datasets.get_dataset import get_dataset
from wip_pred.inference.model_loader import load_model_and_config
from wip_pred.prep_data.normalize_data import normalize_df, transform_df

DEFAULT_MODEL_DIR = Path("checkpoints/_release")
DEFAULT_TRUNCATE_STAGES = []
# How long the batch worker lingers collecting same-split_time requests before a
# forward pass, and the max requests folded into one batch. These pay off under
# high concurrency of *many lots at the same split_time* (cross-lot batching); they
# do nothing for a single lot's many-cut history (each cut is a distinct split_time).
DEFAULT_BATCH_WAIT_SECONDS = env_num("WIP_ANALYSIS_INFERENCE_BATCH_WAIT_SECONDS", 0.02)
DEFAULT_MAX_BATCH_REQUESTS = env_num("WIP_ANALYSIS_INFERENCE_MAX_BATCH_REQUESTS", 96)

_BASE_SNAPSHOT_COLS = (
    "lot_id",
    "parent_lot_id",
    "product_id",
    "stage",
    "lot_status",
    "current_wafer_qty",
    "wafer_start_date",
    "created_at",
)


def _extra_feature_cols_from_embd(embed_config: dict | None) -> list[str]:
    if not embed_config:
        return []
    cols: list[str] = []
    for col in embed_config.get("numeric_feat_cols", []):
        if col not in _BASE_SNAPSHOT_COLS and col not in cols:
            cols.append(col)
    for cat in embed_config.get("categorical_feat_cols", []):
        name = cat["name"] if isinstance(cat, dict) else cat
        if name not in _BASE_SNAPSHOT_COLS and name not in cols:
            cols.append(name)

    logger.opt(lazy=True).debug("Extra Feature Columns: {}", lambda: cols)
    return cols


def _coerce_lot_ids(lot_ids: str | Iterable[str]) -> list[str]:
    if isinstance(lot_ids, str):
        return [lot_ids]
    return list(lot_ids)


def _empty_prediction_result() -> pd.Series:
    return pd.Series(dtype="float64")


class _PredictionRequest:
    def __init__(
        self,
        lot_snapshots: pd.DataFrame | pl.DataFrame,
        lot_ids: tuple[str, ...],
        split_time: pd.Timestamp,
        batch_size: int | None,
    ) -> None:
        self.lot_snapshots = lot_snapshots
        self.lot_ids = lot_ids
        self.split_time = split_time
        self.batch_size = batch_size
        self.done = Event()
        self.result: pd.Series | None = None
        self.error: BaseException | None = None


def predict(
    model,
    df: pd.DataFrame,
    config: dict,
    lot_ids: str | list[str] | set[str] | tuple[str, ...],
    batch_size: int | None = None,
) -> pd.Series:
    """Run batched inference on an already transformed dataframe."""
    lot_ids = _coerce_lot_ids(lot_ids)
    if not lot_ids or df.empty:
        return _empty_prediction_result()

    min_target = config["target_scaler"]["min"]
    max_target = config["target_scaler"]["max"]

    dataset = get_dataset(
        df, config, predict_mode=True, lots_for_test=lot_ids, target_scaler=(min_target, max_target)
    )
    dataloader_kwargs = {"shuffle": False}
    if config["dataset"]["type"] != "fabwise":
        dataloader_kwargs["batch_size"] = batch_size or env_num(
            "WIP_ANALYSIS_INFERENCE_BATCH_SIZE", 512
        )
    dataloader = dataset.to_dataloader(**dataloader_kwargs)
    all_preds = []
    all_ids = []
    with torch.inference_mode():
        for ids, inputs, _, target_masks in dataloader:
            pred = model(inputs).view(-1)
            mask_cpu = target_masks.detach().cpu().bool().view(-1)
            mask_device = mask_cpu.to(pred.device)
            selected = pred[mask_device]
            if selected.numel() == 0:
                continue
            all_preds.append(selected)
            all_ids.append(ids[mask_cpu.numpy()])

    if not all_preds:
        return _empty_prediction_result()

    all_preds = torch.cat(all_preds).detach().cpu().numpy() * (max_target - min_target) + min_target
    all_ids = pd.concat(all_ids)

    result = pd.Series(all_preds, index=all_ids.values)
    filtered_result = result[result.index.isin(lot_ids)]

    return filtered_result


class WipInferenceService:
    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        batch_wait_seconds: float = DEFAULT_BATCH_WAIT_SECONDS,
        max_batch_requests: int = DEFAULT_MAX_BATCH_REQUESTS,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model, self.preprocessor, self.config = load_model_and_config(
            self.model_dir.as_posix()
        )
        self.config = cast(dict, self.config)
        self._model_lock = Lock()
        self._batch_wait_seconds = max(0.0, float(batch_wait_seconds))
        self._max_batch_requests = max(1, int(max_batch_requests))
        self._batch_lock = Lock()
        self._batch_condition = Condition(self._batch_lock)
        self._pending_requests: list[_PredictionRequest] = []
        self._batch_worker_running = False

    @property
    def window_days(self) -> int:
        return int(self.config["dataset"]["params"]["window_days"])

    @property
    def requires_facility_context(self) -> bool:
        return self.config.get("dataset", {}).get("type") != "lotwise"

    @property
    def extra_feat_cols(self) -> list[str]:
        return _extra_feature_cols_from_embd(self.config.get("embed", {}))

    def transform_snapshots(
        self,
        lot_snapshots: pd.DataFrame | pl.DataFrame,
        split_time: pd.Timestamp,
        truncate_stages: Sequence[str] = DEFAULT_TRUNCATE_STAGES,
    ) -> pd.DataFrame:
        """Normalize and transform raw snapshot rows for one prediction timestamp."""
        if isinstance(lot_snapshots, pl.DataFrame):
            raw_df = lot_snapshots.to_pandas()
        else:
            raw_df = lot_snapshots.copy()

        if raw_df.empty:
            return pd.DataFrame()

        required_cols = list(_BASE_SNAPSHOT_COLS)
        for col in self.extra_feat_cols:
            if col not in required_cols:
                required_cols.append(col)

        embed_config = self.config.get("embed", {})
        for col in embed_config.get("numeric_feat_cols", []):
            if col not in required_cols:
                required_cols.append(col)
        for cat in embed_config.get("categorical_feat_cols", []):
            name = cat["name"] if isinstance(cat, dict) else cat
            if name not in required_cols:
                required_cols.append(name)
        missing_cols = [col for col in required_cols if col not in raw_df.columns]
        if missing_cols:
            raise ValueError(f"Missing snapshot columns for prediction: {missing_cols}")

        window_start = split_time - pd.Timedelta(days=self.window_days)
        raw_df = raw_df.loc[pd.to_datetime(raw_df["created_at"]) >= window_start, :]
        if raw_df.empty:
            return pd.DataFrame()

        df_processed = normalize_df(
            raw_df,
            "lot_id",
            "parent_lot_id",
            "product_id",
            "stage",
            "lot_status",
            "current_wafer_qty",
            "created_at",
            truncate_stages=list(truncate_stages),
            do_time_binning=False,
            normalize_by_date=self.config["embed"]["use_given_emb"],
        )
        if df_processed.empty:
            return pd.DataFrame()

        return transform_df(
            df=df_processed,
            embed_config=self.config["embed"],
            split_time=split_time,
            preprocessor=self.preprocessor,
        )

    def predict_transformed(
        self,
        df: pd.DataFrame,
        lot_ids: str | list[str] | set[str] | tuple[str, ...],
        batch_size: int | None = None,
    ) -> pd.Series:
        return self._predict_transformed_now(df, lot_ids, batch_size=batch_size)

    def _predict_transformed_now(
        self,
        df: pd.DataFrame,
        lot_ids: str | list[str] | set[str] | tuple[str, ...],
        batch_size: int | None = None,
    ) -> pd.Series:
        with self._model_lock:
            return predict(self.model, df, self.config, lot_ids, batch_size=batch_size)

    def predict_remaining_days(
        self,
        lot_snapshots: pd.DataFrame | pl.DataFrame,
        lot_ids: str | list[str] | set[str] | tuple[str, ...],
        split_time: pd.Timestamp,
        batch_size: int | None = None,
    ) -> pd.Series:
        """Preprocess raw snapshot rows and return remaining-time predictions in days."""
        request = _PredictionRequest(
            lot_snapshots=lot_snapshots,
            lot_ids=tuple(_coerce_lot_ids(lot_ids)),
            split_time=pd.Timestamp(split_time),
            batch_size=batch_size or env_num("WIP_ANALYSIS_INFERENCE_BATCH_SIZE", 512),
        )
        if not request.lot_ids:
            return _empty_prediction_result()

        self._enqueue_prediction_request(request)
        request.done.wait()
        if request.error is not None:
            raise request.error
        return request.result if request.result is not None else _empty_prediction_result()

    def _enqueue_prediction_request(self, request: _PredictionRequest) -> None:
        with self._batch_condition:
            self._pending_requests.append(request)
            if not self._batch_worker_running:
                self._batch_worker_running = True
                Thread(target=self._batch_worker, daemon=True).start()
            self._batch_condition.notify()

    def _batch_worker(self) -> None:
        while True:
            with self._batch_condition:
                if not self._pending_requests:
                    self._batch_worker_running = False
                    return

                deadline = time.monotonic() + self._batch_wait_seconds
                while len(self._pending_requests) < self._max_batch_requests:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._batch_condition.wait(timeout=remaining)

                requests = self._pending_requests[: self._max_batch_requests]
                del self._pending_requests[: self._max_batch_requests]

            self._process_prediction_batch(requests)

    def _process_prediction_batch(self, requests: list[_PredictionRequest]) -> None:
        grouped: dict[pd.Timestamp, list[_PredictionRequest]] = {}
        for request in requests:
            grouped.setdefault(request.split_time, []).append(request)

        for split_time, group in grouped.items():
            try:
                result = self._predict_remaining_days_group(group, split_time)
            except Exception as exc:
                logger.exception(
                    "Failed to process prediction batch (split_time={}, requests={})",
                    split_time,
                    len(group),
                )
                for request in group:
                    request.error = exc
                    request.done.set()
                continue

            for request in group:
                request.result = result[result.index.isin(request.lot_ids)]
                request.done.set()

    def _predict_remaining_days_group(
        self,
        requests: list[_PredictionRequest],
        split_time: pd.Timestamp,
    ) -> pd.Series:
        raw_frames: list[pd.DataFrame] = []
        lot_ids: list[str] = []
        batch_size_values: list[int] = []
        for request in requests:
            raw_frames.append(self._snapshots_to_pandas(request.lot_snapshots))
            lot_ids.extend(request.lot_ids)
            if request.batch_size is not None:
                batch_size_values.append(int(request.batch_size))

        combined_snapshots = pd.concat(raw_frames, ignore_index=True)
        combined_snapshots = combined_snapshots.drop_duplicates(ignore_index=True)
        unique_lot_ids = list(dict.fromkeys(lot_ids))
        df_transformed = self.transform_snapshots(combined_snapshots, split_time)
        if df_transformed.empty:
            return _empty_prediction_result()
        batch_size = max(batch_size_values) if batch_size_values else None
        return self._predict_transformed_now(df_transformed, unique_lot_ids, batch_size=batch_size)

    @staticmethod
    def _snapshots_to_pandas(lot_snapshots: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        if isinstance(lot_snapshots, pl.DataFrame):
            return lot_snapshots.to_pandas()
        return lot_snapshots.copy()


_SERVICE_LOCK = Lock()
_SERVICE_CACHE: dict[str, WipInferenceService] = {}


def get_inference_service(model_dir: str | Path = DEFAULT_MODEL_DIR) -> WipInferenceService:
    key = Path(model_dir).as_posix()
    service = _SERVICE_CACHE.get(key)
    if service is not None:
        return service

    with _SERVICE_LOCK:
        service = _SERVICE_CACHE.get(key)
        if service is None:
            service = WipInferenceService(model_dir)
            _SERVICE_CACHE[key] = service
        return service


@lru_cache(maxsize=8)
def _load_prediction_config(model_dir: str) -> dict:
    config_path = Path(model_dir) / "config.yaml"
    with config_path.open("r") as f:
        return cast(dict, yaml.safe_load(f))


def prediction_requires_facility_context(model_dir: str | Path = DEFAULT_MODEL_DIR) -> bool:
    """Return whether inference needs non-target lot rows for set-level context."""
    try:
        config = _load_prediction_config(Path(model_dir).as_posix())
    except Exception as e:
        logger.warning("Failed to read prediction config from {}: {}", model_dir, e)
        return True
    return config.get("dataset", {}).get("type") != "lotwise"


def prediction_extra_feat_cols(model_dir: str | Path = DEFAULT_MODEL_DIR) -> list[str]:
    try:
        config = _load_prediction_config(Path(model_dir).as_posix())
    except Exception as e:
        logger.warning(f"Failed to read Prediction Config from {model_dir}: {e}")
        return []
    return _extra_feature_cols_from_embd(config.get("embed", {}))


def clear_inference_service_cache() -> None:
    """Clear cached inference services; intended for tests and controlled reloads."""
    with _SERVICE_LOCK:
        _SERVICE_CACHE.clear()
    _load_prediction_config.cache_clear()


if __name__ == "__main__":
    from wip_pred.inference.model_loader import load_model_and_config
    from wip_pred.prep_data.normalize_data import normalize_df, transform_df

    # load everything except data
    logger.info("Start loading model, preprocessor and config...")
    ckpt_dir = "checkpoints/20260719_211536"
    model, preprocessor, config = load_model_and_config(ckpt_dir)
    logger.info("Done!")

    # load data
    logger.info("Start loading data...")
    # Read data
    df = fetch_raw_data(
        embed_config=config["embed"]
    )  # if test set is needed, truncate df by time here

    # Preprcoess data
    df = normalize_df(
        df,
        "lot_id",
        "parent_lot_id",
        "product_id",
        "stage",
        "lot_status",
        "current_wafer_qty",
        "created_at",
        do_time_binning=True,
        normalize_by_date=config["embed"]["use_given_emb"],
        print_details=False,
    )
    logger.info("Done!")

    # Transform data
    logger.info("Start preprocessing data...")
    pred_time = pd.Timestamp("2024-09-01")
    df = transform_df(df, config["embed"], preprocessor=preprocessor, split_time=pred_time)
    logger.info("Done!")

    # Predict
    logger.info("Predicting...")
    pred_val = predict(
        model,
        df,
        config,
        ["XZu45lELkBzX39IM5Hou7g==", "7d5i/lPaEGTZAI0KHbDaJA==", "IR63q/qBv4KBV59d9g1IZQ=="],
    )
    logger.info("Done!")
    print("Result:\n", pred_val)
