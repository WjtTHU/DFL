import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from wip_pred.datasets.get_dataset import get_dataset
from wip_pred.models.embed import Embedding
from wip_pred.training.warmup_scheduler import GradualWarmupScheduler


def build_loss(config):
    loss_type = config["loss"]["type"]
    loss = getattr(nn, loss_type)(reduction="none")
    return loss


def build_optimizer(model, config):
    opt_type = config["optimizer"]["type"]
    params = config["optimizer"]["params"]
    optimizer = getattr(optim, opt_type)(model.parameters(), **params)
    return optimizer


def build_scheduler(optimizer, config):
    sch_type = config["scheduler"]["type"]
    params = config["scheduler"]["params"]
    warmup_epoch = config["scheduler"].get("warmup_epoch", 0)
    if warmup_epoch:
        after_scheduler = getattr(optim.lr_scheduler, sch_type)(optimizer, **params)
        scheduler = GradualWarmupScheduler(
            optimizer, warmup_epoch=warmup_epoch, after_scheduler=after_scheduler
        )
    else:
        scheduler = getattr(optim.lr_scheduler, sch_type)(optimizer, **params)
    return scheduler


def build_training_components(
    core_config: dict, train_config: dict, pretrained_weights: dict | None, device: str
):
    embedding = Embedding(
        core_config["model"]["params"]["d_model"], **core_config["embed"], device=device
    )
    if core_config["model"]["type"] == "tsf":
        from wip_pred.models.transformer import TSTransformerEncoderRegressor

        model = TSTransformerEncoderRegressor(
            embedding=embedding, **core_config["model"]["params"], device=device
        )
    elif core_config["model"]["type"] == "setseq":
        from wip_pred.models.setseq_transformer import SetSeqModel

        model = SetSeqModel(embedding=embedding, **core_config["model"]["params"], device=device)
    elif core_config["model"]["type"] in ["lstm", "gru", "rnn"]:
        from wip_pred.models.rnns import TSRNNEncoderRegressor

        model = TSRNNEncoderRegressor(
            embedding=embedding,
            **core_config["model"]["params"],
            model_type=core_config["model"]["type"],
            device=device,
        )
    else:
        raise ValueError(
            "Model Types other than `tsf`, `setseq`, `lstm`, `gru`, `rnn` are not supported yet."
        )

    if pretrained_weights:
        model_dict = model.state_dict()
        pretrained_weights = {
            k: v
            for k, v in pretrained_weights.items()
            if k in model_dict and v.size() == model_dict[k].size()
        }
        model_dict.update(pretrained_weights)
        model.load_state_dict(model_dict)

    criterion = build_loss(config=train_config)
    optimizer = build_optimizer(model, config=train_config)
    scheduler = build_scheduler(optimizer, config=train_config)

    return model, criterion, optimizer, scheduler


def get_train_dataset(df: pd.DataFrame, df_of_completed_lots: pd.DataFrame, core_config: dict):
    # TODO: you can change min_pred_time and max_pred_time to control the strictness on prediction time
    df_of_completed_lots["min_pred_time"] = df_of_completed_lots["start_time"]
    df_of_completed_lots["max_pred_time"] = df_of_completed_lots["end_time"] - pd.Timedelta(
        days=1.0
    )
    train_dataset = get_dataset(
        df,
        core_config,  # type: ignore
        df_of_completed_lots=df_of_completed_lots,
        predict_mode=False,
    )

    return train_dataset


def get_val_dataset(
    df: pd.DataFrame,
    df_of_completed_lots: pd.DataFrame,
    core_config: dict,
    val_start_time: pd.Timestamp,
    val_end_time: pd.Timestamp | None = None,
):
    if val_end_time:
        df = df[df["created_at"] <= val_end_time]

    # If you want to optimize performance per day per lot:
    # df_of_completed_lots['min_pred_time'] = df_of_completed_lots['start_time']
    # df_of_completed_lots['max_pred_time'] = df_of_completed_lots['end_time'] - pd.Timedelta(days=1.0)

    # If you want to optimize performance at 1/3 LT of each lot:
    df_of_completed_lots["min_pred_time"] = df_of_completed_lots[
        "start_time"
    ] + df_of_completed_lots["LT"] * 1 / 3 * pd.Timedelta(days=1.0)
    df_of_completed_lots["max_pred_time"] = np.minimum(
        df_of_completed_lots["end_time"] - pd.Timedelta(days=1.0),
        df_of_completed_lots["start_time"]
        + df_of_completed_lots["LT"] * 1 / 3 * pd.Timedelta(days=1.0)
        + pd.Timedelta(days=1.0),
    )  # type: ignore

    df_of_completed_lots = df_of_completed_lots[
        df_of_completed_lots["end_time"] > val_start_time
    ]  # exclude lots end before val_start_time (these labels are obtained in the train dataset)

    step_days = core_config["dataset"]["params"]["step_days"]
    core_config["dataset"]["params"]["step_days"] = 1
    val_dataset = get_dataset(
        df,
        core_config,
        df_of_completed_lots=df_of_completed_lots,  # type: ignore
        predict_mode=False,
        target_scaler=(core_config["target_scaler"]["min"], core_config["target_scaler"]["max"]),
        gamma=0.0,
    )
    core_config["dataset"]["params"]["step_days"] = step_days

    return val_dataset
