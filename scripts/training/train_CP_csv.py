#!/usr/bin/env python
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
from local_paths import configure_imports, dataset_paths

configure_imports()

import os
import pickle
from datetime import datetime

import pandas as pd
import torch
import yaml
from loguru import logger
from sklearn.metrics import mean_absolute_error

from wip_pred.prep_data.normalize_data import normalize_df, transform_df
from wip_pred.prep_data.prepare_data import get_df_of_completed_lots
from wip_pred.training.early_stopping import EarlyStopping
from wip_pred.training.helper import build_training_components, get_train_dataset, get_val_dataset
from wip_pred.utils.helper import seed_everything

from pretrain import pretrain


def load_snapshots(core_config: dict, print_normalize_details: bool = False):
    # df = fetch_raw_data()  # if test set is needed, truncate df by validation end time here
    df = pd.read_csv(dataset_paths("cp")["csv"])
    df = df.drop(columns=['process', 'start_date'])
    df["action_date"] = pd.to_datetime(df["action_date"], dayfirst=True)
    # df['stage'] = df['stage'].fillna('NULL')

    # Preprcoess data
    df = normalize_df(
        df,
        "LotId",
        "parentLotId",
        "device",
        "stage_id",
        "action_id",
        "wafer_id",
        "action_date",
        # truncate_stages=["QC1-PACK", "Inventory", "PA2-PHOTO", "PAD-PHOTO"],
        do_time_binning=True,
        normalize_by_date=core_config["embed"]["use_given_emb"],
        print_details=print_normalize_details,
    )
    logger.info(f"Data normalized, shape: {df.shape}")
    return df


def train(
    core_config: dict,
    train_config: dict,
    pretrain_config: dict | None = None,
    print_normalize_details: bool = False,
    create_timestamp_folder: bool = True,
    plot: bool = False,
    df: pd.DataFrame | None = None,
    train_end: pd.Timestamp | None = None,
    val_end: pd.Timestamp | None = None,
):
    """
    Args:
        core_config (dict): specify params of the dataset and model
        train_config (dict): specify params of training process
        pretrain_config (dict|None): specify params of pretraining process, if None, pretrain will not be used
        print_normalize_details (bool): whether print normalization details or not
        create_timestamp_folder (bool): whether use current timestamp to create a folder to save the logs and/or model
        plot (bool): whether plot prediction of each sample or not
        df (pd.DataFrame|None): You can specified normalized df instead of reading raw data from somewhere, this is useful for quick experiments because normalzed df is usually reusable under different configuration
        train_end (pd.Timestamp|None): You can specified train_end instead of reading from core_config.
        val_end (pd.Timestamp|None): You can specified val_end instead of reading from core_config.

    """

    # Read data
    if df is None:
        df = load_snapshots(core_config, print_normalize_details)

    df_of_completed_lots = get_df_of_completed_lots(df)
    logger.info(f"Got df of completed lots, shape: {df_of_completed_lots.shape}")

    # Test: add features
    # df['progress'] = (df.groupby('lot_id').cumcount() + 1) / df['lot_id'].map(df_of_completed_lots['LT'])
    # df['progress'] = df['progress'].fillna(df['progress'].mean())
    # core_config['embed']['numeric_feat_cols'].append('progress')

    # Split and transform data for input
    if not train_end:
        train_end = pd.Timestamp(train_config["data"]["cuts"]["train_end"])
    if not val_end:
        val_end = (
            pd.Timestamp(train_config["data"]["cuts"]["val_end"])
            if "val_end" in train_config["data"]["cuts"]
            else None
        )

    df_train, preprocessor = transform_df(df, core_config["embed"], split_time=train_end)
    df = transform_df(df, core_config["embed"], preprocessor=preprocessor)
    logger.info(f"Data transformed, train shape: {df_train.shape}, whole shape: {df.shape}")

    # Create datasets
    train_dataset = get_train_dataset(df_train, df_of_completed_lots, core_config)

    core_config["target_scaler"]["min"] = train_dataset.min_target
    core_config["target_scaler"]["max"] = train_dataset.max_target

    val_dataset = get_val_dataset(
        df,
        df_of_completed_lots,
        core_config,
        train_end,
        val_end,
    )

    assert len(val_dataset.lot_ids) > 0 and len(train_dataset.lot_ids) > 0

    seed_everything(train_config["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if pretrain_config:
        pretrained_weights = pretrain(train_dataset, core_config, pretrain_config, device)
    else:
        pretrained_weights = None

    train_loader = train_dataset.to_dataloader(batch_size=train_config["training"]["batch_size"])
    val_loader = val_dataset.to_dataloader(batch_size=train_config["training"]["batch_size"])

    model, criterion, optimizer, scheduler = build_training_components(
        core_config, train_config, pretrained_weights=pretrained_weights, device=device
    )

    log_dir = train_config["training"].get("log_dir", None)
    save_dir = train_config["training"].get("save_dir", None)

    if log_dir:
        if create_timestamp_folder:
            log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(log_dir)
        with open(os.path.join(log_dir, "config.yaml"), mode="w") as f:
            yaml.safe_dump(train_config, f)

    if save_dir:
        if create_timestamp_folder:
            save_dir = os.path.join(save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "config.yaml"), mode="w") as f:
            yaml.safe_dump(core_config, f)
        with open(os.path.join(save_dir, "preprocessor.pkl"), "wb") as f:
            pickle.dump(preprocessor, f)

    if "early_stopping" in train_config:
        early_stopping = EarlyStopping(
            **train_config["early_stopping"],
            path=os.path.join(save_dir or ".", "best_model.pth") if save_dir else save_dir,
        )
    else:
        early_stopping = None

    if plot:
        from wip_pred.training.trainers.visualizing_trainer import VisualizingTrainer

        if not log_dir:
            raise ValueError("When plot==True, you must pass log_dir to save the figures.")
        trainer = VisualizingTrainer(
            model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics={"MAE": mean_absolute_error},
            target_scaler=(train_dataset.min_target, train_dataset.max_target),
            device=device,
            log_dir=log_dir,
            save_dir=save_dir,
            early_stopping=early_stopping,
        )
        products_seen = df_of_completed_lots[
            df_of_completed_lots.index.isin(train_dataset.lot_ids)
        ]["product_id"].unique()
        lots_of_seen_prod = set(
            df_of_completed_lots[
                (df_of_completed_lots["product_id"].isin(products_seen))
                & (df_of_completed_lots.index.isin(val_dataset.lot_ids))
            ]
            .index.unique()
            .tolist()
        )
        id_groups = {
            "Seen product": lots_of_seen_prod,
            "Unknown Product": val_dataset.lot_ids - lots_of_seen_prod,
        }
        trainer.run(
            train_loader,
            val_loader,
            n_epoch=train_config["training"]["epochs"],
            id_groups=id_groups,
        )
    else:
        from wip_pred.training.trainers.reg_trainer import RegTrainer

        trainer = RegTrainer(
            model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics={"MAE": mean_absolute_error},
            target_scaler=(train_dataset.min_target, train_dataset.max_target),
            device=device,
            log_dir=log_dir,
            save_dir=save_dir,
            early_stopping=early_stopping,
        )

        trainer.run(
            train_loader,
            val_loader,
            n_epoch=train_config["training"]["epochs"],
            collect_train_details=True,
            collect_val_details=True,
        )

    model.load_state_dict(trainer.best_model_weights)

    return model, preprocessor


if __name__ == "__main__":
    from run import main

    raise SystemExit(main(["train", "--dataset", "cp", *sys.argv[1:]]))
