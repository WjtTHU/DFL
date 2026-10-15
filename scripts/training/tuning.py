import copy
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

from wip_pred.data_loader.fetch_data import fetch_raw_data
from wip_pred.prep_data.normalize_data import normalize_df, transform_df
from wip_pred.prep_data.prepare_data import get_df_of_completed_lots
from wip_pred.training.helper import (
    build_training_components,
    get_train_dataset,
    get_val_dataset,
)
from wip_pred.training.trainers.reg_trainer import RegTrainer
from wip_pred.utils.helper import seed_everything


def set_nested(config: Dict[str, Any], dotted_key: str, value: Any):
    """Set a nested dictionary key using dot notation (e.g., 'model.d_model')."""
    keys = dotted_key.split(".")
    d = config
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def sync_back(merged, original):
    for key, value in original.items():
        if key in merged:
            if isinstance(value, dict) and isinstance(merged[key], dict):
                sync_back(merged[key], value)
            else:
                original[key] = merged[key]


def prepare_datasets(core_config, train_config, train_end, val_end, df=None):
    # Read data
    if df is None:
        df = fetch_raw_data(
            embed_config=core_config["embed"],
            snapshot_start=train_config["data"]["range"]["snapshot_start"],
            snapshot_end=train_config["data"]["range"]["snapshot_end"],
        )  # if test set is needed, truncate df by validation end time here

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
            normalize_by_date=core_config["embed"]["use_given_emb"],
            print_details=False,
        )

    df_of_completed_lots = get_df_of_completed_lots(
        df,
        margin_in_days=train_config["data"].get("margin_in_days", 3.0),
    )

    # Split and transform data for input
    df_train, preprocessor = transform_df(df, core_config["embed"], split_time=train_end)
    df = transform_df(df, core_config["embed"], preprocessor=preprocessor)

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

    return train_dataset, val_dataset


def run_training(
    basic_config,
    seed,
    trial=None,
    train_dataset=None,
    val_dataset=None,
    df=None,
    train_end=None,
    val_end=None,
    rebuild_datasets=False,
):
    """
    Runs a complete training and evaluation session.
    - If 'trial' is not None, it is used for pruning.
    - If 'trial' is None, it runs as a standard validation (for Top-K evaluation).
    - If rebuild_datasets=True and df is provided, datasets are rebuilt each run (for dataset hyperparam tuning).
    """

    # Set the seed for this specific run
    seed_everything(seed)

    # Setup device, load data
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not train_end:
        train_end = pd.Timestamp(basic_config["data"]["cuts"]["train_end"])

    if not val_end:
        val_end = pd.Timestamp(basic_config["data"]["cuts"]["val_end"])

    if rebuild_datasets and df is not None:
        train_dataset, val_dataset = prepare_datasets(
            basic_config,
            train_config,
            train_end=train_end,
            val_end=val_end,
            df=df,
        )
    elif train_dataset is None or val_dataset is None:
        train_dataset, val_dataset = prepare_datasets(
            basic_config,
            train_config,
            train_end=train_end,
            val_end=val_end,
            df=df,
        )

    # train_loader, valid_loader = load_data(config)
    print(f"  [RUNNING] Seed: {seed} | Trial: {trial.number if trial else 'Validation'}")

    # Instantiate Model, Optimizer, Loss Function
    train_loader = train_dataset.to_dataloader(batch_size=basic_config["training"]["batch_size"])
    val_loader = val_dataset.to_dataloader(batch_size=basic_config["training"]["batch_size"])

    model, criterion, optimizer, scheduler = build_training_components(
        basic_config, basic_config, pretrained_weights=None, device=device
    )

    trainer = RegTrainer(
        model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics={"MAE": mean_absolute_error},
        target_scaler=(train_dataset.min_target, train_dataset.max_target),
        device=device,
        use_logger=False,
    )

    best_val_MAE = float("inf")

    # Training Loop
    for epoch in range(30):
        trainer.train_one_epoch(train_loader)
        avg_loss, metric_results = trainer.validate(val_loader, collect_details=True)
        val_MAE = metric_results["MAE"]

        if val_MAE < best_val_MAE:
            best_val_MAE = val_MAE

        # Optuna Pruning Logic
        if trial:
            # Report the intermediate validation result to Optuna
            trial.report(val_MAE, epoch)
            # Check if the trial should be pruned
            if trial.should_prune():
                print(f"  [PRUNED] Trial {trial.number}, Epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

    # Return the best metric found
    print(f"  [FINISHED] Seed: {seed}, Best MAE: {best_val_MAE:.6f}")
    return best_val_MAE


def tune_stage(
    stage_name: str,
    base_config: Dict[str, Any],
    search_space_fn: Callable[[optuna.Trial], Dict[str, Any]],
    pruner,
    n_trials: int = 20,
    top_k: int = 5,
    n_repeats: int = 3,
    direction: str = "minimize",
    plot=False,
    train_dataset=None,
    val_dataset=None,
    df=None,
    train_end=None,
    val_end=None,
    rebuild_datasets=False,
    n_jobs_stage1: int = 1,
    n_jobs_stage2: int = 1,
) -> Dict[str, Any]:
    """
    Generic function to tune one stage.

    - search_space_fn: defines which params to tune
    - rebuild_datasets: if True, rebuild train/val datasets on every run_training call
    - Returns updated config merged with best parameters
    """

    if not rebuild_datasets and train_dataset is None and val_dataset is None:
        train_dataset, val_dataset = prepare_datasets(
            base_config,
            base_config,
            train_end=train_end,
            val_end=val_end,
            df=df,
        )

    def objective(trial: optuna.Trial):
        try:
            # Generate parameter dict
            params = search_space_fn(trial)
            assert params.keys() == trial.params.keys()
            cfg = copy.deepcopy(base_config)
            for k, v in params.items():
                set_nested(cfg, k, v)
            result = run_training(
                cfg,
                seed=trial.number,
                trial=trial,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                df=df,
                train_end=train_end,
                val_end=val_end,
                rebuild_datasets=rebuild_datasets,
            )
        except optuna.exceptions.TrialPruned:
            raise
        except AssertionError:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return float("inf")  # Return a bad metric for failed trials

        torch.cuda.empty_cache()
        return result

    # -------- Stage 1: Rough Search --------
    print(f"\n[Stage 1] Tuning stage: {stage_name}")
    study = optuna.create_study(direction=direction, pruner=pruner, study_name=stage_name)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs_stage1, gc_after_trial=True)

    df_trials = study.trials_dataframe()
    df_trials = df_trials[df_trials["state"] == "COMPLETE"].sort_values(
        "value", ascending=(direction == "minimize")
    )
    top_params = df_trials.head(top_k)

    print(f"\n[Stage 1] Top-{top_k} Candidates:")
    print(top_params[["number", "value"]])

    # -------- Stage 2: Fine-Search (parallelized) --------
    print(f"\n[Stage 2] Assessing Top-{top_k} (n_repeats={n_repeats}, n_jobs={n_jobs_stage2})")
    results_summary = []

    for _, row in top_params.iterrows():
        trial_number = int(row["number"])
        params = study.trials[trial_number].params

        cfg = copy.deepcopy(base_config)
        for k, v in params.items():
            set_nested(cfg, k, v)

        seeds = random.sample(range(0, n_repeats * 100), n_repeats)

        def _run_one_seed(s, _cfg=cfg):
            return run_training(
                _cfg,
                seed=s,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                df=df,
                train_end=train_end,
                val_end=val_end,
                rebuild_datasets=rebuild_datasets,
            )

        n_workers = min(n_repeats, n_jobs_stage2)
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_run_one_seed, s) for s in seeds]
                repeat_results = [f.result() for f in futures]
        else:
            repeat_results = [_run_one_seed(s) for s in seeds]

        mean_val = np.mean(repeat_results)
        std_val = np.std(repeat_results)
        results_summary.append((params, mean_val, std_val))

        print(f"Trial {trial_number}: mean={mean_val:.4f}, std={std_val:.4f}")

    # -------- Robust Params --------
    if not results_summary:
        print("\n[Stage 2] No completed trials; returning base config unchanged.")
        return copy.deepcopy(base_config)

    if direction == "minimize":
        best = min(results_summary, key=lambda x: (x[1] + 0.1 * x[2]))
    else:
        best = max(results_summary, key=lambda x: (x[1] - 0.1 * x[2]))

    best_params, best_mean, best_std = best
    print(f"\nBest Params:\nmean={best_mean:.4f}, std={best_std:.4f}\n{best_params}")

    tuned_config = copy.deepcopy(base_config)
    for k, v in best_params.items():
        set_nested(tuned_config, k, v)

    # plot
    if plot:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.show()

        try:
            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.show()
        except Exception as e:
            print(f"Cannot plot param importances: {e}")

        fig3 = optuna.visualization.plot_slice(study)
        fig3.show()

    return tuned_config


def tune(
    core_config,
    train_config,
    df,
    train_end,
    val_end,
    search_space,
    pruner,
    rebuild_datasets=False,
    n_jobs_stage1: int = 1,
    n_jobs_stage2: int = 1,
):
    base_config = core_config | train_config
    train_dataset = val_dataset = None
    if not rebuild_datasets:
        train_dataset, val_dataset = prepare_datasets(
            core_config,
            train_config,
            train_end=train_end,
            val_end=val_end,
            df=df,
        )

    tuned_config = tune_stage(
        "Opt dataset hyperparams",
        base_config,
        search_space,
        pruner,
        n_trials=10,
        top_k=3,
        n_repeats=5,
        plot=False,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        df=df,
        train_end=train_end,
        val_end=val_end,
        rebuild_datasets=rebuild_datasets,
        n_jobs_stage1=n_jobs_stage1,
        n_jobs_stage2=n_jobs_stage2,
    )

    sync_back(tuned_config, core_config)
    sync_back(tuned_config, train_config)


if __name__ == "__main__":
    import yaml

    # define the searching range here
    def search_stage(trial):
        return {
            # "dataset.params.window_days": trial.suggest_int(
            #     "dataset.params.window_days", 15, 100, step=5
            # ),
            # "dataset.params.step_days": trial.suggest_categorical(
            #     "dataset.params.step_days", [3, 5, 7, 10, 15, 20]
            # ),
            "training.batch_size": trial.suggest_categorical(
                "training.batch_size", [32, 64, 128, 256, 512, 1024]
            ),
            "optimizer.params.lr": trial.suggest_float("optimizer.params.lr", 1e-5, 1e-2, log=True),
            "optimizer.params.weight_decay": trial.suggest_float(
                "optimizer.params.weight_decay", 1e-6, 1e-2, log=True
            ),
        }  # Important: key and param name must be the same!

    with open("config/pred_template/core_template.yaml") as f:
        core_config = yaml.safe_load(f)
    with open("config/pred_template/train_template.yaml") as f:
        train_config = yaml.safe_load(f)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20, interval_steps=1)

    tune(
        core_config,
        train_config,
        df=None,
        train_end=None,
        val_end=None,
        search_space=search_stage,
        pruner=pruner,
        rebuild_datasets=True,  # set False when tuning only training/optimizer params
    )
    # You can write this tuned config back into the yaml file directly
