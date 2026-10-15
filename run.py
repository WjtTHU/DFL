"""Local entry point. `check` and `--help` work without ML dependencies."""

import argparse
import importlib
import importlib.util
import sys
from datetime import datetime

from local_paths import DATASETS, PROJECT_ROOT, configure_imports, dataset_paths, load_configs, project_path


RUNTIME_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "torch": "torch",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "loguru": "loguru",
    "polars": "polars",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "tensorboard": "tensorboard",
    "tqdm": "tqdm",
    "dotenv": "python-dotenv",
}


def missing_modules() -> list[str]:
    return [package for module, package in RUNTIME_MODULES.items() if importlib.util.find_spec(module) is None]


def check_paths(datasets: tuple[str, ...]) -> bool:
    valid = True
    for dataset in datasets:
        paths = dataset_paths(dataset)
        inputs = {
            "csv": paths["csv"],
            "training script": paths["train_script"],
            "core config": paths["config_dir"] / "core_template.yaml",
            "train config": paths["config_dir"] / "train_template.yaml",
            "pretrain config": paths["config_dir"] / "pretrain_template.yaml",
        }
        print(f"[{dataset.upper()}]")
        for name, path in inputs.items():
            exists = path.is_file()
            valid = valid and exists
            print(f"  {'OK' if exists else 'MISSING'} {name}: {path}")
        for name in ("checkpoint_dir", "experiment_dir", "log_dir"):
            print(f"  OUTPUT {name}: {paths[name]}")
    for name in ("src/wip_pred/inference/predict_tmp.py", "scripts/training/pretrain.py"):
        path = PROJECT_ROOT / name
        exists = path.is_file()
        valid = valid and exists
        print(f"{'OK' if exists else 'MISSING'} {path}")
    return valid


def training_module(dataset: str):
    configure_imports()
    return importlib.import_module(f"train_{dataset.upper()}_csv")


def run_training(args) -> None:
    from loguru import logger

    paths = dataset_paths(args.dataset)
    paths["log_dir"].mkdir(parents=True, exist_ok=True)
    sink = logger.add(
        str(paths["log_dir"] / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"),
        level="DEBUG",
        rotation="5GB",
    )
    try:
        core, training = load_configs(args.dataset)
        plot = paths["plot"] if args.plot is None else args.plot
        training_module(args.dataset).train(core, training, plot=plot)
    finally:
        logger.remove(sink)


def run_prediction(args) -> None:
    import pandas as pd

    from wip_pred.inference.model_loader import load_model_and_config
    from wip_pred.inference.predict_tmp import predict
    from wip_pred.prep_data.normalize_data import transform_df

    checkpoint = project_path(args.checkpoint)
    for name in ("config.yaml", "best_model.pth", "preprocessor.pkl"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    model, preprocessor, config = load_model_and_config(str(checkpoint))
    # Reuse the selected CSV training script's exact preparation procedure.
    snapshots = training_module(args.dataset).load_snapshots(config)
    transformed = transform_df(
        snapshots, config["embed"], preprocessor=preprocessor, split_time=pd.Timestamp(args.at)
    )
    result = predict(model, transformed, config, args.lot)
    print("Remaining days (original model output):")
    print(result.to_string())


def main(argv=None) -> int:
    # Preserve Chinese paths in both Windows terminals and captured output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Local CSV paths, training and inference.")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Check paths without loading CSV data or a model.")
    check.add_argument("--dataset", choices=DATASETS)
    check.add_argument("--environment", action="store_true", help="Also check this Python environment.")
    train = sub.add_parser("train", help="Train with a supplied dataset configuration.")
    train.add_argument("--dataset", choices=DATASETS, required=True)
    train.add_argument("--plot", action=argparse.BooleanOptionalAction, default=None)
    prediction = sub.add_parser("predict", help="Predict with a matching trained checkpoint.")
    prediction.add_argument("--dataset", choices=DATASETS, required=True)
    prediction.add_argument("--checkpoint", required=True, help="Absolute path or path relative to local_wip.")
    prediction.add_argument("--at", required=True, help="Prediction cutoff, e.g. 2026-06-01 06:58:26.")
    prediction.add_argument("--lot", action="append", required=True, help="Lot ID; repeat for multiple lots.")
    args = parser.parse_args(argv)

    datasets = (args.dataset,) if args.dataset else DATASETS
    if not check_paths(datasets):
        return 1
    if args.command == "check":
        if args.environment:
            missing = missing_modules()
            print(f"Python: {sys.executable}")
            print("Missing dependencies: " + (", ".join(missing) if missing else "none detected"))
            return int(bool(missing))
        return 0

    missing = missing_modules()
    if missing:
        parser.error("This Python environment is missing dependencies: " + ", ".join(missing))
    configure_imports()
    if args.command == "train":
        run_training(args)
    else:
        run_prediction(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
