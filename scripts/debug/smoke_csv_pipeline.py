"""Exercise the enterprise CSV pipeline on complete Lot histories, not an accuracy benchmark."""

import argparse
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from local_paths import configure_imports, load_configs


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cp", "dk", "hb"), default="dk")
    parser.add_argument("--train-lots", type=int, default=32)
    parser.add_argument("--val-lots", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.train_lots, args.val_lots, args.epochs, args.threads) < 1:
        parser.error("Lot counts, epochs and threads must be positive.")

    configure_imports()
    import numpy as np
    import pandas as pd
    import torch
    from loguru import logger

    from wip_pred.inference.model_loader import load_model_and_config
    from wip_pred.inference.predict_tmp import predict
    from wip_pred.prep_data.normalize_data import transform_df
    from wip_pred.prep_data.prepare_data import get_df_of_completed_lots

    torch.set_num_threads(args.threads)
    module = importlib.import_module(f"train_{args.dataset.upper()}_csv")
    core, training = load_configs(args.dataset)
    train_end = pd.Timestamp(training["data"]["cuts"]["train_end"])
    val_end = pd.Timestamp(training["data"]["cuts"]["val_end"])
    seed = training["training"]["seed"]
    output = PROJECT_ROOT / "outputs" / args.dataset / "smoke" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    checkpoint = output / "checkpoint"
    logs = output / "experiments"
    checkpoint.mkdir(parents=True)
    logs.mkdir()
    sink = logger.add(str(output / "pipeline.log"), encoding="utf-8")
    started = time.monotonic()
    try:
        # Normalize the entire input with the enterprise entry point before selecting
        # complete Lot groups for a bounded functional test. No raw rows are rewritten.
        snapshots = module.load_snapshots(core, print_normalize_details=True)
        completed = get_df_of_completed_lots(snapshots)
        eligible = completed[completed["duration"] > 3].sort_index()
        train_candidates = eligible[eligible["end_time"] <= train_end]
        val_candidates = eligible[
            (eligible["start_time"] < train_end)
            & (eligible["end_time"] > train_end)
            & (eligible["end_time"] <= val_end)
        ]
        if train_candidates.empty or val_candidates.empty:
            raise ValueError("No eligible complete Lot histories on both sides of the supplied cutoff.")
        train_ids = train_candidates.sample(min(args.train_lots, len(train_candidates)), random_state=seed).index.tolist()
        val_ids = val_candidates.sample(min(args.val_lots, len(val_candidates)), random_state=seed).index.tolist()
        selected_ids = train_ids + val_ids
        subset = snapshots[snapshots["lot_id"].isin(selected_ids)].copy()

        # Only runtime and output settings are overridden. The enterprise core model,
        # feature definitions, time cutoffs and source YAML files remain unchanged.
        training["training"]["epochs"] = args.epochs
        training["training"]["batch_size"] = min(training["training"]["batch_size"], 32)
        training["training"]["save_dir"] = str(checkpoint)
        training["training"]["log_dir"] = str(logs)
        manifest = {
            "purpose": "Functional smoke test only; metrics do not establish model quality.",
            "dataset": args.dataset,
            "normalized_rows_full_input": len(snapshots),
            "normalized_lots_full_input": int(snapshots["lot_id"].nunique()),
            "completed_root_lots_full_input": len(completed),
            "selected_rows": len(subset),
            "selected_train_lots": train_ids,
            "selected_validation_lots": val_ids,
            "sampling": "Whole Lot histories, seed from enterprise config; duration > 3 days for this functional test only.",
            "epochs": args.epochs,
            "batch_size": training["training"]["batch_size"],
            "threads": args.threads,
            "python": sys.version,
            "torch": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "train_end": str(train_end),
            "val_end": str(val_end),
            "checkpoint": str(checkpoint),
        }
        (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Smoke test: {} rows from {} complete Lot histories", len(subset), len(selected_ids))
        model, _ = module.train(core, training, df=subset, create_timestamp_folder=False, plot=False)
        for filename in ("config.yaml", "best_model.pth", "preprocessor.pkl"):
            if not (checkpoint / filename).is_file():
                raise FileNotFoundError(checkpoint / filename)

        restored, preprocessor, saved_config = load_model_and_config(str(checkpoint))
        transformed = transform_df(subset, saved_config["embed"], preprocessor=preprocessor, split_time=train_end)
        prediction_ids = [lot for lot in val_ids if lot in set(transformed["lot_id"])][:5]
        if not prediction_ids:
            raise ValueError("No validation Lot has a snapshot before the prediction cutoff.")
        result = predict(restored, transformed, saved_config, prediction_ids)
        if set(result.index) != set(prediction_ids) or not np.isfinite(result.to_numpy()).all():
            raise ValueError("Inference did not produce one finite result for each requested Lot.")
        model.eval()
        in_memory = predict(model, transformed, core, prediction_ids)
        np.testing.assert_allclose(result.sort_index(), in_memory.sort_index(), rtol=1e-6, atol=1e-6)
        result.rename("predicted_remaining_days").rename_axis("lot_id").to_csv(output / "predictions.csv", encoding="utf-8-sig")
        manifest.update(status="passed", predictions=len(result), checkpoint_reload_matches=True,
                        elapsed_seconds=round(time.monotonic() - started, 2))
        (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "passed", "output": str(output), "checkpoint": str(checkpoint),
                          "prediction_cutoff": str(train_end), "prediction_lots": prediction_ids,
                          "elapsed_seconds": manifest["elapsed_seconds"]}, ensure_ascii=False, indent=2))
    finally:
        logger.remove(sink)


if __name__ == "__main__":
    main()
