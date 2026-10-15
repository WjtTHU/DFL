"""Resolve local CSV, configuration and output paths independently of the shell cwd."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PATH_CONFIG = PROJECT_ROOT / "config" / "local_paths.json"
DATASETS = ("cp", "dk", "hb")


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def dataset_paths(dataset: str) -> dict:
    with PATH_CONFIG.open(encoding="utf-8") as stream:
        entries = json.load(stream)["datasets"]
    key = dataset.lower()
    if key not in entries:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {', '.join(entries)}.")
    entry = entries[key]
    return {
        name: project_path(value) if name != "plot" else value
        for name, value in entry.items()
    }


def configure_imports() -> None:
    for path in reversed((PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "training", PROJECT_ROOT)):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def load_configs(dataset: str) -> tuple[dict, dict]:
    import yaml

    paths = dataset_paths(dataset)
    with (paths["config_dir"] / "core_template.yaml").open(encoding="utf-8") as stream:
        core = yaml.safe_load(stream)
    with (paths["config_dir"] / "train_template.yaml").open(encoding="utf-8") as stream:
        training = yaml.safe_load(stream)
    # Keep the supplied YAML unchanged; route runtime artifacts by dataset.
    training["training"]["save_dir"] = str(paths["checkpoint_dir"])
    training["training"]["log_dir"] = str(paths["experiment_dir"])
    return core, training
