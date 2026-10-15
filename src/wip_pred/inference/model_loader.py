import os
import pickle

import torch
import yaml
from loguru import logger

from wip_pred.models.embed import Embedding


def load_model_and_config(checkpoint_dir: str):
    """Load a model from a checkpoint directory."""
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
    preprocessor_path = os.path.join(checkpoint_dir, "preprocessor.pkl")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    max_key_len = max(len(key) for key in config.keys()) + 1
    for arg, val in config.items():
        logger.info(f"  {arg:<{max_key_len}} {val}")

    # Load preprocessor
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Handle compiled model checkpoints
    if any(key.startswith("_orig_mod.") for key in checkpoint.keys()):
        logger.info("Detected compiled model checkpoint, removing _orig_mod. prefixes")
        checkpoint = {key.replace("_orig_mod.", ""): value for key, value in checkpoint.items()}

    # Create model
    embedding = Embedding(config["model"]["params"]["d_model"], device=device, **config["embed"])
    if config["model"]["type"] == "tsf":
        from wip_pred.models.transformer import TSTransformerEncoderRegressor

        model = TSTransformerEncoderRegressor(
            embedding=embedding, **config["model"]["params"], device=device
        )
    elif config["model"]["type"] == "setseq":
        from wip_pred.models.setseq_transformer import SetSeqModel

        model = SetSeqModel(embedding=embedding, **config["model"]["params"], device=device)
    elif config["model"]["type"] in ("lstm", "gru", "rnn"):
        from wip_pred.models.rnns import TSRNNEncoderRegressor

        model = TSRNNEncoderRegressor(
            embedding=embedding,
            **config["model"]["params"],
            model_type=config["model"]["type"],
            device=device,
        )
    else:
        raise ValueError(
            "Model Types other than `tsf`, `setseq`, `lstm`, `gru`, `rnn` are not supported yet."
        )

    # Load state dict
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    logger.info(f"Loaded model checkpoint from {ckpt_path}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model, preprocessor, config
