from pandas import DataFrame


def get_dataset(df: DataFrame, config: dict, predict_mode: bool = False, **kwargs):
    embed_config = config["embed"]

    numeric_feat_cols = embed_config["numeric_feat_cols"]
    categorical_feat_cols = [col["name"] for col in embed_config["categorical_feat_cols"]]
    temporal_feat_cols = embed_config["temporal_feat_cols"]
    position_col = embed_config["position_col"]
    is_given_col = "is_given" if "is_given" in df.columns else None

    # make dataset
    if config["dataset"]["type"] == "fabwise":
        from wip_pred.datasets.fabwise_dataset import FabwiseRegDataset1

        dataset = FabwiseRegDataset1(
            df,
            numeric_feat_cols=numeric_feat_cols,
            categorical_feat_cols=categorical_feat_cols,
            temporal_feat_cols=temporal_feat_cols,
            position_col=position_col,
            predict_mode=predict_mode,
            is_given_col=is_given_col,
            **config["dataset"]["params"],
            **kwargs,
        )
    elif config["dataset"]["type"] == "lotwise":
        from wip_pred.datasets.lotwise_dataset import LotwiseTensorRegDataset

        dataset = LotwiseTensorRegDataset(
            df,
            numeric_feat_cols=numeric_feat_cols,
            categorical_feat_cols=categorical_feat_cols,
            temporal_feat_cols=temporal_feat_cols,
            position_col=position_col,
            is_given_col=is_given_col,
            **config["dataset"]["params"],
            predict_mode=predict_mode,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Dataset type {config['dataset']['type']} is not supported yet. Supported types: `fabwise` and `lotwise`."
        )

    target_scaler = kwargs.get("target_scaler")
    if target_scaler is not None:
        actual_scaler = (dataset.min_target, dataset.max_target)
        if actual_scaler != tuple(target_scaler):
            raise ValueError(
                f"Dataset changed the supplied target scaler: "
                f"expected {tuple(target_scaler)}, got {actual_scaler}. "
                "Validation and prediction must reuse the training target scaler."
            )

    return dataset
