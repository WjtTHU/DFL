from wip_pred.models.embed import Embedding
from wip_pred.models.transformer import TSTransformerEncoder
from wip_pred.training.helper import build_optimizer, build_scheduler
from wip_pred.training.trainers.pretrainer import Pretrainer


def pretrain(train_dataset, core_config: dict, pretrain_config: dict, device: str):
    if not pretrain_config["dataset"]["extra_params"].get("noise_cols", None):
        raise ValueError("dataset.extra_params.noise_cols must be passed in pretrain config.")

    pretrain_loader = train_dataset.to_dataloader(
        superv=False, **pretrain_config["dataset"]["extra_params"]
    )

    embedding = Embedding(
        core_config["model"]["params"]["d_model"], **core_config["embed"], device=device
    )
    num_class_dict = dict()
    for col in core_config["embed"]["categorical_feat_cols"]:
        if col["name"] in pretrain_config["dataset"]["extra_params"]["noise_cols"]:
            num_class_dict[col["name"]] = col["num_classes"]

    model = TSTransformerEncoder(
        num_class_dict=num_class_dict,
        embedding=embedding,
        **core_config["model"]["params"],
        device=device,
    ).to(device)

    optimizer = build_optimizer(model, config=pretrain_config)
    scheduler = build_scheduler(optimizer, config=pretrain_config)

    trainer = Pretrainer(model, optimizer, device, scheduler)

    trainer.run(pretrain_loader, pretrain_config["pretraining"]["epochs"])

    return model.state_dict()
