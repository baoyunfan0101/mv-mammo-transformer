# scripts/train.py

import argparse
import os
import yaml
import torch

from src.data.status import Status
from src.data.bbox import BBox
from src.data.breast_level import BreastLevel
from src.models import get_model
from src.training.dataloader import build_dataset, build_dataloader
from src.training.collate import CollateFn
from src.training.freeze_scheduler import BackboneFreezeScheduler
from src.training.lr_scheduler import LRScheduler
from src.training.trainer import Trainer
from src.transforms import get_transform
from src.utils.device_utils import get_device
from src.utils.config_utils import apply_overrides
from src.utils.log_utils import log, log_section, log_config

from config import ENV, PROJECT_ROOT, RAW_DATA_DIR, CONFIG_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Train MV-Mammo-Transformer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML.")
    parser.add_argument("overrides", nargs="*", help="Override YAML.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    # Log environment
    log_section("Environment")
    log(f"Detected environment: {ENV}", "Environment")
    log(f"Loaded project root: {PROJECT_ROOT}", "ProjectRoot")
    log(f"Loaded raw data dir: {RAW_DATA_DIR}", "RawDataDir")
    log(f"Using device: {device}", "Device")

    # Load YAML
    with open(os.path.join(CONFIG_DIR, args.config), "r") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    config = apply_overrides(config, args.overrides)

    # Data objects
    status = Status()
    bbox = BBox()
    breast_level = BreastLevel()

    model_cfg = config["model"]
    data_cfg = config["data"]
    training_cfg = config["training"]

    # Model
    model = get_model(
        name=model_cfg["name"],
        backbone=model_cfg["backbone"],
        head=model_cfg["head"],
        in_chans=model_cfg["in_chans"],
        birads_classes=model_cfg["birads_classes"],
        density_classes=model_cfg["density_classes"],
        backbone_kwargs=model_cfg.get("backbone_kwargs", {}),
        head_kwargs=model_cfg.get("head_kwargs", {}),
        **model_cfg.get("model_kwargs", {}),
    ).to(device)

    # Datasets
    train_dataset = build_dataset(
        status=status,
        breast_level=breast_level,
        data_version=data_cfg["version"],
        splits=data_cfg["train_splits"],
        mode=data_cfg["mode"],
        with_labels=True,
        label_columns=data_cfg["label_columns"],
        label_mapping=data_cfg["label_mapping"],
    )

    val_dataset = build_dataset(
        status=status,
        breast_level=breast_level,
        data_version=data_cfg["version"],
        splits=data_cfg["val_splits"],
        mode=data_cfg["mode"],
        with_labels=True,
        label_columns=data_cfg["label_columns"],
        label_mapping=data_cfg["label_mapping"],
    )

    # Transforms
    transform = get_transform(
        name=data_cfg["transform"],
        **data_cfg.get("transform_kwargs", {}),
    )
    default_transform = get_transform(
        name="mv_test" if data_cfg["mode"] == "multi" else "sv_test",
        **data_cfg.get("transform_kwargs", {}),
    )

    # Dataloaders
    train_loader = build_dataloader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=data_cfg["shuffle"],
        num_workers=data_cfg["num_workers"],
        collate_fn=CollateFn(transform),
    )

    val_loader = build_dataloader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        collate_fn=CollateFn(default_transform),
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        [{
            "params": model.parameters(),
            "lr": training_cfg["lr"],
            "initial_lr": training_cfg["lr"]
        }],
        weight_decay=training_cfg["weight_decay"],
    )

    # Schedulers
    lr_scheduler = LRScheduler(
        optimizer,
        training_cfg.get("lr_schedule"),
        max_epoch=training_cfg["max_epoch"],
    )

    freeze_scheduler = BackboneFreezeScheduler(
        backbone=model.backbone,
        schedule_cfg=training_cfg.get("freeze_schedule", []),
    )

    # trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        loss_kwargs=training_cfg["loss_kwargs"],
        weights=train_dataset.get_class_weight(),
        max_epoch=training_cfg["max_epoch"],
        lr_scheduler=lr_scheduler,
        freeze_scheduler=freeze_scheduler,
        use_amp=training_cfg.get("use_amp", True),
        bbox=bbox,
    )

    log_section("Training Start")

    best_birads_acc = -1.0
    best_epoch = -1

    save_dir = training_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(1, training_cfg["max_epoch"] + 1):
        log_section(f"Epoch [{ep}/{training_cfg['max_epoch']}]")

        result = trainer.fit_one_epoch()

        log_config(result, "Metrics")

        birads_acc = result["val"]["acc"]["breast_birads"]

        if birads_acc > best_birads_acc:
            best_birads_acc = birads_acc
            best_epoch = ep

            ckpt_name = (
                f"{model_cfg['name']}"
                f"_ep{ep}-{training_cfg['max_epoch']}"
                f"_acc{birads_acc:.4f}.pt"
            )
            ckpt_path = os.path.join(save_dir, ckpt_name)
            torch.save(
                {
                    "epoch": ep,
                    "config": config,
                    "model_state": model.state_dict(),
                },
                ckpt_path,
            )

            log(
                f"Updated best model (epoch={ep}, birads_acc={birads_acc:.4f})",
                "Checkpoint",
            )

    log_section("Training Finished")
    log(
        f"Best breast_birads acc = {best_birads_acc:.4f} "
        f"at epoch {best_epoch}",
        tag="Summary",
    )


if __name__ == "__main__":
    main()
