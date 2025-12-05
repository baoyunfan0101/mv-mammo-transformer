# scripts/evaluate.py

from __future__ import annotations
from typing import Dict, Any, Optional

import os
import torch

from src.models import get_model
from src.training.dataloader import build_dataset, build_dataloader
from src.training.collate import CollateFn
from src.transforms import get_transform
from src.data.status import Status
from src.data.breast_level import BreastLevel
from src.dataio.image_dataset import ImageDataset
from src.dataio.index_provider import EvalIndexProvider
from src.evaluation.evaluator import evaluate_multitask
from src.evaluation.gradcam import GradCAM
from src.utils.device_utils import get_device, move_to_device

__all__ = ["evaluate"]


def _load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    model_cfg = config["model"]

    model = get_model(
        name=model_cfg["name"],
        backbone=model_cfg.get("backbone"),
        head=model_cfg.get("head"),
        in_chans=model_cfg.get("in_chans"),
        birads_classes=model_cfg["birads_classes"],
        density_classes=model_cfg["density_classes"],
        backbone_kwargs=model_cfg.get("backbone_kwargs", {}),
        head_kwargs=model_cfg.get("head_kwargs", {}),
        **model_cfg.get("model_kwargs", {}),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, config


def evaluate(
        checkpoint_path: str,
        *,
        gradcam_study_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[scripts.evaluate] Checkpoint not found: {checkpoint_path}")

    device = get_device()

    # Load model & config
    model, config = _load_model(checkpoint_path, device)
    model_cfg = config["model"]
    data_cfg = config["data"]

    # Build test dataloader (for metrics)
    dataset = build_dataset(
        status=Status(),
        breast_level=BreastLevel(),
        data_version=data_cfg["version"],
        splits=["test"],
        mode=data_cfg["mode"],
        with_labels=True,
        label_columns=data_cfg["label_columns"],
        label_mapping=data_cfg["label_mapping"],
    )

    transform = get_transform(
        name="mv_test" if data_cfg["mode"] == "multi" else "sv_test"
    )

    dataloader = build_dataloader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        collate_fn=CollateFn(transform),
    )

    # Metrics
    metrics = evaluate_multitask(
        model=model,
        dataloader=dataloader,
        device=device,
        birads_classes=model_cfg["birads_classes"],
        density_classes=model_cfg["density_classes"],
    )

    result: Dict[str, Any] = {
        "metrics": metrics,
        "gradcam": None,
    }

    # Grad-CAM
    if gradcam_study_id is not None:
        # EvalIndexProvider
        eval_index_provider = EvalIndexProvider(
            status=Status(),
            study_id_list=[gradcam_study_id]
        )

        # Dataset
        eval_dataset = ImageDataset(
            index_provider=eval_index_provider,
            image_provider=dataset.image_provider,
            label_provider=dataset.label_provider,
            mode=data_cfg["mode"],
        )

        transform = get_transform(
            name="mv_test" if data_cfg["mode"] == "multi" else "sv_test",
            **data_cfg.get("transform_kwargs", {}),
        )

        dataloader = build_dataloader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=CollateFn(transform),
        )

        # Get the first batch
        for batch in dataloader:
            # Move tensors to device
            batch = move_to_device(batch, device)

            gradcam = GradCAM(model)

            cams = gradcam(
                batch=batch,
                task="breast_birads",
                use_predicted_class=True,
            )

            result["gradcam"] = {
                "study_id": gradcam_study_id,
                "key": batch["key"][0],
                "images": batch["images"],
                "cams": cams,
            }

            break

    return result
