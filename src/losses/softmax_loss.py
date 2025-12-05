# src/losses/softmax_loss.py

from __future__ import annotations
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F

from src.losses import register_loss, build_weight_tensor

__all__ = ["SoftmaxCELoss", "SoftmaxLSCELoss"]


def _check_softmax_output(output: Dict[str, torch.Tensor]):
    required = ["breast_birads", "breast_density"]
    for k in required:
        if k not in output:
            raise ValueError(
                f"[SoftmaxLoss] Expected keys {required}, but got {list(output.keys())}. "
                f"This loss only supports MultiTaskSoftmaxHead outputs."
            )


# Cross entropy
def softmax_ce_loss(
        output: dict,
        batch: Dict[str, Any],
        weights: Optional[Dict[str, Dict[int, float]]] = None,
        lambda_density: float = 0.2,
        smoothing: float = 0.0,
):
    _check_softmax_output(output)

    logits_b = output["breast_birads"]["logits"]
    logits_d = output["breast_density"]["logits"]
    target_b = batch["label"]["breast_birads"]
    target_d = batch["label"]["breast_density"]
    if weights is not None:
        weight_b = build_weight_tensor(
            weights["breast_birads"],
            num_classes=logits_b.shape[1],
            device=logits_b.device,
            dtype=logits_b.dtype,
        )
        weight_d = build_weight_tensor(
            weights["breast_density"],
            num_classes=logits_d.shape[1],
            device=logits_d.device,
            dtype=logits_d.dtype,
        )
    else:
        weight_b = None
        weight_d = None

    loss_b = F.cross_entropy(
        logits_b,
        target_b,
        weight=weight_b,
        label_smoothing=smoothing
    )
    loss_d = F.cross_entropy(
        logits_d,
        target_d,
        weight=weight_d,
        label_smoothing=smoothing
    )

    # lambda of BI-RADS = 1, default lambda of density = 0.2
    total = loss_b + lambda_density * loss_d

    return {
        "birads_loss": loss_b,
        "density_loss": loss_d,
        "total_loss": total,
    }


@register_loss("softmax_ce")
class SoftmaxCELoss:
    def __init__(
            self,
            weights: Optional[Dict[str, Dict[int, float]]] = None,
            lambda_density: float = 0.2,
            smoothing: float = 0.0,
    ):
        self.weights = weights
        self.lambda_density = lambda_density
        self.smoothing = smoothing

    def __call__(
            self,
            output: Dict[str, Any],
            batch: Dict[str, Any],
    ):
        return softmax_ce_loss(
            output=output,
            batch=batch,
            weights=self.weights,
            lambda_density=self.lambda_density,
            smoothing=self.smoothing,
        )


# Label smoothing cross entropy
def softmax_lsce_loss(
        output: dict,
        batch: Dict[str, Any],
        weights: Optional[Dict[str, Dict[int, float]]] = None,
        lambda_density: float = 0.2,
        smoothing: float = 0.1,
):
    return softmax_ce_loss(
        output=output,
        batch=batch,
        weights=weights,
        lambda_density=lambda_density,
        smoothing=smoothing,
    )


@register_loss("softmax_lsce")
class SoftmaxLSCELoss:
    def __init__(
            self,
            weights: Optional[Dict[str, Dict[int, float]]] = None,
            lambda_density: float = 0.2,
            smoothing: float = 0.1,
    ):
        self.loss = SoftmaxCELoss(
            weights=weights,
            lambda_density=lambda_density,
            smoothing=smoothing,
        )

    def __call__(
            self,
            output: Dict[str, Any],
            batch: Dict[str, Any]
    ):
        return self.loss(
            output=output,
            batch=batch,
        )
