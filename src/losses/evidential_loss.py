# src/losses/evidential_loss.py

from __future__ import annotations
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F

from src.losses import register_loss, build_weight_tensor

__all__ = ["EvidentialCELoss", "EvidentialKLCELoss"]


def _check_evidential_output(
        output: Dict[str, Dict[str, torch.Tensor]]
):
    required = ["breast_birads", "breast_density"]

    for k in required:
        if k not in output:
            raise ValueError(
                f"[EvidentialLoss] Expected keys {required}, but got {list(output.keys())}. "
                f"This loss only supports MultiTaskEvidentialHead outputs."
            )

    for task in required:
        inner = output[task]
        if "alpha" not in inner:
            raise ValueError(
                f"[EvidentialLoss] Expected '{task}[\"alpha\"]' inside evidential head output."
            )


def expected_cross_entropy(
        alpha: torch.Tensor,
        target_onehot: torch.Tensor
) -> torch.Tensor:
    S = torch.sum(alpha, dim=1, keepdim=True)
    return torch.sum(target_onehot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)


def kl_dirichlet(
        alpha: torch.Tensor,
) -> torch.Tensor:
    K = alpha.size(1)
    S = torch.sum(alpha, dim=1, keepdim=True)

    log_norm = torch.lgamma(S) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    log_norm += torch.sum((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S)), dim=1, keepdim=True)

    return log_norm.squeeze(1)


def single_evidential_loss(
        alpha: torch.Tensor,
        target: torch.Tensor,
        kl_weight: float,
        weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_classes = alpha.size(1)
    target_onehot = F.one_hot(target, num_classes=num_classes).float()

    # Data term
    data_term = expected_cross_entropy(alpha, target_onehot)

    # Apply class weight only to data term
    if weight is not None:
        # Gather weight for each target class
        w = weight[target]
        data_term = data_term * w

    data_term = data_term.mean()

    # KL term
    kl_term = kl_dirichlet(alpha).mean()

    return data_term + kl_weight * kl_term


# Expected cross-entropy loss under the Dirichlet prior
def evidential_ce_loss(
        output: Dict[str, Any],
        batch: Dict[str, Any],
        weights: Optional[Dict[str, Dict[int, float]]] = None,
        lambda_density: float = 0.2,
        kl_weight: float = 1e-3,
):
    _check_evidential_output(output)

    alpha_b = output["breast_birads"]["alpha"]
    alpha_d = output["breast_density"]["alpha"]
    target_b = batch["label"]["breast_birads"]
    target_d = batch["label"]["breast_density"]
    if weights is not None:
        weight_b = build_weight_tensor(
            weights["breast_birads"],
            num_classes=alpha_b.shape[1],
            device=alpha_b.device,
            dtype=alpha_b.dtype,
        )
        weight_d = build_weight_tensor(
            weights["breast_density"],
            num_classes=alpha_d.shape[1],
            device=alpha_d.device,
            dtype=alpha_d.dtype,
        )
    else:
        weight_b = None
        weight_d = None

    loss_b = single_evidential_loss(alpha_b, target_b, kl_weight, weight_b)
    loss_d = single_evidential_loss(alpha_d, target_d, kl_weight, weight_d)

    # lambda of BI-RADS = 1, default lambda of density = 0.2
    total = loss_b + lambda_density * loss_d

    return {
        "birads_loss": loss_b,
        "density_loss": loss_d,
        "total_loss": total,
    }


@register_loss("evidential_ce")
class EvidentialCELoss:
    def __init__(
            self,
            weights: Optional[Dict[str, Dict[int, float]]] = None,
            lambda_density: float = 0.2,
            kl_weight: float = 1e-3,
    ):
        self.weights = weights
        self.lambda_density = lambda_density
        self.kl_weight = kl_weight

    def __call__(
            self,
            output: Dict[str, Any],
            batch: Dict[str, Any],
    ) -> torch.Tensor:
        return evidential_ce_loss(
            output,
            batch,
            self.weights,
            self.lambda_density,
            self.kl_weight
        )


# Expected cross-entropy loss with annealed KL-regularization weight
def evidential_klce_loss(
        output: dict,
        batch: Dict[str, Any],
        weights: Optional[Dict[str, Dict[int, float]]] = None,
        lambda_density: float = 0.2,
        epoch: int = 0,
        max_epochs: int = 100,
        base_weight: float = 1e-3,
):
    _check_evidential_output(output)

    alpha_b = output["breast_birads"]["alpha"]
    alpha_d = output["breast_density"]["alpha"]
    target_b = batch["label"]["breast_birads"]
    target_d = batch["label"]["breast_density"]
    if weights is not None:
        weight_b = build_weight_tensor(
            weights["breast_birads"],
            num_classes=alpha_b.shape[1],
            device=alpha_b.device,
            dtype=alpha_b.dtype,
        )
        weight_d = build_weight_tensor(
            weights["breast_density"],
            num_classes=alpha_d.shape[1],
            device=alpha_d.device,
            dtype=alpha_d.dtype,
        )
    else:
        weight_b = None
        weight_d = None

    kl_weight = base_weight * min(1.0, epoch / max_epochs)

    loss_b = single_evidential_loss(alpha_b, target_b, kl_weight, weight_b)
    loss_d = single_evidential_loss(alpha_d, target_d, kl_weight, weight_d)

    # lambda of BI-RADS = 1, default lambda of density = 0.2
    total = loss_b + lambda_density * loss_d

    return {
        "birads_loss": loss_b,
        "density_loss": loss_d,
        "total_loss": total,
        "kl_weight": torch.tensor(kl_weight),
    }


@register_loss("evidential_klce")
class EvidentialKLCELoss:
    def __init__(
            self,
            weights: Optional[Dict[str, Dict[int, float]]] = None,
            lambda_density: float = 0.2,
            epoch: int = 0,
            max_epochs: int = 100,
            base_weight: float = 1e-3,
    ):
        self.weights = weights
        self.lambda_density = lambda_density
        self.epoch = epoch
        self.max_epochs = max_epochs
        self.base_weight = base_weight

    def __call__(
            self,
            output: Dict[str, Any],
            batch: Dict[str, Any],
    ) -> torch.Tensor:
        return evidential_klce_loss(
            output,
            batch,
            self.weights,
            self.lambda_density,
            self.epoch,
            self.max_epochs,
            self.base_weight,
        )
