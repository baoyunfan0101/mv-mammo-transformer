# src/evaluation/evaluator.py

from __future__ import annotations
from typing import Dict, Any, Union

import torch
import torch.nn.functional as F

from src.training.forward import forward_batch
from src.evaluation.metrics import compute_classification_metrics
from src.utils.device_utils import move_to_device

__all__ = ["evaluate_multitask"]


def _extract_probs_and_preds(outputs: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    is_evidential = False
    for output in outputs.values():
        if "alpha" in output:
            is_evidential = True
            break

    result: Dict[str, Dict[str, torch.Tensor]] = {}

    # Evidential head
    if is_evidential:
        for task, output in outputs.items():
            alpha = output["alpha"]

            S = alpha.sum(dim=1, keepdim=True)

            prob = alpha / S.clamp_min(1e-8)

            result[task] = {
                "prob": prob,
                "pred": prob.argmax(dim=1),
            }

    # Softmax head
    else:
        for task, output in outputs.items():
            logits = output["logits"]

            prob = F.softmax(logits, dim=1)

            result[task] = {
                "prob": prob,
                "pred": logits.argmax(dim=1),
            }

    return result


def evaluate_multitask(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: Union[str, torch.device],
        birads_classes: int = 5,
        density_classes: int = 4,
) -> Dict[str, Any]:
    model.eval()
    device = torch.device(device)

    birads_true = []
    birads_pred = []
    birads_prob = []

    density_true = []
    density_pred = []
    density_prob = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_to_device(batch, device)

            outputs = forward_batch(model, batch)
            parsed = _extract_probs_and_preds(outputs)

            # (B,)
            b_pred = parsed["breast_birads"]["pred"]
            d_pred = parsed["breast_density"]["pred"]

            # (B, C)
            b_prob = parsed["breast_birads"]["prob"]
            d_prob = parsed["breast_density"]["prob"]

            b_true = batch["label"]["breast_birads"]
            d_true = batch["label"]["breast_density"]

            birads_true.append(b_true.detach().cpu())
            birads_pred.append(b_pred.detach().cpu())
            birads_prob.append(b_prob.detach().cpu())

            density_true.append(d_true.detach().cpu())
            density_pred.append(d_pred.detach().cpu())
            density_prob.append(d_prob.detach().cpu())

    if len(birads_true) == 0:
        # Empty dataloader
        zero_b = {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "roc_auc_ovr_macro": 0.0,
            "cohen_kappa_quadratic": 0.0,
            "confusion_matrix": [[0] * birads_classes for _ in range(birads_classes)],
        }
        zero_d = {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "roc_auc_ovr_macro": 0.0,
            "cohen_kappa_quadratic": 0.0,
            "confusion_matrix": [[0] * density_classes for _ in range(density_classes)],
        }
        return {
            "birads": zero_b,
            "density": zero_d,
        }

    # Stack to numpy
    birads_true_np = torch.cat(birads_true).numpy()
    birads_pred_np = torch.cat(birads_pred).numpy()
    birads_prob_np = torch.cat(birads_prob).numpy()

    density_true_np = torch.cat(density_true).numpy()
    density_pred_np = torch.cat(density_pred).numpy()
    density_prob_np = torch.cat(density_prob).numpy()

    # Compute metrics
    birads_metrics = compute_classification_metrics(
        y_true=birads_true_np,
        y_pred=birads_pred_np,
        y_prob=birads_prob_np,
        num_classes=birads_classes,
    )

    density_metrics = compute_classification_metrics(
        y_true=density_true_np,
        y_pred=density_pred_np,
        y_prob=density_prob_np,
        num_classes=density_classes,
    )

    return {
        "birads": birads_metrics,
        "density": density_metrics,
    }
