# src/training/evaluate.py

from __future__ import annotations
from typing import Dict, Any, Callable, Optional

import torch

from src.training.forward import forward_batch
from src.utils.device_utils import move_to_device

__all__ = ["evaluate_one_epoch"]


@torch.no_grad()
def evaluate_one_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        *,
        device: torch.device,
        loss_fn: Callable,
        max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    model.to(device)

    # Evaluating variables
    running: Dict[str, float] = {}
    correct: Dict[str, float] = {}
    total: Dict[str, int] = {}
    step_count = 0

    # Evaluate a batch
    for step, batch in enumerate(dataloader):
        if (max_batches is not None) and (step >= max_batches):
            break

        step_count += 1

        batch = move_to_device(batch, device)
        output = forward_batch(model, batch)

        # Accuracy
        for task, head_out in output.items():

            # skip tasks without label
            if task not in batch["label"]:
                continue

            # select score tensor
            if "alpha" in head_out:
                score = head_out["alpha"]
            else:
                score = head_out["logits"]

            # predicted class index
            pred_idx = torch.argmax(score, dim=1)

            # ground truth
            target = batch["label"][task]

            # init counters
            if task not in correct:
                correct[task] = 0
                total[task] = 0

            # count
            correct[task] += (pred_idx == target).sum().item()
            total[task] += target.numel()

        # Loss
        loss_dict = loss_fn(output, batch)
        for k, v in loss_dict.items():
            if k not in running:
                running[k] = 0.0
            running[k] += v.item()

    if step_count == 0:
        return {
            "acc": {
                k: 0.0 for k, v in correct.items()
            },
            "loss": {
                k: 0.0 for k, v in running.items()
            }
        }

    return {
        "acc": {
            k: v / total[k] for k, v in correct.items()
        },
        "loss": {
            k: v / step_count for k, v in running.items()
        }
    }
