# src/training/train.py

from __future__ import annotations
from typing import Dict, Callable, Optional

import torch
from torch.amp import GradScaler, autocast

from src.losses import get_loss
from src.training.forward import forward_batch
from src.utils.device_utils import move_to_device

__all__ = ["train_one_epoch"]


def train_one_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device,
        loss_fn: Callable,
        scaler: Optional[GradScaler] = None,
        max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    model.to(device)

    # Training variables
    running: Dict[str, float] = {}
    step_count = 0

    # Train a batch
    for step, batch in enumerate(dataloader):
        if (max_batches is not None) and (step >= max_batches):
            break

        step_count += 1

        batch = move_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast(device_type=device.type):
                outputs = forward_batch(model, batch)
                loss_dict = loss_fn(outputs, batch)
                loss_total = loss_dict["total_loss"]

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = forward_batch(model, batch)
            loss_dict = loss_fn(outputs, batch)
            loss_total = loss_dict["total_loss"]

            loss_total.backward()
            optimizer.step()

        for k, v in loss_dict.items():
            if k not in running:
                running[k] = 0.0
            running[k] += v.item()

    if step_count == 0:
        return {k: 0.0 for k in running.keys()}

    return {k: v / step_count for k, v in running.items()}
