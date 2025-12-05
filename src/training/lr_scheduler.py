# src/training/lr_scheduler.py

from __future__ import annotations
from typing import Dict, Any

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    MultiStepLR,
    ReduceLROnPlateau,
)

__all__ = ["LRScheduler"]


class LRScheduler:

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            cfg: Dict[str, Any] | None,
            max_epoch: int,
    ):
        self.optimizer = optimizer
        self.cfg = cfg or {}
        self.max_epoch = max_epoch

        # Warmup config
        self.warmup_epochs = self.cfg.get("warmup_epochs", 0)

        # Scheduler name
        self.name = self.cfg.get("name", "cosine").lower()

        self.scheduler = self._build_scheduler()

    # Construct scheduler
    def _build_scheduler(self):
        name = self.name

        # Cosine Annealing
        if name == "cosine":
            eta_min = self.cfg.get("eta_min", 1e-6)
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_epoch - self.warmup_epochs,
                eta_min=eta_min,
            )

        # StepLR
        if name == "step":
            step_size = self.cfg.get("step_size", 5)
            gamma = self.cfg.get("gamma", 0.1)
            return StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        # MultiStepLR
        if name == "multistep":
            milestones = self.cfg.get("milestones", [10, 20, 30])
            gamma = self.cfg.get("gamma", 0.1)
            return MultiStepLR(self.optimizer, milestones, gamma)

        # ReduceLROnPlateau
        if name == "plateau":
            factor = self.cfg.get("factor", 0.1)
            patience = self.cfg.get("patience", 5)
            return ReduceLROnPlateau(
                self.optimizer,
                factor=factor,
                patience=patience,
                verbose=True,
            )

        raise ValueError(f"[LRScheduler] Unknown scheduler name: {name}")

    # Warmup logic
    def _apply_warmup(self, epoch: int):
        # Warmup finished
        if epoch > self.warmup_epochs:
            return False

        warm_ratio = epoch / max(1, self.warmup_epochs)
        for group in self.optimizer.param_groups:
            base_lr = group.get("initial_lr", group["lr"])
            group["lr"] = base_lr * warm_ratio
        return True

    # Public API: step()
    def step(self, epoch: int, metrics: float | None = None):
        # Apply warmup if needed
        if self.warmup_epochs > 0:
            if self._apply_warmup(epoch):
                return  # warmup takes full control

        # Normal schedulers
        if self.name == "plateau":
            # Plateau scheduler requires validation metric
            if metrics is None:
                raise ValueError("[LRScheduler] Plateau scheduler needs a 'metrics' argument.")
            self.scheduler.step(metrics)
        else:
            # Cosine / Step / MultiStep
            self.scheduler.step()

    # Print current LR
    def get_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]
