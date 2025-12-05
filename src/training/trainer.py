# src/training/trainer.py

from __future__ import annotations
from typing import List, Dict, Any, Optional

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.data.bbox import BBox
from src.losses import get_loss
from src.training.train import train_one_epoch
from src.training.evaluate import evaluate_one_epoch


class Trainer:
    def __init__(
            self,
            *,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader],
            device: torch.device,
            loss_kwargs: Dict[str, Any],
            weights: Dict[str, Dict[int, float]],
            max_epoch: int,
            lr_scheduler=None,
            freeze_scheduler=None,
            use_amp: bool = False,
            bbox: Optional[BBox] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.loss_kwargs = loss_kwargs
        self.weights = weights
        self.max_epoch = max_epoch
        self.lr_scheduler = lr_scheduler
        self.freeze_scheduler = freeze_scheduler

        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None

        self.bbox = bbox

        self.epoch = 0
        self.model.to(device)

    def train_epoch(self) -> Dict[str, float]:
        loss_fn = get_loss(
            **self.loss_kwargs,
            weights=self.weights,
            model=self.model,
            bbox=self.bbox,
            epoch=self.epoch,
            max_epochs=self.max_epoch,
        )

        return train_one_epoch(
            model=self.model,
            dataloader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            loss_fn=loss_fn,
            scaler=self.scaler,
        )

    def eval_epoch(self) -> Optional[Dict[str, Any]]:
        if self.val_loader is None:
            return None

        loss_fn = get_loss(
            **self.loss_kwargs,
            weights=self.weights,
            model=self.model,
            bbox=self.bbox,
            epoch=self.epoch,
            max_epochs=self.max_epoch,
        )

        return evaluate_one_epoch(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            loss_fn=loss_fn,
        )

    def step_freeze_scheduler(self):
        if self.freeze_scheduler is not None:
            self.freeze_scheduler.apply(self.epoch)

    def step_lr_scheduler(self, val_result: Optional[Dict[str, Any]]):
        if self.lr_scheduler is not None:
            if val_result is not None:
                self.lr_scheduler.step(
                    epoch=self.epoch,
                    metrics=val_result["total_loss"],
                )
            else:
                self.lr_scheduler.step(
                    epoch=self.epoch,
                )

    def fit_one_epoch(self) -> Dict[str, Any]:
        self.step_freeze_scheduler()

        train_stats = self.train_epoch()

        self.step_lr_scheduler(train_stats)

        val_stats = self.eval_epoch()

        self.epoch += 1

        return {
            "epoch": self.epoch,
            "train": train_stats,
            "val": val_stats,
        }

    def fit(self, num_epochs: Optional[int] = None):
        if num_epochs is None:
            num_epochs = self.max_epoch

        running: List[Dict[str, Any]] = []
        while self.epoch < num_epochs:
            running.append(self.fit_one_epoch())
