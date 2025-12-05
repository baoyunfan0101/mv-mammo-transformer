# src/losses/composite_loss.py

from __future__ import annotations
from typing import Dict, Any, Optional

import torch

__all__ = ["CompositeLoss"]


class CompositeLoss:

    def __init__(
            self,
            base_loss_fn: Callable,
            *,
            gradcam_loss: Optional[object] = None,
    ):
        self.base_loss_fn = base_loss_fn
        self.gradcam_loss = gradcam_loss

    def __call__(
            self,
            output: Dict[str, Any],
            batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        loss_dict = self.base_loss_fn(output, batch)

        if self.gradcam_loss is not None:
            cam_loss = self.gradcam_loss(output, batch)

            loss_dict["gradcam_loss"] = cam_loss
            loss_dict["total_loss"] = loss_dict["total_loss"] + cam_loss

        return loss_dict
