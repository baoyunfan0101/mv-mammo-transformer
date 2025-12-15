# src/losses/softmax_gradcam_loss.py

from __future__ import annotations
from typing import Dict, Any, Optional

import torch

from src.data.bbox import BBox
from src.losses import register_loss, get_loss
from src.losses.composite_loss import CompositeLoss
from src.losses.gradcam_loss import GradCAMLoss

__all__ = ["SoftmaxGradCAMLoss"]


@register_loss("softmax+gradcam")
class SoftmaxGradCAMLoss:
    def __init__(
            self,
            *,
            weights: Optional[Dict[str, Dict[int, float]]] = None,
            lambda_density: float = 0.2,
            smoothing: float = 0.0,
            # GradCAM related
            epoch: int = 0,
            model: Optional[torch.nn.Module] = None,
            target_layer: Optional[torch.nn.Module] = None,
            bbox: Optional[BBox] = None,
            gradcam_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Base loss
        self.base_loss = get_loss(
            "softmax_lsce",
            weights=weights,
            lambda_density=lambda_density,
            smoothing=smoothing,
        )

        # GradCAM loss
        self.gradcam_loss = None
        if gradcam_kwargs is not None and model is not None:
            start_epoch = gradcam_kwargs.get("start_epoch", 0)
            if epoch >= start_epoch:
                self.gradcam_loss = GradCAMLoss(
                    model=model,
                    weight=gradcam_kwargs.get("weight", 1.0),
                    target_layer=target_layer,
                    bbox=bbox,
                )

        self.loss = CompositeLoss(
            base_loss_fn=self.base_loss,
            gradcam_loss=self.gradcam_loss,
        )

    def __call__(
            self,
            output: Dict[str, Any],
            batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        return self.loss(output, batch)
