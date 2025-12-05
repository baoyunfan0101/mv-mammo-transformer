# src/evaluation/gradcam.py

from __future__ import annotations
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataio.keys import ImageKey, MultiViewKey

__all__ = ["GradCAM"]


class GradCAM:

    def __init__(
            self,
            model: nn.Module,
            *,
            target_layer: Optional[nn.Module] = None,
    ):
        self.model = model
        self.target_layer = target_layer or self._infer_target_layer()

        # Buffers for features and gradients (view-aware)
        self._features: List[torch.Tensor] = []
        self._grads: List[torch.Tensor] = []

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            # output: (B, C, H, W)
            self._features.append(output)

        def backward_hook(_, grad_input, grad_output):
            # grad_output[0]: (B, C, H, W)
            self._grads.append(grad_output[0])

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def _infer_target_layer(self) -> nn.Module:
        backbone = getattr(self.model, "backbone", None)

        # CNN-based backbones: use the last Conv2d layer
        if hasattr(backbone, "body"):
            for m in reversed(list(backbone.body.modules())):
                if isinstance(m, nn.Conv2d):
                    return m

        # Swin Transformer: use patch embedding projection
        if hasattr(backbone, "patch_embed"):
            proj = getattr(backbone.patch_embed, "proj", None)
            if isinstance(proj, nn.Conv2d):
                return proj

            raise RuntimeError(
                "[GradCAM] Swin backbone found, but patch_embed.proj is not Conv2d."
            )

        raise RuntimeError(
            "[GradCAM] Cannot infer target_layer automatically for backbone "
            f"{type(backbone).__name__}. Please specify target_layer manually."
        )

    def __call__(
            self,
            *,
            batch: Dict[str, Any],
            task: str = "breast_birads",
            use_predicted_class: bool = True,
    ) -> List[torch.Tensor]:
        # Determine number of views
        images = batch.get("images", None)
        if isinstance(images, torch.Tensor):
            num_views = 1
        elif isinstance(images, (list, tuple)):
            num_views = len(images)
        else:
            raise RuntimeError(
                f"[GradCAM] Unsupported images type: {type(images)}"
            )

        # Reset buffers
        self._features.clear()
        self._grads.clear()

        # Forward
        output = self.model(batch["images"])
        logits = output[task]["logits"]

        # Select class for CAM score
        if use_predicted_class:
            target = logits.argmax(dim=1)
        else:
            target = batch["label"][task]

        # Backward score
        score = logits.gather(1, target.unsqueeze(1)).sum()

        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        n = min(len(self._features), len(self._grads), num_views)
        if n == 0:
            return []

        feats_list = self._features[-n:]
        grads_list = self._grads[-n:]

        cams: List[torch.Tensor] = []

        for features, grads in zip(feats_list, grads_list):
            # features / grads: (B, C, H, W)
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
            cam = (weights * features).sum(dim=1)  # (B, H, W)
            cam = F.relu(cam)
            cam = cam / (cam.amax(dim=(-2, -1), keepdim=True) + 1e-6)
            cams.append(cam.detach())

        return cams
