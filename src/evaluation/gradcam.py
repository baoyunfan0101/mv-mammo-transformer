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

        # Buffers for features and gradients
        self._features: List[torch.Tensor] = []
        self._grads: List[torch.Tensor] = []

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            # output: (4B, C, H, W)
            self._features.append(output)

        def backward_hook(_, grad_input, grad_output):
            # grad_output[0]: (4B, C, H, W)
            self._grads.append(grad_output[0])

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def _infer_target_layer(self) -> nn.Module:
        backbone = getattr(self.model, "backbone", None)

        # CNN-based backbones: use the last Conv2d layer
        if hasattr(backbone, "body"):
            body = backbone.body

            if hasattr(body, "layer3"):
                layer3 = body.layer3
                # Use the last Conv2d inside layer3
                for m in reversed(list(layer3.modules())):
                    if isinstance(m, nn.Conv2d):
                        return m

            convs = [m for m in body.modules() if isinstance(m, nn.Conv2d)]
            if len(convs) >= 2:
                # Second-to-last Conv2d instead of the very last one
                return convs[-2]

            elif len(convs) == 1:
                return convs[0]

        # Swin Transformer: use patch embedding projection
        if hasattr(backbone, "patch_embed"):
            proj = getattr(backbone.patch_embed, "proj", None)
            if isinstance(proj, nn.Conv2d):
                return proj

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

        images = batch["images"]
        num_views = len(images) if isinstance(images, (list, tuple)) else 1

        # Reset buffers
        self._features.clear()
        self._grads.clear()

        # Forward pass
        # List[4 * (B, C, H, W)] -> (4B, C, H, W)
        output = self.model(batch["images"])
        logits = output[task]["logits"]  # (B, num_classes)

        # Select class for CAM score
        if use_predicted_class:
            target = logits.argmax(dim=1)  # (B,)
        else:
            target = batch["label"][task]  # (B,)

        # Backward target score
        score = logits.gather(1, target.unsqueeze(1)).sum()

        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        # Collect hooked features and gradients
        if not self._features or not self._grads:
            return []

        features = self._features[-1]  # (4B, C, H, W)
        grads = self._grads[-1]  # (4B, C, H, W)

        # Channel weights
        # (4B, C, H, W) -> (4B, C, 1, 1)
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of all channels
        # (4B, C, H, W) -> (4B, H, W)
        cam = (weights * features).sum(dim=1)

        # Activate
        cam = F.relu(cam)

        # Calculate heatmap
        cam = cam / (cam.amax(dim=(-2, -1), keepdim=True) + 1e-6)

        # Batch size
        total_B = cam.shape[0]
        B = total_B // num_views

        # (4B, H, W) -> (4, B, H, W)
        cam = cam.view(num_views, B, *cam.shape[1:])

        # (4, B, H, W) -> List[4 * (B, H, W)]
        cams_per_view: List[torch.Tensor] = [
            cam[v].detach()
            for v in range(num_views)
        ]

        return cams_per_view
