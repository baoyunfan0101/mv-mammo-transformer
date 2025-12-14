# src/losses/gradcam_loss.py

from __future__ import annotations
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.bbox import BBox
from src.dataio.keys import ImageKey, MultiViewKey

__all__ = ["GradCAMLoss"]


# Gradient-weighted Class Activation Mapping
class GradCAMLoss(nn.Module):

    def __init__(
            self,
            model: nn.Module,
            *,
            weight: float = 1.0,
            target_layer: Optional[nn.Module] = None,
            bbox: Optional[BBox] = None,
    ):
        super().__init__()
        self.model = model
        self.weight = weight
        self.bbox = bbox

        self.target_layer = target_layer or self._infer_target_layer()

        # Buffers for features and gradients
        self._features: List[torch.Tensor] = []
        self._grads: List[torch.Tensor] = []

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            # output: (4B, C, H, W)
            self._features = [output]

        def backward_hook(_, grad_input, grad_output):
            # grad_output[0]: (4B, C, H, W)
            self._grads = [grad_output[0]]

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
            "[GradCAMLoss] Cannot infer target_layer automatically for backbone "
            f"{type(backbone).__name__}. Please specify target_layer manually."
        )

    def _get_bboxes(
            self,
            key: ImageKey,
    ) -> List[tuple]:
        if self.bbox is None:
            return []

        # List[(xmin, ymin, xmax, ymax)]
        return self.bbox.by_index.get(
            key.study_id,
            key.laterality,
            key.view_position,
        )

    def forward(
            self,
            output: Dict[str, Any],
            batch: Dict[str, Any],
    ) -> torch.Tensor:

        logits = output["breast_birads"]["logits"]  # (B, num_classes)
        labels = batch["label"]["breast_birads"]  # (B,)
        keys = batch["key"]  # List[ImageKey | MultiViewKey], len = B
        images = batch["images"]
        num_views = len(images) if isinstance(images, (list, tuple)) else 1

        if self.bbox is None:
            return logits.new_zeros(())

        else:
            has_any_bbox = False
            for key in keys:
                if isinstance(key, MultiViewKey):
                    for image_key in key.views()[:num_views]:
                        if self._get_bboxes(image_key):
                            has_any_bbox = True
                            break
                else:
                    if self._get_bboxes(key):
                        has_any_bbox = True

                if has_any_bbox:
                    break

            # Early exit if no bbox anywhere
            if not has_any_bbox:
                return logits.new_zeros(())

        # Reset buffers
        self._features.clear()
        self._grads.clear()

        # Backward target score
        score = logits.gather(1, labels.unsqueeze(1)).sum()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        if not self._features or not self._grads:
            return logits.new_zeros(())

        features = self._features[0]  # (4B, C, H, W)
        grads = self._grads[0]  # (4B, C, H, W)

        N, C, H, W = features.shape
        IMG_H, IMG_W = self.bbox.get_image_size()
        B = N // num_views

        # Compute GradCAM
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (N, C, 1, 1)
        cam = (weights * features).sum(dim=1)  # (N, H, W)
        cam = F.relu(cam)
        cam = cam / (cam.amax(dim=(-2, -1), keepdim=True) + 1e-6)

        # (N, H, W) -> (4, B, H, W)
        cam = cam.view(num_views, B, H, W)

        view_losses: List[torch.Tensor] = []

        # Compute loss for each view
        for view_idx in range(num_views):
            cam_v = cam[view_idx]  # (B, H, W)

            # (B, H, W) -> (B, 224, 224)
            cam_v = F.interpolate(
                cam_v.unsqueeze(1),  # (B, 1, H, W)
                size=(IMG_H, IMG_W),  # (224, 224)
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # (B, 224, 224)

            mask = torch.zeros_like(cam_v)

            for i, key in enumerate(keys):
                image_key = (
                    key.views()[view_idx]
                    if isinstance(key, MultiViewKey)
                    else key
                )

                for (x1, y1, x2, y2) in self._get_bboxes(image_key):
                    x1i = int(max(0, min(x1, IMG_W)))
                    x2i = int(max(0, min(x2, IMG_W)))
                    y1i = int(max(0, min(y1, IMG_H)))
                    y2i = int(max(0, min(y2, IMG_H)))
                    if x2i > x1i and y2i > y1i:
                        mask[i, y1i:y2i, x1i:x2i] = 1.0

            mask_sum = mask.sum(dim=(1, 2))
            valid = mask_sum > 0
            if not valid.any():
                continue

            inside = (cam_v * mask).sum(dim=(1, 2)) / (mask_sum + 1e-6)
            outside = (cam_v * (1.0 - mask)).sum(dim=(1, 2)) / (
                    (IMG_H * IMG_W - mask_sum) + 1e-6
            )

            view_losses.append((outside[valid] - inside[valid]).mean())

        if not view_losses:
            return logits.new_zeros(())

        return self.weight * torch.stack(view_losses).mean()