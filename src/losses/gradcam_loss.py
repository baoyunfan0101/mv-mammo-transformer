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

        # Buffers for features and gradients from hooks
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
                "[GradCAMLoss] Swin backbone found, but patch_embed.proj is not Conv2d."
            )

        # Unsupported backbone
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
        logits = output["breast_birads"]["logits"]  # (B, C_b)
        target = batch["label"]["breast_birads"]  # (B,)
        keys = batch["key"]  # List[ImageKey | MultiViewKey], len = B

        # Determine number of views from batch["images"]
        images = batch.get("images", None)
        if isinstance(images, torch.Tensor):
            num_views = 1
        elif isinstance(images, (list, tuple)):
            num_views = len(images)
        else:
            raise RuntimeError(
                f"[GradCAMLoss] Unsupported images type: {type(images)}"
            )

        # Early exit: check whether the entire input has any bbox at all
        if self.bbox is not None:
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

            # If no bbox exists for any view of any sample, skip this loss
            if not has_any_bbox:
                return logits.new_zeros(())

        # Keep only features from the current forward (latest num_views calls)
        if len(self._features) >= num_views:
            self._features = self._features[-num_views:]
        self._grads = []

        # Select ground-truth class scores
        score = logits.gather(1, target.unsqueeze(1)).sum()

        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        if len(self._features) < num_views or len(self._grads) < num_views:
            return logits.new_zeros(())

        feats_list = self._features[-num_views:]
        grads_list = self._grads[-num_views:]

        view_losses: List[torch.Tensor] = []

        for view_idx, (features, grads) in enumerate(zip(feats_list, grads_list)):
            # features / grads shape: (B, C, H, W)
            B, C, H, W = features.shape

            # Grad-CAM computation
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
            cam = (weights * features).sum(dim=1)  # (B, H, W)
            cam = F.relu(cam)
            cam = cam / (cam.amax(dim=(-2, -1), keepdim=True) + 1e-6)

            if self.bbox is None:
                view_losses.append(cam.mean())
                continue

            # Build union mask for this view
            mask = torch.zeros_like(cam)

            for i, key in enumerate(keys):
                if isinstance(key, MultiViewKey):
                    image_key = key.views()[view_idx]
                else:
                    image_key = key

                bboxes = self._get_bboxes(image_key)
                if not bboxes:
                    continue

                for (x1, y1, x2, y2) in bboxes:
                    x1i = int(max(0, min(x1, W)))
                    x2i = int(max(0, min(x2, W)))
                    y1i = int(max(0, min(y1, H)))
                    y2i = int(max(0, min(y2, H)))
                    if x2i <= x1i or y2i <= y1i:
                        continue
                    mask[i, y1i:y2i, x1i:x2i] = 1.0

            mask_sum = mask.sum(dim=(1, 2))  # (B,)
            valid = mask_sum > 0
            if not valid.any():
                continue

            # Inside vs outside activation
            inside = (cam * mask).sum(dim=(1, 2)) / (mask_sum + 1e-6)
            outside = (cam * (1.0 - mask)).sum(dim=(1, 2)) / (
                    (H * W - mask_sum) + 1e-6
            )

            loss_view = outside[valid] - inside[valid]
            view_losses.append(loss_view.mean())

        if not view_losses:
            return logits.new_zeros(())

        return self.weight * torch.stack(view_losses).mean()
