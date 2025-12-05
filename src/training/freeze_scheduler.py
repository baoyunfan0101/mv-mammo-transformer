# src/training/freeze_scheduler.py

from __future__ import annotations

from src.utils.log_utils import log

__all__ = ["BackboneFreezeScheduler"]


class BackboneFreezeScheduler:

    def __init__(self, backbone, schedule_cfg):
        self.backbone = backbone
        self.schedule = schedule_cfg or []

    def apply(self, epoch: int):

        for rule in self.schedule:
            if rule["epoch"] != epoch:
                continue

            action = rule["action"]

            if action == "freeze_all":
                self.backbone.freeze_all()

            elif action == "unfreeze_all":
                self.backbone.unfreeze_all()

            elif action == "freeze_until":
                self.backbone.freeze_until(rule["n"])

            elif action == "unfreeze_from":
                self.backbone.unfreeze_from(rule["n"])

            elif action == "freeze_blocks":
                self.backbone.freeze_blocks(rule["idxs"])

            log(f"Applied: {rule}", "BackboneFreezeScheduler")
