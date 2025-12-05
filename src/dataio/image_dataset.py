# src/dataio/image_dataset.py

from __future__ import annotations
from typing import Dict, Any, Literal, Optional

from collections import Counter

from torch.utils.data import Dataset

from src.dataio.keys import ImageKey, MultiViewKey
from src.dataio.index_provider import IndexProvider, EvalIndexProvider
from src.dataio.image_provider import ImageProvider
from src.dataio.label_provider import LabelProvider

__all__ = ["ImageDataset"]


class ImageDataset(Dataset):

    def __init__(
            self,
            *,
            index_provider: IndexProvider,
            image_provider: ImageProvider,
            label_provider: Optional[LabelRowProvider] = None,
            mode: Literal["single", "multi"] = "multi",
    ):
        self.index_provider = index_provider
        self.image_provider = image_provider
        self.label_provider = label_provider
        self.mode = mode

        if self.label_provider is not None:
            self._class_weight: Dict[str, Dict[int, float]] = self._compute_class_weight()

    def __len__(self) -> int:
        if self.mode == "single":
            return len(self.index_provider.single_index)
        else:
            return len(self.index_provider.multi_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.mode == "single":
            key = self.index_provider.get_single_index(idx)
            images = [self.image_provider.get_image(key)]
            if self.label_provider is not None:
                label = self.label_provider.get_single_label(key)

        else:
            key = self.index_provider.get_multi_index(idx)
            images = [self.image_provider.get_image(k) for k in key.views()]
            if self.label_provider is not None:
                label = self.label_provider.get_multi_label(key)

        item = {
            "key": key,  # ImageKey / MultiViewKey
            "images": images,  # List[np.ndarray]
        }

        if self.label_provider is not None:
            item["label"] = label

        return item

    def _compute_class_weight(self) -> Dict[str, Dict[int, float]]:
        counters: Dict[str, Counter] = {}

        for i in range(len(self)):
            if self.mode == "single":
                key = self.index_provider.get_single_index(i)
                label = self.label_provider.get_single_label(key)

            else:
                key = self.index_provider.get_multi_index(i)
                label = self.label_provider.get_multi_label(key)

            for name, val in label.items():
                counters.setdefault(name, Counter())[val] += 1

        self._class_weight = {
            name: {
                cls: sum(counter.values()) / count
                for cls, count in counter.items()
            }
            for name, counter in counters.items()
        }

        return self._class_weight

    def get_class_weight(self) -> Dict[str, Dict[int, float]]:
        return {k: dict(v) for k, v in self._class_weight.items()}
