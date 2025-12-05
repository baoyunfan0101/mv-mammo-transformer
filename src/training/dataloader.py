# src/training/dataloader.py

from __future__ import annotations
from typing import Dict, Any, Sequence, Callable, Literal, Optional

from torch.utils.data import DataLoader, default_collate

from src.data.status import Status
from src.data.breast_level import BreastLevel
from src.dataio.index_provider import IndexProvider
from src.dataio.image_provider import ImageProvider
from src.dataio.label_provider import LabelProvider
from src.dataio.image_dataset import ImageDataset

__all__ = [
    "build_dataset",
    "build_dataloader",
]


def build_dataset(
        *,
        status: Status,
        breast_level: Optional[BreastLevel] = None,
        data_version: str,
        splits: Sequence[str],
        mode: Literal["single", "multi"] = "single",
        with_labels: bool = True,
        label_columns: Sequence[str],
        label_mapping: Optional[Dict[str, Dict[Any, int]]] = None,
) -> ImageDataset:
    index_provider = IndexProvider(
        status=status,
        data_version=data_version,
        splits=list(splits),
    )

    image_provider = ImageProvider(
        status=status,
        array_key="img",
    )

    label_provider = None
    if with_labels:
        if breast_level is None:
            raise ValueError(
                "[src.training.build_dataset] with_labels=True but breast_level is None."
            )
        if label_columns is None:
            raise ValueError(
                "[src.training.build_dataset] with_labels=True but label_columns is None."
            )

        label_provider = LabelProvider(
            breast_level=breast_level,
            columns=list(label_columns),
            mapping=label_mapping or {},
        )

    dataset = ImageDataset(
        index_provider=index_provider,
        image_provider=image_provider,
        label_provider=label_provider,
        mode=mode,
    )
    return dataset


def build_dataloader(
        dataset: ImageDataset,
        *,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        collate_fn: Callable = default_collate,
) -> DataLoader:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    return loader
