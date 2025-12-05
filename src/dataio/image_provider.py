# src/dataio/image_provider.py

from __future__ import annotations

import numpy as np

from src.data.status import Status
from src.dataio.keys import ImageKey

__all__ = ["ImageProvider"]


class ImageProvider:

    def __init__(
            self,
            status: Status,
            *,
            array_key: str = "img",
    ):
        self.status = status
        self.array_key = array_key

    def get_image(
            self,
            key: ImageKey
    ) -> np.ndarray:
        entry = self.status.by_index.get(
            key.study_id,
            key.laterality,
            key.view,
        )

        if not entry["preprocessed"]:
            raise RuntimeError(f"Image not preprocessed: {key}")

        data = np.load(entry["preprocessed_path"])
        return data[self.array_key]
