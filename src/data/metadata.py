# src/data/metadata.py

from __future__ import annotations
from typing import List, Dict

import pandas as pd

from config import METADATA_PATH

__all__ = ["Metadata"]
METADATA_COLUMNS = (
    "series_id",
    "image_id",
    "laterality",
    "view_position",
    "photometric_interpretation",
)


class Metadata:
    def __init__(
            self,
            metadata_path: str = METADATA_PATH,
            columns: List[str] = METADATA_COLUMNS,
    ):
        self.metadata_path = metadata_path
        self.columns = list(columns)
        self.metadata_df = self._load_metadata()

        self._build_map()
        self.by_image_id = ByImageID(dict(self._by_image_id))

    def _load_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self.metadata_path)

        # Rename
        df = df.rename(columns={
            "Series Instance UID": "series_id",
            "SOP Instance UID.1": "image_id",
            "Image Laterality": "laterality",
            "View Position": "view_position",
            "Photometric Interpretation": "photometric_interpretation",
        })

        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(f"[Metadata] Missing columns: {missing}")

        return df[self.columns].copy()

    def _build_map(self):
        self._by_image_id: Dict[str, Dict[str, str]] = {}

        for _, row in self.metadata_df.iterrows():
            image_id = row["image_id"]
            self._by_image_id[image_id] = {
                column: row[column] for column in self.columns if column != "image_id"
            }


class ByImageID:
    def __init__(
            self,
            image_id_map: Dict[str, Dict[str, str]],
    ):
        self._by_image_id = image_id_map

    def get_all(self) -> Dict[str, Dict[str, str]]:
        return {k: dict(v) for k, v in self._by_image_id.items()}

    def get(self, image_id: str) -> Dict[str, str]:
        if image_id not in self._by_image_id:
            raise KeyError(f"[Metadata] image_id={image_id} not found in metadata.")

        return dict(self._by_image_id[image_id])

    def get_laterality(self, image_id: str) -> str:
        return self.get(image_id)["laterality"]

    def get_view_position(self, image_id: str) -> str:
        return self.get(image_id)["view_position"]

    def get_photometric_interpretation(self, image_id: str) -> str:
        return self.get(image_id)["photometric_interpretation"]
