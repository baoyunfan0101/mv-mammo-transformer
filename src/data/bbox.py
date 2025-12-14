# src/data/bbox.py

from __future__ import annotations
from typing import List, Tuple, Dict, Callable

import os
import pandas as pd

from config import BBOX_PATH

__all__ = ["BBox"]
BBOX_COLUMNS = (
    "study_id",
    "laterality",
    "view_position",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
)
DEFAULT_COLUMN_VALUES = {
    "xmin": -1,
    "ymin": -1,
    "xmax": -1,
    "ymax": -1,
}
IMAGE_SIZE = (224, 224)


class BBox:
    def __init__(
            self,
            bbox_path: str = BBOX_PATH,
            columns: List[str] = BBOX_COLUMNS,
            image_size: Tuple[int, int] = (224, 224),
    ):
        self.bbox_path = bbox_path
        self.columns = list(columns)
        self.image_size = image_size
        self.bbox_df = self._load_bbox()

        self.by_index = self._build_map()

    def _load_bbox(self) -> pd.DataFrame:
        if not os.path.exists(self.bbox_path):
            raise FileNotFoundError(
                f"[BBox] {self.bbox_path} not found. "
                f"Please call BBox.build() first."
            )

        df = pd.read_csv(self.bbox_path)

        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"[BBox] Missing columns: {missing}. "
                f"Please call BBox.repair() first."
            )

        return df[self.columns].copy()

    def _build_map(self):
        _by_index: Dict[Tuple[str, str, str], List[Tuple[int, int, int, int]]] = {}

        for _, row in self.bbox_df.iterrows():
            key = row["study_id"], row["laterality"], row["view_position"]
            if key not in _by_index:
                _by_index[key] = []
            _by_index[key].append((
                row["xmin"],
                row["ymin"],
                row["xmax"],
                row["ymax"]
            ))

        return ByIndex(_by_index)

    def save(self):
        self.bbox_df.to_csv(self.bbox_path, index=False)
        self.by_index = self._build_map()

    @staticmethod
    def _build_bbox(
            *,
            columns: List[str],
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=columns)

    @classmethod
    def build(
            cls,
            *,
            bbox_path: str = BBOX_PATH,
            columns: List[str] = BBOX_COLUMNS,
    ) -> "BBox":
        columns = list(columns)

        if os.path.exists(bbox_path):
            raise FileExistsError(
                f"[BBox] {bbox_path} already exists. "
                f"Please call BBox.repair() instead."
            )

        df = cls._build_bbox(
            columns=columns,
        )

        df.to_csv(bbox_path, index=False)

        return cls(
            bbox_path=bbox_path,
            columns=columns,
        )

    @classmethod
    def repair(
            cls,
            *,
            bbox_path: str = BBOX_PATH,
            columns: List[str] = BBOX_COLUMNS,
    ) -> "BBox":
        columns = list(columns)

        if not os.path.exists(bbox_path):
            raise FileNotFoundError(
                f"[BBox] {bbox_path} not found. "
                f"Please call BBox.build() instead."
            )

        df = pd.read_csv(bbox_path)

        missing = [c for c in columns if c not in df.columns]
        if not missing:
            return cls(
                bbox_path=bbox_path,
                columns=columns,
            )

        if "study_id" in missing or "laterality" in missing or "view_position" in missing:
            df = cls._build_bbox(
                columns=columns,
            )
        else:
            for missing_column in missing:
                df[missing_column] = DEFAULT_COLUMN_VALUES[missing_column]

        df = df[columns]
        df.to_csv(bbox_path, index=False)

        return cls(
            bbox_path=bbox_path,
            columns=columns,
        )

    def reset(
            self,
    ):
        self.bbox_df = self._build_bbox(
            columns=self.columns,
        )

        self.save()

    def get_image_size(self) -> Tuple[int, int]:
        return self.image_size

    def get_study_ids(self) -> List[str]:
        return self.bbox_df["study_id"].unique().tolist()

    def get_index(self) -> List[Dict[str, str]]:
        return self.bbox_df[
            ["study_id", "laterality", "view_position"]
        ].to_dict(orient="records")

    def bbox(self) -> "BBoxManager":
        return BBoxManager(self.bbox_df, self.save)


class ByIndex:
    def __init__(
            self,
            index_map: Dict[Tuple[str, str, str], List[Tuple[int, int, int, int]]],
    ):
        self._by_index = index_map

    def get_all(self) -> Dict[Tuple[str, str, str], List[Tuple[int, int, int, int]]]:
        return {
            k: [bbox for bbox in v]
            for k, v in self._by_index.items()
        }

    def get(
            self,
            study_id: str,
            laterality: str,
            view_position: str,
    ) -> List[Tuple[int, int, int, int]]:
        if (study_id, laterality, view_position) not in self._by_index:
            return []

        return self._by_index[(study_id, laterality, view_position)]


class BBoxManager:
    def __init__(
            self,
            bbox_df: pd.DataFrame,
            save: Callable,
    ):
        self.bbox_df = bbox_df
        self.save = save

    def __enter__(self) -> "BBoxManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.save()
        return False

    def add(
            self,
            study_id: str,
            laterality: str,
            view_position: str,
            bbox: Tuple[int, int, int, int],
    ):
        xmin, ymin, xmax, ymax = bbox
        row = {
            "study_id": study_id,
            "laterality": laterality,
            "view_position": view_position,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
        self.bbox_df.loc[len(self.bbox_df)] = row

    def reset(self):
        self.bbox_df["xmin"] = DEFAULT_COLUMN_VALUES["xmin"]
        self.bbox_df["ymin"] = DEFAULT_COLUMN_VALUES["ymin"]
        self.bbox_df["xmax"] = DEFAULT_COLUMN_VALUES["xmax"]
        self.bbox_df["ymax"] = DEFAULT_COLUMN_VALUES["ymax"]
