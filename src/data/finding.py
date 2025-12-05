# src/data/finding.py

from __future__ import annotations
from typing import List, Tuple, Dict, Any

import pandas as pd

from config import FINDING_PATH

__all__ = ["Finding"]
FINDING_COLUMNS = (
    "study_id",
    "series_id",
    "image_id",
    "laterality",
    "view_position",
    "breast_birads",
    "breast_density",
    "finding_categories",
    "finding_birads",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "split",
)


class Finding:
    def __init__(
            self,
            finding_path: str = FINDING_PATH,
            columns: List[str] = FINDING_COLUMNS,
    ):
        self.finding_path = finding_path
        self.columns = list(columns)
        self.finding_df = self._load_finding()

        self._build_map()
        self.by_index = ByIndex(dict(self._by_index))
        self.by_image_id = ByImageID(dict(self._by_image_id))

    def _load_finding(self) -> pd.DataFrame:
        df = pd.read_csv(self.finding_path)

        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(f"[Finding] Missing columns: {missing}")

        return df[self.columns].copy()

    def _build_map(self):
        self._by_index: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        self._by_image_id: Dict[str, List[Dict[str, Any]]] = {}

        for _, row in self.finding_df.iterrows():
            # Skip rows with no findings
            if pd.isna(row["xmin"]) or pd.isna(row["ymin"]) or pd.isna(row["xmax"]) or pd.isna(row["ymax"]):
                continue

            key = row["study_id"], row["laterality"], row["view_position"]
            if key not in self._by_index:
                self._by_index[key] = []
            self._by_index[key].append({
                column: row[column]
                for column in self.columns
                if column not in ["study_id", "laterality", "view_position"]
            })

            image_id = row["image_id"]
            if image_id not in self._by_image_id:
                self._by_image_id[image_id] = []
            self._by_image_id[image_id].append({
                column: row[column] for column in self.columns if column != "image_id"
            })


class ByIndex:
    def __init__(
            self,
            index_map: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
    ):
        self._by_index = index_map

    def get_all(self) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
        return {
            k: [dict(item) for item in v]
            for k, v in self._by_index.items()
        }

    def get(
            self,
            study_id: str,
            laterality: str,
            view_position: str,
    ) -> List[Dict[str, Any]]:
        if (study_id, laterality, view_position) not in self._by_index:
            return []

        return [
            dict(item)
            for item in self._by_index[(study_id, laterality, view_position)]
        ]

    def get_bbox(
            self,
            study_id: str,
            laterality: str,
            view_position: str,
    ) -> List[Tuple[int, int, int, int]]:
        return [
            (item["xmin"], item["ymin"], item["xmax"], item["ymax"])
            for item in self.get(study_id, laterality, view_position)
        ]


class ByImageID:
    def __init__(
            self,
            image_id_map: Dict[str, List[Dict[str, Any]]],
    ):
        self._by_image_id = image_id_map

    def get_all(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            k: [dict(item) for item in v]
            for k, v in self._by_image_id.items()
        }

    def get(self, image_id: str) -> List[Dict[str, Any]]:
        if image_id not in self._by_image_id:
            return []

        return [
            dict(item)
            for item in self._by_image_id[image_id]
        ]

    def get_bbox(self, image_id: str) -> List[Tuple[int, int, int, int]]:
        return [
            (item["xmin"], item["ymin"], item["xmax"], item["ymax"])
            for item in self.get(image_id)
        ]
