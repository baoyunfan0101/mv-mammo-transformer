# src/data/breast_level.py

from __future__ import annotations
from typing import List, Dict

import pandas as pd

from config import METADATA_PATH, BREAST_LEVEL_PATH

__all__ = ["BreastLevel"]
BREAST_LEVEL_COLUMNS = (
    "study_id",
    "series_id",
    "image_id",
    "laterality",
    "view_position",
    "breast_birads",
    "breast_density",
    "split",
)


class BreastLevel:
    def __init__(
            self,
            breast_level_path: str = BREAST_LEVEL_PATH,
            columns: List[str] = BREAST_LEVEL_COLUMNS,
    ):
        self.breast_level_path = breast_level_path
        self.columns = list(columns)
        self.breast_level_df = self._load_breast_level()

        self._build_map()
        self.by_study_id = ByStudyID(dict(self._by_study_id))
        self.by_series_id = BySeriesID(dict(self._by_series_id))
        self.by_image_id = ByImageID(dict(self._by_image_id))
        self.by_split = BySplit(self.breast_level_df.copy())

    def _load_breast_level(self) -> pd.DataFrame:
        df = pd.read_csv(self.breast_level_path)

        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(f"[BreastLevel] Missing columns: {missing}")

        return df[self.columns].copy()

    def _build_map(self):
        self._by_study_id: Dict[str, List[Dict[str, str]]] = {}
        self._by_series_id: Dict[str, List[Dict[str, str]]] = {}
        self._by_image_id: Dict[str, Dict[str, str]] = {}

        for _, row in self.breast_level_df.iterrows():
            study_id = row["study_id"]
            if study_id not in self._by_study_id:
                self._by_study_id[study_id] = []
            self._by_study_id[study_id].append({
                column: row[column] for column in self.columns if column != "study_id"
            })

            series_id = row["series_id"]
            if series_id not in self._by_series_id:
                self._by_series_id[series_id] = []
            self._by_series_id[series_id].append({
                column: row[column] for column in self.columns if column != "series_id"
            })

            image_id = row["image_id"]
            self._by_image_id[image_id] = {
                column: row[column] for column in self.columns if column != "image_id"
            }


class ByStudyID:
    def __init__(
            self,
            study_id_map: Dict[str, List[Dict[str, str]]],
    ):
        self._by_study_id = study_id_map

    def get_all(self) -> Dict[str, List[Dict[str, str]]]:
        return {
            study_id: [dict(item) for item in items]
            for study_id, items in self._by_study_id.items()
        }

    def get(self, study_id: str) -> List[Dict[str, str]]:
        if study_id not in self._by_study_id:
            raise KeyError(f"[BreastLevel] study_id={study_id} not found in breast_level.")

        return [dict(item) for item in self._by_study_id[study_id]]

    def get_series_id(self, study_id: str) -> str:
        return self.get(study_id)[0]["series_id"]

    def get_image_id(self, study_id: str) -> Dict[Tuple[str, str], str]:
        return {
            (item["laterality"], item["view_position"]): item["image_id"]
            for item in self.get(study_id)
        }

    def get_breast_birads(self, study_id: str) -> Dict[str, str]:
        return {item["laterality"]: item["breast_birads"] for item in self.get(study_id)}

    def get_breast_density(self, study_id: str) -> Dict[str, str]:
        return {item["laterality"]: item["breast_density"] for item in self.get(study_id)}

    def get_split(self, study_id: str) -> str:
        return self.get(study_id)[0]["split"]


class BySeriesID:
    def __init__(
            self,
            series_id_map: Dict[str, List[Dict[str, str]]],
    ):
        self._by_series_id = series_id_map

    def get_all(self) -> Dict[str, List[Dict[str, str]]]:
        return {
            series_id: [dict(item) for item in items]
            for series_id, items in self._by_series_id.items()
        }

    def get(self, series_id: str) -> List[Dict[str, str]]:
        if series_id not in self._by_series_id:
            raise KeyError(f"[BreastLevel] series_id={series_id} not found in breast_level.")

        return [dict(item) for item in self._by_series_id[series_id]]

    def get_study_id(self, series_id: str) -> str:
        return self.get(series_id)[0]["study_id"]

    def get_image_id(self, series_id: str) -> Dict[Tuple[str, str], str]:
        return {
            (item["laterality"], item["view_position"]): item["image_id"]
            for item in self.get(series_id)
        }

    def get_breast_birads(self, series_id: str) -> Dict[str, str]:
        return {item["laterality"]: item["breast_birads"] for item in self.get(series_id)}

    def get_breast_density(self, series_id: str) -> Dict[str, str]:
        return {item["laterality"]: item["breast_density"] for item in self.get(series_id)}

    def get_split_by_series_id(self, series_id: str) -> str:
        return self.get(series_id)[0]["split"]


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
            raise KeyError(f"[BreastLevel] image_id={image_id} not found in breast_level.")

        return dict(self._by_image_id[image_id])

    def get_study_id(self, image_id: str) -> str:
        return self.get(image_id)["study_id"]

    def get_series_id(self, image_id: str) -> str:
        return self.get(image_id)["series_id"]

    def get_laterality(self, image_id: str) -> str:
        return self.get(image_id)["laterality"]

    def get_view_position(self, image_id: str) -> str:
        return self.get(image_id)["view_position"]

    def get_breast_birads(self, image_id: str) -> str:
        return self.get(image_id)["breast_birads"]

    def get_breast_density(self, image_id: str) -> str:
        return self.get(image_id)["breast_density"]

    def get_split(self, image_id: str) -> str:
        return self.get(image_id)["split"]


class BySplit:
    def __init__(
            self,
            breast_level_df: pd.DataFrame
    ):
        self.breast_level_df = breast_level_df

    def get_all_study_id(self) -> Dict[str, List[str]]:
        return {
            str(split): df["study_id"].unique().tolist()
            for split, df in self.breast_level_df.groupby("split")
        }

    def get_all_series_id_dict(self) -> Dict[str, List[str]]:
        return {
            str(split): df["series_id"].unique().tolist()
            for split, df in self.breast_level_df.groupby("split")
        }

    def get_all_image_id_dict(self) -> Dict[str, List[str]]:
        return {
            str(split): df["image_id"].unique().tolist()
            for split, df in self.breast_level_df.groupby("split")
        }
