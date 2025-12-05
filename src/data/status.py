# src/data/status.py

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Callable

import os
import pandas as pd

from config import STATUS_PATH

__all__ = ["Status"]
STATUS_COLUMNS = (
    "study_id",
    "laterality",
    "view_position",
    "downloaded",
    "downloaded_path",
    "preprocessed",
    "preprocessed_path",
    "first_10",
    "first_1000",
    "full",
)
DEFAULT_COLUMN_VALUES = {
    "downloaded": False,
    "downloaded_path": "",
    "preprocessed": False,
    "preprocessed_path": "",
    "first_10": "",
    "first_1000": "",
    "full": "",
}


class Status:
    def __init__(
            self,
            status_path: str = STATUS_PATH,
            columns: List[str] = STATUS_COLUMNS,
    ):
        self.status_path = status_path
        self.columns = list(columns)
        self.status_df = self._load_status()

        self.by_index = self._build_map()

    def _load_status(self) -> pd.DataFrame:
        if not os.path.exists(self.status_path):
            raise FileNotFoundError(
                f"[Status] {self.status_path} not found. "
                f"Please call Status.build() first."
            )

        df = pd.read_csv(self.status_path)

        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"[Status] Missing columns: {missing}. "
                f"Please call Status.repair() first."
            )

        return df[self.columns].copy()

    def _build_map(self):
        _by_index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for _, row in self.status_df.iterrows():
            key = row["study_id"], row["laterality"], row["view_position"]
            _by_index[key] = {
                column: row[column]
                for column in self.columns
                if column not in ["study_id", "laterality", "view_position"]
            }

        return ByIndex(_by_index)

    def save(self):
        self.status_df.to_csv(self.status_path, index=False)
        self.by_index = self._build_map()

    @staticmethod
    def _build_status(
            *,
            columns: List[str],
            breast_level,
    ) -> pd.DataFrame:
        records = []

        for v in breast_level.by_image_id.get_all().values():
            records.append(
                {
                    "study_id": v["study_id"],
                    "laterality": v["laterality"],
                    "view_position": v["view_position"],
                } | DEFAULT_COLUMN_VALUES
            )

        return pd.DataFrame.from_records(records, columns=columns)

    @classmethod
    def build(
            cls,
            *,
            status_path: str = STATUS_PATH,
            columns: List[str] = STATUS_COLUMNS,
            breast_level,
    ) -> "Status":
        columns = list(columns)

        if os.path.exists(status_path):
            raise FileExistsError(
                f"[Status] {status_path} already exists. "
                f"Please call Status.repair() instead."
            )

        df = cls._build_status(
            columns=columns,
            breast_level=breast_level,
        )

        df.to_csv(status_path, index=False)

        return cls(
            status_path=status_path,
            columns=columns,
        )

    @classmethod
    def repair(
            cls,
            *,
            status_path: str = STATUS_PATH,
            columns: List[str] = STATUS_COLUMNS,
            breast_level,
    ) -> "Status":
        columns = list(columns)

        if not os.path.exists(status_path):
            raise FileNotFoundError(
                f"[Status] {status_path} not found. "
                f"Please call Status.build() instead."
            )

        df = pd.read_csv(status_path)

        missing = [c for c in columns if c not in df.columns]
        if not missing:
            return cls(
                status_path=status_path,
                columns=columns,
            )

        if "study_id" in missing or "laterality" in missing or "view_position" in missing:
            df = cls._build_status(
                columns=columns,
                breast_level=breast_level,
            )
        else:
            for missing_column in missing:
                df[missing_column] = DEFAULT_COLUMN_VALUES[missing_column]

        df = df[columns]
        df.to_csv(status_path, index=False)

        return cls(
            status_path=status_path,
            columns=columns,
        )

    def reset(
            self,
            breast_level,
    ):
        self.status_df = self._build_status(
            columns=self.columns,
            breast_level=breast_level,
        )

        self.save()

    def get_study_ids(self) -> List[str]:
        return self.status_df["study_id"].unique().tolist()

    def get_index(self) -> List[Dict[str, str]]:
        return self.status_df[
            ["study_id", "laterality", "view_position"]
        ].to_dict(orient="records")

    def get_downloaded(self) -> List[Dict[str, str]]:
        return self.status_df[self.status_df["downloaded"] == True][
            ["study_id", "laterality", "view_position"]
        ].to_dict(orient="records")

    def get_undownloaded(self) -> List[Dict[str, str]]:
        return self.status_df[self.status_df["downloaded"] == False][
            ["study_id", "laterality", "view_position"]
        ].to_dict(orient="records")

    def get_preprocessed(self) -> List[Dict[str, str]]:
        return self.status_df[self.status_df["preprocessed"] == True][
            ["study_id", "laterality", "view_position"]
        ].to_dict(orient="records")

    def get_unpreprocessed(self) -> List[Dict[str, str]]:
        return self.status_df[self.status_df["preprocessed"] == False][
            ["study_id", "laterality", "view_position"]
        ].to_dict(orient="records")

    def get_split(
            self,
            data_version: str,
    ) -> Dict[str, List[Dict[str, str]]]:
        return {
            str(split): df[
                ["study_id", "laterality", "view_position"]
            ].to_dict(orient="records")
            for split, df in self.status_df.groupby(data_version)
        }

    def download(self) -> "DownloadManager":
        return DownloadManager(self.status_df, self.save)

    def preprocess(self) -> "PreprocessManager":
        return PreprocessManager(self.status_df, self.save)

    def split(self) -> "SplitManager":
        return SplitManager(self.status_df, self.save)

    @property
    def info(self) -> Dict[str, Any]:
        df = self.status_df

        info = {}

        def _completion_stats(col: str):
            # image-level
            image_total = len(df)
            image_done = int(df[col].sum())

            # study-level: all images in a study must be done
            study_group = df.groupby("study_id")[col].all()
            study_total = study_group.shape[0]
            study_done = int(study_group.sum())

            return {
                "study": {
                    "done": study_done,
                    "total": study_total,
                },
                "image": {
                    "done": image_done,
                    "total": image_total,
                },
            }

        info["download"] = _completion_stats("downloaded")
        info["preprocess"] = _completion_stats("preprocessed")

        splits_info = {}

        data_versions = ["first_10", "first_1000", "full"]

        for dv in data_versions:
            if dv not in df.columns:
                continue

            dv_info = {}

            # deduplicate at study level
            study_df = (
                df[["study_id", dv]]
                .drop_duplicates(subset=["study_id"])
            )

            for split, sub_df in study_df.groupby(dv):
                if split is None or split == "":
                    continue

                num_studies = len(sub_df)
                num_images = num_studies * 4

                dv_info[str(split)] = {
                    "studies": num_studies,
                    "images": num_images,
                }

            splits_info[dv] = dv_info

        info["split"] = splits_info

        return info


class ByIndex:
    def __init__(
            self,
            index_map: Dict[Tuple[str, str, str], Dict[str, Any]],
    ):
        self._by_index = index_map

    def get_all(self) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
        return {k: dict(v) for k, v in self._by_index.items()}

    def get(
            self,
            study_id: str,
            laterality: str,
            view_position: str,
    ) -> Dict[str, Any]:
        if (study_id, laterality, view_position) not in self._by_index:
            raise KeyError(
                f"[Status] index=({image_id}, {laterality}, {view_position}) "
                "not found in status."
            )

        return dict(self._by_index[(study_id, laterality, view_position)])


class DownloadManager:
    def __init__(
            self,
            status_df: pd.DataFrame,
            save: Callable,
    ):
        self.status_df = status_df
        self.save = save

    def __enter__(self) -> "DownloadManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.save()
        return False

    def mark(
            self,
            study_id: str,
            laterality: str,
            view_position: str,
            downloaded_path: str,
    ):
        mask = (
                (self.status_df["study_id"] == study_id)
                & (self.status_df["laterality"] == laterality)
                & (self.status_df["view_position"] == view_position)
        )

        self.status_df.loc[mask, "downloaded"] = True
        self.status_df.loc[mask, "downloaded_path"] = downloaded_path

    def unmark(
            self,
            study_id: str,
            laterality: str,
            view_position: str,
    ):
        mask = (
                (self.status_df["study_id"] == study_id)
                & (self.status_df["laterality"] == laterality)
                & (self.status_df["view_position"] == view_position)
        )

        self.status_df.loc[mask, "downloaded"] = DEFAULT_COLUMN_VALUES["downloaded"]
        self.status_df.loc[mask, "downloaded_path"] = DEFAULT_COLUMN_VALUES["downloaded_path"]

    def reset(self):
        self.status_df["downloaded"] = DEFAULT_COLUMN_VALUES["downloaded"]
        self.status_df["downloaded_path"] = DEFAULT_COLUMN_VALUES["downloaded_path"]

    def update(
            self,
            downloaded_dir: str,
            metadata,
    ):
        if not os.path.isdir(downloaded_dir):
            raise ValueError(f"[Status.DownloadManager] Not a directory: {downloaded_dir}")

        found = updated = 0

        for study_id in os.listdir(downloaded_dir):
            study_dir = os.path.join(downloaded_dir, study_id)

            if not os.path.isdir(study_dir):
                continue

            for fname in os.listdir(study_dir):
                if not (fname.endswith(".dicom") or fname.endswith(".dcm")) or fname.startswith("._"):
                    continue

                found += 1

                image_id = os.path.splitext(fname)[0]

                try:
                    info = metadata.by_image_id.get(image_id)
                except KeyError:
                    raise ValueError(
                        f"[Status.DownloadManager] Invalid image_id in {study_dir}: {image_id}"
                    )

                laterality = info["laterality"]
                view_position = info["view_position"]

                mask = (
                        (self.status_df["study_id"] == study_id)
                        & (self.status_df["laterality"] == laterality)
                        & (self.status_df["view_position"] == view_position)
                )

                n = mask.sum()
                if n == 0:
                    continue

                if n > 1:
                    raise ValueError(
                        f"[Status.DownloadManager] Multiple rows matched for "
                        f"{study_id}_{laterality}_{view_position}"
                    )

                full_path = os.path.join(study_dir, fname)

                self.status_df.loc[mask, "downloaded"] = True
                self.status_df.loc[mask, "downloaded_path"] = full_path

                # A record updated
                updated += 1

        return {
            "found": found,
            "updated": updated,
        }


class PreprocessManager:
    def __init__(
            self,
            status_df: pd.DataFrame,
            save: Callable,
    ):
        self.status_df = status_df
        self.save = save

    def __enter__(self) -> "PreprocessManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.save()
        return False

    def mark(
            self,
            study_id: str,
            laterality: str,
            view_position: str,
            preprocessed_path: str,
    ):
        mask = (
                (self.status_df["study_id"] == study_id)
                & (self.status_df["laterality"] == laterality)
                & (self.status_df["view_position"] == view_position)
        )

        self.status_df.loc[mask, "preprocessed"] = True
        self.status_df.loc[mask, "preprocessed_path"] = preprocessed_path

    def unmark(
            self,
            study_id: str,
            laterality: str,
            view_position: str,
    ):
        mask = (
                (self.status_df["study_id"] == study_id)
                & (self.status_df["laterality"] == laterality)
                & (self.status_df["view_position"] == view_position)
        )

        self.status_df.loc[mask, "preprocessed"] = DEFAULT_COLUMN_VALUES["preprocessed"]
        self.status_df.loc[mask, "preprocessed_path"] = DEFAULT_COLUMN_VALUES["preprocessed_path"]

    def reset(self):
        self.status_df["preprocessed"] = DEFAULT_COLUMN_VALUES["preprocessed"]
        self.status_df["preprocessed_path"] = DEFAULT_COLUMN_VALUES["preprocessed_path"]

    def update(
            self,
            preprocessed_dir: str,
            separator: str = "_",
    ):
        if not os.path.isdir(preprocessed_dir):
            raise ValueError(f"[Status.PreprocessManager] Not a directory: {preprocessed_dir}")

        found = updated = 0

        for fname in os.listdir(preprocessed_dir):
            if not (fname.endswith(".npz") or fname.endswith(".npy")) or fname.startswith("._"):
                continue

            stem = os.path.splitext(fname)[0]
            parts = stem.split(separator)

            if len(parts) != 3:
                continue

            # A file found
            found += 1

            study_id, laterality, view_position = parts

            mask = (
                    (self.status_df["study_id"] == study_id)
                    & (self.status_df["laterality"] == laterality)
                    & (self.status_df["view_position"] == view_position)
            )

            n = mask.sum()
            if n == 0:
                continue

            if n > 1:
                raise ValueError(
                    f"[Status.PreprocessManager] Multiple rows matched for "
                    f"{study_id}_{laterality}_{view_position}"
                )

            # A record updated
            updated += 1

            full_path = os.path.join(preprocessed_dir, fname)

            self.status_df.loc[mask, "preprocessed"] = True
            self.status_df.loc[mask, "preprocessed_path"] = full_path

        return {
            "found": found,
            "updated": updated,
        }


class SplitManager:
    def __init__(
            self,
            status_df: pd.DataFrame,
            save: Callable,
    ):
        self.status_df = status_df
        self.save = save

    def __enter__(self) -> "SplitManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.save()
        return False

    def mark(
            self,
            study_id: str,
            laterality: str,
            view_position: str,
            data_version: str,
            split: str,
    ):
        if data_version not in self.status_df.columns:
            raise ValueError(
                f"[Status.PreprocessManager] Unknown data version: {data_version}"
            )

        mask = (
                (self.status_df["study_id"] == study_id)
                & (self.status_df["laterality"] == laterality)
                & (self.status_df["view_position"] == view_position)
        )

        self.status_df.loc[mask, data_version] = split

    def reset(
            self,
            data_version: str,
    ):
        self.status_df[data_version] = DEFAULT_COLUMN_VALUES[data_version]
