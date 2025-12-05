# src/dataio/index_provider.py

from __future__ import annotations
from typing import List, Set, Optional

from src.data.status import Status
from src.dataio.keys import ImageKey, MultiViewKey

__all__ = ["IndexProvider", "EvalIndexProvider"]


class IndexProvider:

    def __init__(
            self,
            status: Status,
            *,
            data_version: str,
            splits: List[str],
    ):
        self.status = status
        self.data_version = data_version
        self.splits = splits

        self.single_index = self._build_single_index()
        self.multi_index = self._build_multi_index()

    def _build_single_index(self) -> List[ImageKey]:
        split_map = self.status.get_split(self.data_version)

        index: List[ImageKey] = []

        for split in self.splits:
            rows = split_map[split]
            for r in rows:
                index.append(
                    ImageKey(
                        study_id=r["study_id"],
                        laterality=r["laterality"],
                        view=r["view_position"],
                    )
                )

        return sorted(index, key=lambda x: x.study_id)

    def _build_multi_index(self) -> List[MultiViewKey]:
        split_map = self.status.get_split(self.data_version)

        study_id_set: Set[str] = set()
        index: List[MultiViewKey] = []

        for split in self.splits:
            rows = split_map[split]
            for r in rows:
                if r["study_id"] not in study_id_set:
                    study_id_set.add(r["study_id"])
                    index.append(
                        MultiViewKey(
                            study_id=r["study_id"],
                        )
                    )

        return sorted(index, key=lambda x: x.study_id)

    def get_single_index(
            self,
            idx: int,
    ) -> ImageKey:
        return self.single_index[idx]

    def get_multi_index(
            self,
            idx: int,
    ) -> MultiViewKey:
        return self.multi_index[idx]


class EvalIndexProvider(IndexProvider):
    def __init__(
            self,
            status: Status,
            *,
            study_id_list: List[str],
            data_version: Optional[str] = None,
            splits: Optional[List[str]] = None,
    ):
        self.study_id_list = study_id_list

        super().__init__(
            status=status,
            data_version=data_version,
            splits=[],
        )

    def _build_single_index(self) -> List[ImageKey]:
        index: List[ImageKey] = []

        for study_id in self.study_id_list:
            for laterality in ["L", "R"]:
                for view_position in ["CC", "MLO"]:
                    index.append(
                        ImageKey(
                            study_id=study_id,
                            laterality=laterality,
                            view=view_position,
                        )
                    )

        return sorted(index, key=lambda x: x.study_id)

    def _build_multi_index(self) -> List[MultiViewKey]:
        index: List[MultiViewKey] = []

        for study_id in self.study_id_list:
            index.append(
                MultiViewKey(
                    study_id=study_id,
                )
            )

        return sorted(index, key=lambda x: x.study_id)
