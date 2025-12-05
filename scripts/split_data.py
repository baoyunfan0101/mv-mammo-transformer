# scripts/split_data.py

from __future__ import annotations
from typing import Dict, List
import argparse
import random

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data.breast_level import BreastLevel
from src.data.status import Status
from src.utils.log_utils import log


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split studies into F1–F5 (training) and test."
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of training studies to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=47,
        help="Random seed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.n > 4000:
        raise ValueError(
            f"Requested {args.n} training studies, "
            f"but only 4000 are available."
        )

    data_version = f"first_{args.n}" if args.n < 4000 else "full"

    # Load data sources
    breast_level = BreastLevel()
    status = Status()

    by_study = breast_level.by_study_id

    # Collect study_ids by original split
    training_studies: List[str] = []
    test_studies: List[str] = []

    for study_id in by_study.get_all().keys():
        split = by_study.get_split(study_id)

        if split == "training":
            training_studies.append(study_id)
        elif split == "test":
            test_studies.append(study_id)

    # Limit training set to first N studies
    training_studies = training_studies[: args.n]

    # Build stratification labels (study-level)
    labels: List[str] = []

    for study_id in training_studies:
        birads_map = by_study.get_breast_birads(study_id)
        max_birads = max(
            int(v.split()[-1])
            for v in birads_map.values()
        )
        labels.append(max_birads)

    # Stratified split into F1–F5 (study-level)
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=args.seed,
    )

    study_to_split: Dict[str, str] = {}

    for fold_idx, (_, val_idx) in enumerate(
            skf.split(training_studies, labels)
    ):
        fold_name = f"F{fold_idx + 1}"
        for i in val_idx:
            study_to_split[training_studies[i]] = fold_name

    # -------------------------------------------------
    # Write splits into status.csv
    # Each study_id: all 4 images marked together
    # -------------------------------------------------
    with status.split() as sm:
        if data_version not in status.status_df.columns:
            status.status_df[data_version] = ""

        # Training folds
        for study_id, split in study_to_split.items():
            entries = by_study.get(study_id)

            for e in entries:
                sm.mark(
                    study_id=study_id,
                    laterality=e["laterality"],
                    view_position=e["view_position"],
                    data_version=data_version,
                    split=split,
                )

        # Test (kept as test)
        for study_id in test_studies:
            entries = by_study.get(study_id)

            for e in entries:
                sm.mark(
                    study_id=study_id,
                    laterality=e["laterality"],
                    view_position=e["view_position"],
                    data_version=data_version,
                    split="test",
                )

    # Summary
    log(
        f"data_version='{data_version}' "
        f"training={len(training_studies)} studies (F1–F5), "
        f"test={len(test_studies)} studies.",
        "SplitData"
    )


if __name__ == "__main__":
    main()
