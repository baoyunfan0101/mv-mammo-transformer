# scripts/preprocess.py

from __future__ import annotations
from typing import Tuple

import os
import numpy as np
from tqdm import tqdm

from src.data.breast_level import BreastLevel
from src.data.metadata import Metadata
from src.data.status import Status
from src.preprocessing.pipeline import preprocess_one_image
from src.utils.log_utils import log, log_section

from config import PROCESSED_DIR


def main(
        processed_dir: str = PROCESSED_DIR,
        target_size: Tuple[int, int] = (2560, 1440),
        clahe_clip: float = 2.0,
        clahe_grid: Tuple[int, int] = (16, 16),
):
    status = Status()
    metadata = Metadata()
    breast_level = BreastLevel()

    # Get downloaded but unpreprocessed index (study_id, laterality, view_position)
    downloaded = status.get_downloaded()
    unpreprocessed = status.get_unpreprocessed()
    key = lambda x: (x["study_id"], x["laterality"], x["view_position"])
    todo = [x for x in unpreprocessed if key(x) in {key(y) for y in downloaded}]
    log(f"{len(todo)} downloaded but unpreprocessed images", "Preprocess")

    if not todo:
        return

    log_section("Preprocess")
    with status.preprocess() as pm:
        for item in tqdm(todo):
            study_id = item["study_id"]
            laterality = item["laterality"]
            view_position = item["view_position"]

            # Get image_id
            images = breast_level.by_study_id.get_image_id(study_id)
            image_id = images[(laterality, view_position)]

            # In path
            dicom_path = status.by_index.get(study_id, laterality, view_position)["downloaded_path"]

            # Out dir
            out_dir = os.path.join(processed_dir)
            os.makedirs(out_dir, exist_ok=True)

            # Out path
            out_path = os.path.join(out_dir, f"{study_id}_{laterality}_{view_position}.npz")

            if not os.path.exists(out_path):
                try:
                    img = preprocess_one_image(
                        dicom_path=dicom_path,
                        metadata=metadata.by_image_id.get(image_id),
                        target_size=target_size,
                        clahe_clip=clahe_clip,
                        clahe_grid=clahe_grid,
                    )

                    # Compress to float16
                    img = img.astype(np.float16)
                    np.savez_compressed(out_path, img=img)

                except Exception as e:
                    log(f"Failed to preprocess {image_id}: {e}", "Preprocess")

            pm.mark(
                study_id=study_id,
                laterality=laterality,
                view_position=view_position,
                preprocessed_path=out_path,
            )

    log("Preprocessing finished", "Preprocess")


if __name__ == "__main__":
    main()
