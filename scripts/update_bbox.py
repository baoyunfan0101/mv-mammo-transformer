# scripts/update_bbox.py

from __future__ import annotations
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from src.data.status import Status
from src.data.bbox import BBox
from src.data.breast_level import BreastLevel
from src.data.metadata import Metadata
from src.data.finding import Finding
from src.preprocessing.dicom import load_dicom, remove_corner_text
from src.preprocessing.orientation import canonicalize_laterality
from src.preprocessing.crop import fixed_roi_crop

from config import BBOX_PATH


def update_bbox(
        *,
        target_size: Tuple[int, int] = (2560, 1440),
        image_size: Tuple[int, int] = (224, 224),
):
    status = Status()
    breast_level = BreastLevel()
    metadata = Metadata()
    finding = Finding()

    bbox = BBox.build(bbox_path=BBOX_PATH)

    with bbox.bbox() as bm:
        all_dict = finding.by_index.get_all()

        for k, v in tqdm(all_dict.items()):
            for item in v:
                if pd.isna(item["xmin"]) or pd.isna(item["ymin"]) or pd.isna(item["xmax"]) or pd.isna(item["ymax"]):
                    continue

                # Get index
                study_id, laterality, view_position = k

                # Get image_id
                images = breast_level.by_study_id.get_image_id(study_id)
                image_id = images[(laterality, view_position)]

                orig_bbox = (item["xmin"], item["ymin"], item["xmax"], item["ymax"])

                # Load original image
                dicom_path = status.by_index.get(
                    study_id=study_id,
                    laterality=laterality,
                    view_position=view_position,
                )["downloaded_path"]

                photometric_interpretation = metadata.by_image_id.get(image_id).get(
                    "photometric_interpretation",
                    "MONOCHROME2"
                ),

                img = load_dicom(
                    dicom_path,
                    photometric_interpretation=photometric_interpretation,
                )
                img, _ = remove_corner_text(img)

                H, W = img.shape

                # Apply horizontal flip if laterality == R
                if laterality.upper() == "R":
                    img = canonicalize_laterality(img, laterality)

                    # Flip bbox in original image coordinate
                    x1, y1, x2, y2 = orig_bbox
                    orig_bbox = (
                        W - x2,
                        y1,
                        W - x1,
                        y2,
                    )

                # Apply fixed ROI crop (track crop bbox)
                _, crop_bbox = fixed_roi_crop(
                    img,
                    target_size=target_size,
                )
                cx1, cy1, cx2, cy2 = crop_bbox

                # Map original bbox into cropped image coordinate
                bx1 = orig_bbox[0] - cx1
                by1 = orig_bbox[1] - cy1
                bx2 = orig_bbox[2] - cx1
                by2 = orig_bbox[3] - cy1

                # Clamp to crop area
                bx1 = max(0, min(bx1, cx2 - cx1))
                bx2 = max(0, min(bx2, cx2 - cx1))
                by1 = max(0, min(by1, cy2 - cy1))
                by2 = max(0, min(by2, cy2 - cy1))

                if bx2 <= bx1 or by2 <= by1:
                    continue

                # Resize bbox from crop_size to model image_size
                crop_h = cy2 - cy1
                crop_w = cx2 - cx1
                out_h, out_w = image_size

                sx = out_w / crop_w
                sy = out_h / crop_h

                x1, y1, x2, y2 = (
                    bx1 * sx,
                    by1 * sy,
                    bx2 * sx,
                    by2 * sy,
                )

                # Convert to int
                final_bbox = (
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                )

                # Write to bbox
                bm.add(
                    study_id=study_id,
                    laterality=laterality,
                    view_position=view_position,
                    bbox=final_bbox,
                )


if __name__ == "__main__":
    update_bbox()
