# src/scripts/visualize_bbox.py

from __future__ import annotations
from typing import List, Tuple

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.data.status import Status
from src.data.breast_level import BreastLevel
from src.data.metadata import Metadata
from src.data.finding import Finding
from src.preprocessing.dicom import load_dicom, to_uint8


def draw_bboxes(
        img: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        *,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 5,
) -> np.ndarray:
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color=color,
            thickness=thickness,
        )

    return img


def display(
        img: np.ndarray,
        title: str,
        mode: str = "plt",
        out_dir: str | None = None,
):
    if mode == "plt":
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(title, fontsize=9)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    elif mode == "save":
        assert out_dir is not None
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{title}.png")
        cv2.imwrite(out_path, img)


def visualize_raw(
        *,
        study_id: str,
        laterality: str,
        view_position: str,
        mode: str = "plt",
        out_dir: str | None = None,
):
    status = Status()
    breast_level = BreastLevel()
    metadata = Metadata()
    finding = Finding()

    title = f"{study_id}_{laterality}_{view_position}"

    # Get dicom_path
    dicom_path = status.by_index.get(
        study_id,
        laterality,
        view_position,
    )["downloaded_path"]

    # Get image_id
    images = breast_level.by_study_id.get_image_id(study_id)
    image_id = images[(laterality, view_position)]

    # Get metadata
    meta = metadata.by_image_id.get(image_id)

    img = load_dicom(
        dicom_path,
        photometric_interpretation=meta.get("photometric_interpretation", "MONOCHROME2"),
    )

    # Convert to grayscale unit8
    img = to_uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    display(
        img,
        title + "_raw",
        mode,
        out_dir,
    )

    # Get bbox
    bboxes = finding.by_index.get_bbox(
        study_id,
        laterality,
        view_position,
    )

    # Draw bboxes
    vis = draw_bboxes(img, bboxes)

    display(
        vis,
        title + "_bbox",
        mode,
        out_dir,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize raw images and lesion bounding boxes"
    )

    parser.add_argument("--study-id", type=str, required=True)
    parser.add_argument("--laterality", type=str, required=True)
    parser.add_argument("--view-position", type=str, required=True)

    parser.add_argument(
        "--mode",
        type=str,
        choices=["plt", "save"],
        default="plt",
        help="plt: show with matplotlib; save: save to file",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory when mode=save",
    )

    args = parser.parse_args()

    visualize_raw(
        study_id=args.study_id,
        laterality=args.laterality,
        view_position=args.view_position,
        mode=args.mode,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

"""
python -m scripts.visualize_raw --study-id 5ae06adb1a2b2bcfeeef2ba52c234c96 --laterality L --view-position CC
python -m scripts.visualize_raw --study-id 5ae06adb1a2b2bcfeeef2ba52c234c96 --laterality L --view-position MLO
python -m scripts.visualize_raw --study-id 5ae06adb1a2b2bcfeeef2ba52c234c96 --laterality R --view-position CC
python -m scripts.visualize_raw --study-id 5ae06adb1a2b2bcfeeef2ba52c234c96 --laterality R --view-position MLO

python -m scripts.visualize_raw --study-id 8a25c49c269a45ae002ad52e9e88dd94 --laterality L --view-position CC
python -m scripts.visualize_raw --study-id 8a25c49c269a45ae002ad52e9e88dd94 --laterality L --view-position MLO
python -m scripts.visualize_raw --study-id 8a25c49c269a45ae002ad52e9e88dd94 --laterality R --view-position CC
python -m scripts.visualize_raw --study-id 8a25c49c269a45ae002ad52e9e88dd94 --laterality R --view-position MLO

python -m scripts.visualize_raw --study-id 2f4d26ae21e1fb85ec2d97f9464aadff --laterality L --view-position CC
python -m scripts.visualize_raw --study-id 2f4d26ae21e1fb85ec2d97f9464aadff --laterality L --view-position MLO
python -m scripts.visualize_raw --study-id 2f4d26ae21e1fb85ec2d97f9464aadff --laterality R --view-position CC
python -m scripts.visualize_raw --study-id 2f4d26ae21e1fb85ec2d97f9464aadff --laterality R --view-position MLO

final:
python -m scripts.visualize_raw --study-id 2f4d26ae21e1fb85ec2d97f9464aadff --laterality L --view-position CC --mode save --out-dir ./visualize
python -m scripts.visualize_raw --study-id 2f4d26ae21e1fb85ec2d97f9464aadff --laterality L --view-position MLO --mode save --out-dir ./visualize
python -m scripts.visualize_raw --study-id 2f4d26ae21e1fb85ec2d97f9464aadff --laterality R --view-position CC --mode save --out-dir ./visualize
python -m scripts.visualize_raw --study-id 2f4d26ae21e1fb85ec2d97f9464aadff --laterality R --view-position MLO --mode save --out-dir ./visualize
"""
