# src/scripts/visualize_bbox.py

from __future__ import annotations
from typing import List, Tuple

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.data.status import Status
from src.data.bbox import BBox
from src.preprocessing.dicom import to_uint8


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


def visualize_bbox(
        *,
        study_id: str,
        laterality: str,
        view_position: str,
        mode: str = "plt",
        out_dir: str | None = None,
        image_size: Tuple[int, int] = (224, 224),
):
    status = Status()
    bbox_db = BBox()

    # Resolve paths
    record = status.by_index.get(
        study_id=study_id,
        laterality=laterality,
        view_position=view_position,
    )

    npz_path = record["preprocessed_path"]
    if not os.path.exists(npz_path):
        raise RuntimeError(f"Preprocessed file not found: {npz_path}")

    # Load preprocessed image
    data = np.load(npz_path)
    img = data["img"].astype(np.float32)

    H_img, W_img = img.shape
    H_out, W_out = image_size

    # Load all bboxes for this image
    bboxes_model_space = bbox_db.by_index.get(
        study_id,
        laterality,
        view_position,
    )

    if len(bboxes_model_space) > 0:
        scale_x = W_img / float(W_out)
        scale_y = H_img / float(H_out)

        bboxes_npz_space: List[Tuple[int, int, int, int]] = []
        for (x1, y1, x2, y2) in bboxes_model_space:
            bx1 = x1 * scale_x
            bx2 = x2 * scale_x
            by1 = y1 * scale_y
            by2 = y2 * scale_y
            bboxes_npz_space.append((
                int(round(bx1)),
                int(round(by1)),
                int(round(bx2)),
                int(round(by2)),
            ))
    else:
        bboxes_npz_space = []

    # Convert to grayscale unit8
    img = to_uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw bboxes
    vis = draw_bboxes(img, bboxes_npz_space)

    title = f"{study_id}_{laterality}_{view_position}"

    if mode == "plt":
        plt.figure(figsize=(4, 4))
        plt.imshow(vis)
        plt.title(title, fontsize=9)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    elif mode == "save":
        assert out_dir is not None
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{title}.png")
        cv2.imwrite(out_path, vis)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize lesion bounding boxes on preprocessed images"
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

    visualize_bbox(
        study_id=args.study_id,
        laterality=args.laterality,
        view_position=args.view_position,
        mode=args.mode,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

"""
2 bboxes per image:
python -m scripts.visualize_bbox --study-id 014d843f13a9d86fec945664e2541fe6 --laterality R --view-position MLO

python -m scripts.visualize_bbox --study-id 01c3c13f0b852aed00b5147c21d0e649 --laterality L --view-position CC

python -m scripts.visualize_bbox --study-id 02bd0bd83c6d9fedc49b0df6ecd952c6 --laterality L --view-position CC
python -m scripts.visualize_bbox --study-id 02bd0bd83c6d9fedc49b0df6ecd952c6 --laterality L --view-position MLO

3 bboxes per image:
python -m scripts.visualize_bbox --study-id 0649398d9f65ab43072c2a475964e872 --laterality R --view-position MLO

python -m scripts.visualize_bbox --study-id 08db0293a555cc089d14e6357381d42d --laterality L --view-position MLO

python -m scripts.visualize_bbox --study-id 0ac8b6988e7443dc5e45d601a2f813f1 --laterality R --view-position CC

4 bboxes per image:
python -m scripts.visualize_bbox --study-id 01c3c13f0b852aed00b5147c21d0e649 --laterality L --view-position MLO

python -m scripts.visualize_bbox --study-id 0246c6050121c176d2e8fcf91f22f926 --laterality L --view-position CC

python -m scripts.visualize_bbox --study-id 0246c6050121c176d2e8fcf91f22f926 --laterality L --view-position MLO

final:
python -m scripts.visualize_bbox --study-id 02bd0bd83c6d9fedc49b0df6ecd952c6 --laterality L --view-position MLO --mode save --out-dir ./visualize
"""
