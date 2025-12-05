# scripts/visualize_preprocess.py

from __future__ import annotations
from typing import Dict, Any

import os
import argparse
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.data.breast_level import BreastLevel
from src.data.metadata import Metadata
from src.data.status import Status
from src.preprocessing.dicom import load_dicom, to_uint8, remove_corner_text
from src.preprocessing.orientation import canonicalize_laterality
from src.preprocessing.crop import fixed_roi_crop
from src.preprocessing.intensity import clahe_equalize

from config import IMG_ROOT, PROCESSED_DIR


def run_preprocess_flow(
        dicom_path: str,
        metadata: Dict[str, Any],
):
    results = {}

    img = load_dicom(
        dicom_path,
        photometric_interpretation=metadata["photometric_interpretation"],
    )
    results["raw"] = img.copy()

    img, _ = remove_corner_text(img)
    results["cleaned"] = img.copy()

    img = canonicalize_laterality(
        img,
        laterality=metadata["laterality"],
    )
    results["oriented"] = img.copy()

    img, bbox = fixed_roi_crop(
        img,
    )
    results["cropped"] = img.copy()
    results["bbox"] = bbox

    img = clahe_equalize(
        img,
    )
    results["enhanced"] = img.copy()

    return results


def show_preprocess_flow(
        results: Dict[str, Any],
        title: str = None,
):
    stages = ["raw", "cleaned", "oriented", "cropped", "enhanced"]

    plt.figure(figsize=(4 * len(stages), 4))

    for i, k in enumerate(stages):
        plt.subplot(1, len(stages), i + 1)
        disp = to_uint8(results[k])
        plt.imshow(disp, cmap="gray")
        plt.title(k)
        plt.axis("off")

    if title:
        plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()


def save_preprocess_flow(
        results: Dict[str, Any],
        out_dir: str,
        prefix: str,
):
    os.makedirs(out_dir, exist_ok=True)

    for k, img in results.items():
        if k == "bbox":
            continue

        out_path = os.path.join(out_dir, f"{prefix}_{k}.png")
        cv2.imwrite(out_path, to_uint8(img))


# Visualize given preprocessed image
def show_preprocessed_image(
        npz_path: str,
        mode: str,
        out_dir: str | None,
):
    if not os.path.exists(npz_path):
        raise RuntimeError(f"Invalid preprocessed image path: {npz_path}")

    data = np.load(npz_path)

    img = data["img"].astype(np.float32)

    title = os.path.basename(npz_path).replace(".npz", "")

    if mode == "plt":
        plt.figure(figsize=(4, 4))
        plt.imshow(to_uint8(img), cmap="gray")
        plt.title(title, fontsize=8)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    elif mode == "save":
        assert out_dir is not None
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{title}.png")
        cv2.imwrite(out_path, to_uint8(img))


# Visualize random preprocessed samples
def show_preprocessed_samples(
        preprocessed_dir: str,
        num_samples: int = 8,
        mode: str = "plt",
        out_dir: str | None = None,
):
    files = [
        f for f in os.listdir(preprocessed_dir)
        if f.endswith(".npz")
    ]

    if len(files) == 0:
        raise RuntimeError(f"No .npz files found in {preprocessed_dir}")

    samples = random.sample(
        files,
        k=min(num_samples, len(files)),
    )

    imgs = []
    titles = []

    for fname in samples:
        path = os.path.join(preprocessed_dir, fname)
        data = np.load(path)

        img = data["img"].astype(np.float32)

        imgs.append(img)
        titles.append(fname.replace(".npz", ""))

    if mode == "plt":
        plt.figure(figsize=(4 * len(imgs), 4))
        for i, (img, title) in enumerate(zip(imgs, titles)):
            plt.subplot(1, len(imgs), i + 1)
            plt.imshow(to_uint8(img), cmap="gray")
            plt.title(title, fontsize=8)
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    elif mode == "save":
        assert out_dir is not None
        os.makedirs(out_dir, exist_ok=True)

        for img, title in zip(imgs, titles):
            out_path = os.path.join(out_dir, f"{title}.png")
            cv2.imwrite(out_path, to_uint8(img))


# Visualize preprocess flow from raw images (.dicom)
def visualize_from_raw(
        study_id: str,
        laterality: str,
        view_position: str,
        dicom_path: str,
        meta: Dict[str, Any],
        mode: str,
        out_dir: str | None,
):
    results = run_preprocess_flow(dicom_path, meta)

    prefix = f"{study_id}_{laterality}_{view_position}"

    if mode == "plt":
        show_preprocess_flow(
            results,
            title=prefix,
        )
    elif mode == "save":
        assert out_dir is not None
        save_preprocess_flow(
            results,
            out_dir=out_dir,
            prefix=prefix,
        )


def main(args):
    status = Status()
    metadata = Metadata()
    breast_level = BreastLevel()

    # Visualize already preprocessed images (.npz)
    from_preprocessed = args.from_preprocessed
    num_samples = args.num_samples
    preprocessed_dir = args.preprocessed_dir

    mode = args.mode
    out_dir = args.out_dir

    # Visualize preprocessed samples, no image index needed
    if from_preprocessed and num_samples > 0:
        assert preprocessed_dir is not None
        show_preprocessed_samples(
            preprocessed_dir=preprocessed_dir,
            num_samples=num_samples,
            mode=mode,
            out_dir=out_dir,
        )
        return

    # Resolve image index: image_id / (study_id, laterality, view_position)
    if args.image_id is not None:
        image_id = args.image_id
        d = status.by_image_id.get(image_id)
        study_id = d["study_id"]
        laterality = d["laterality"]
        view_position = d["view_position"]
        dicom_path = d["downloaded_path"]
        npz_path = d["preprocessed_path"]
        meta = metadata.by_image_id.get(image_id)
    else:
        assert args.study_id and args.laterality and args.view_position
        study_id = args.study_id
        laterality = args.laterality
        view_position = args.view_position
        images = breast_level.by_study_id.get(study_id)
        image_id = next(
            item["image_id"]
            for item in images if
            item["laterality"] == laterality and item["view_position"] == view_position
        )
        d = status.by_index.get(study_id, laterality, view_position)
        dicom_path = d["downloaded_path"]
        npz_path = d["preprocessed_path"]
        meta = metadata.by_image_id.get(image_id)

    if from_preprocessed:
        show_preprocessed_image(
            npz_path=npz_path,
            mode=mode,
            out_dir=out_dir,
        )

    else:
        visualize_from_raw(
            study_id=study_id,
            laterality=laterality,
            view_position=view_position,
            dicom_path=dicom_path,
            meta=meta,
            mode=mode,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize preprocessing pipeline"
    )

    # Identification
    parser.add_argument("--image-id", type=str, default=None)
    parser.add_argument("--study-id", type=str, default=None)
    parser.add_argument("--laterality", type=str, default=None)
    parser.add_argument("--view-position", type=str, default=None)

    # Show preprocessed image (.npz)
    parser.add_argument(
        "--from-preprocessed",
        action="store_true",
        help="Visualize preprocessed images (.npz)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help=">0: number of random samples; =0: use given image",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default=PROCESSED_DIR,
        help="Directory containing preprocessed images",
    )

    # Output control
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
    main(args)

"""
random_samples:
python -m scripts.visualize_preprocess --from-preprocessed

bbox_case:
python -m scripts.visualize_preprocess --study-id 02bd0bd83c6d9fedc49b0df6ecd952c6 --laterality L --view-position MLO
python -m scripts.visualize_preprocess --from-preprocessed --num-samples 0 --study-id 02bd0bd83c6d9fedc49b0df6ecd952c6 --laterality L --view-position MLO

text_case:
python -m scripts.visualize_preprocess --study-id 0aef9de26f71136b4b078408e922bd26 --laterality L --view-position CC

lesion_case:
python -m scripts.visualize_preprocess --study-id 2ad573bf0a419ab289c3a23c6a6db18f --laterality L --view-position CC

normalization_sensitive_case:
python -m scripts.visualize_preprocess --study-id 0a83f94db524266b567882c2695a4cfe --laterality L --view-position CC

black_margin_case:
python -m scripts.visualize_preprocess --study-id 42162c2fa52a4927562b812d5b87603c --laterality L --view-position CC

view_position_error_case:
python -m scripts.visualize_preprocess --study-id dbca9d28baa3207b3187c4d07dc81a80 --laterality L --view-position CC

huge_breast_case:
python -m scripts.visualize_preprocess --study-id 2e59c9d87903cf06b7c1d42fa0bb744c --laterality L --view-position MLO
python -m scripts.visualize_preprocess --study-id 2e59c9d87903cf06b7c1d42fa0bb744c --laterality R --view-position MLO

super_huge_breast_case:
python -m scripts.visualize_preprocess --study-id 5b3e8561d3617327a027a25250049e29 --laterality L --view-position MLO
python -m scripts.visualize_preprocess --study-id 5b3e8561d3617327a027a25250049e29 --laterality R --view-position MLO

white_corner_case:
python -m scripts.visualize_preprocess --study-id a39afdb69c38ea994a945ba2dbbd0ffd --laterality R --view-position CC
python -m scripts.visualize_preprocess --from-preprocessed --num-samples 0 --study-id a39afdb69c38ea994a945ba2dbbd0ffd --laterality R --view-position CC

final display:
python -m scripts.visualize_preprocess --study-id 2e59c9d87903cf06b7c1d42fa0bb744c --laterality L --view-position MLO --mode save --out-dir ./visualize
"""
