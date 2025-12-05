# script/download.py

from __future__ import annotations

import os
import getpass
import pydicom
import platform
import subprocess
from tqdm import tqdm

from src.data.breast_level import BreastLevel
from src.data.metadata import Metadata
from src.data.status import Status
from src.utils.log_utils import log, log_section

from config import IMG_ROOT


# Sleep prevention
class SleepGuard:

    def __init__(self):
        self.system = platform.system()

        self.proc = None

    def start(self):
        if self.system == "Darwin":
            log("macOS detected, starting caffeinate", "SleepGuard")
            self.proc = subprocess.Popen(["caffeinate", "-ims"])
        else:
            log(f"Unknown system: {self.system}", "SleepGuard")

    def stop(self):
        if self.proc is not None:
            log("Terminating...", "SleepGuard")
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None
        log("SleepGuard stopped.", "SleepGuard")


# Download dataset
def main(
        img_root: str = IMG_ROOT,
        base_url: str = "https://physionet.org/files/vindr-mammo/1.0.0",
        username: str = None,
        password: str = None,
        max_retries: int = 3,
        keep_awake: bool = True,
):
    status = Status()
    breast_level = BreastLevel()

    if username is None:
        username = input("Enter username: ")
    if password is None:
        password = getpass.getpass("Enter password: ")

    # Get undownloaded index (study_id, laterality, view_position)
    missing = status.get_undownloaded()
    log(f"{len(missing)} undownloaded images", "Download")

    if not missing:
        return

    guard = SleepGuard()
    if keep_awake:
        guard.start()

    log_section("Download")
    try:
        with status.download() as dm:
            for item in tqdm(missing):
                study_id = item["study_id"]
                laterality = item["laterality"]
                view_position = item["view_position"]

                # Get image_id
                images = breast_level.by_study_id.get_image_id(study_id)
                image_id = images[(laterality, view_position)]

                # Out dir
                out_dir = os.path.join(img_root, study_id)
                os.makedirs(out_dir, exist_ok=True)

                # Out path
                out_path = os.path.join(out_dir, f"{image_id}.dicom")

                # If the image exists, try to open it
                if os.path.exists(out_path):
                    try:
                        pydicom.dcmread(out_path, stop_before_pixels=True)

                        dm.mark(
                            study_id=study_id,
                            laterality=laterality,
                            view_position=view_position,
                            downloaded_path=out_path,
                        )
                        continue
                    except Exception:
                        os.remove(out_path)

                url = f"{base_url}/images/{study_id}/{image_id}.dicom"

                ok = False
                for attempt in range(1, max_retries + 1):
                    try:
                        subprocess.run(
                            [
                                "wget",
                                "-q",
                                "--user", username,
                                "--password", password,
                                "-O", out_path,
                                url,
                            ],
                            check=True,
                        )
                        ok = True
                        break

                    except subprocess.CalledProcessError:
                        log(
                            f"Retry {attempt}/{max_retries} failed for {image_id}",
                            "Download"
                        )

                if not ok:
                    log(f"Failed to download {image_id}", "Download")
                    continue

                dm.mark(
                    study_id=study_id,
                    laterality=laterality,
                    view_position=view_position,
                    downloaded_path=out_path,
                )

    finally:
        if keep_awake:
            guard.stop()


if __name__ == "__main__":
    main()
