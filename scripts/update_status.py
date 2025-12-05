# scripts/update_status.py

from src.data.status import Status
from src.data.breast_level import BreastLevel
from src.data.metadata import Metadata
from src.utils.log_utils import log, log_section, log_config

from config import IMG_ROOT, PROCESSED_DIR

breast_level = BreastLevel()
metadata = Metadata()


def get_status(mode: str) -> "Status":
    if mode == "build":
        return Status.build(breast_level=breast_level)
    elif mode == "repair":
        return Status.repair(breast_level=breast_level)
    else:
        return Status()


def update_download(status, downloaded_dir: str):
    log_section("Updating download")
    with status.download() as dm:
        result = dm.update(
            downloaded_dir=downloaded_dir,
            metadata=metadata,
        )
        log(f"Download updated: {downloaded_dir}", "update_status")
        log(f"  Found: {result['found']}", "update_status")
        log(f"  Updated: {result['updated']}", "update_status")


def update_preprocess(status, preprocessed_dir: str):
    log_section("Updating preprocess")
    with status.preprocess() as pm:
        result = pm.update(
            preprocessed_dir=preprocessed_dir,
        )
        log(f"Preprocess updated: {preprocessed_dir}", "update_status")
        log(f"  Found: {result['found']}", "update_status")
        log(f"  Updated: {result['updated']}", "update_status")


def display_info(status):
    log_section("Status info")
    log_config(status.info, "update_status")


if __name__ == "__main__":
    status = get_status("get")

    # update_download(status, IMG_ROOT)
    # update_download(status, "/Volumes/YUNFANSD/images")

    # with status.preprocess() as pm:
    #     pm.reset()
    # update_preprocess(status, PROCESSED_DIR)

    display_info(status)
