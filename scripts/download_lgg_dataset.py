"""
download_lgg_dataset.py
------------------------
Downloads the Kaggle "LGG Segmentation Dataset" (mateuszbuda/lgg-mri-segmentation)
into data/lgg_raw/. This is a genuinely independent dataset from the one this
project already trains on — sourced from TCGA's lower-grade-glioma collection,
not the figshare/SARTAJ/Br35H sources behind brain-tumor-mri-dataset — and its
images are FLAIR-based (pre-contrast, FLAIR, post-contrast composite), unlike
the T1-contrast-enhanced images the model has trained on so far.

Requires the same Kaggle API credentials as scripts/download_dataset.py.

Usage:
    python scripts/download_lgg_dataset.py
"""

import os
import subprocess
import sys
import zipfile

DATASET = "mateuszbuda/lgg-mri-segmentation"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
RAW_DIR = os.path.join(DATA_DIR, "lgg_raw")


def check_kaggle_credentials():
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json) and not (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    ):
        print("ERROR: No Kaggle API credentials found. See README.md 'Getting the dataset'.")
        sys.exit(1)


def main():
    check_kaggle_credentials()
    os.makedirs(RAW_DIR, exist_ok=True)

    print(f"Downloading {DATASET} into {RAW_DIR} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", RAW_DIR],
        check=True,
    )

    zip_candidates = [f for f in os.listdir(RAW_DIR) if f.endswith(".zip")]
    if not zip_candidates:
        print("ERROR: Download finished but no .zip file was found.")
        sys.exit(1)
    zip_path = os.path.join(RAW_DIR, zip_candidates[0])

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)

    print("Done.")
    print(f"Raw LGG data extracted to {RAW_DIR}")
    print("Next step: python scripts/prepare_lgg_data.py")


if __name__ == "__main__":
    main()
