"""
download_dataset.py
--------------------
Downloads and unpacks the Kaggle "Brain Tumor MRI Dataset" into data/.

Requires a Kaggle API token (kaggle.json). See README.md "Getting the dataset"
section for how to obtain one — Kaggle requires authentication to download
datasets, so this can't be fully automated without your credentials.

Usage:
    python scripts/download_dataset.py
"""

import os
import subprocess
import sys
import zipfile

DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATA_DIR = os.path.abspath(DATA_DIR)


def check_kaggle_credentials():
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json) and not (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    ):
        print("ERROR: No Kaggle API credentials found.")
        print("Expected either:")
        print("  - a token file at ~/.kaggle/kaggle.json, or")
        print("  - the KAGGLE_USERNAME and KAGGLE_KEY environment variables set")
        print("\nSee README.md 'Getting the dataset' for step-by-step instructions.")
        sys.exit(1)


def main():
    check_kaggle_credentials()
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Downloading {DATASET} into {DATA_DIR} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", DATA_DIR],
        check=True,
    )

    zip_path = os.path.join(DATA_DIR, "brain-tumor-mri-dataset.zip")
    if not os.path.exists(zip_path):
        # Kaggle sometimes names the zip after the dataset slug's last segment
        candidates = [f for f in os.listdir(DATA_DIR) if f.endswith(".zip")]
        if not candidates:
            print("ERROR: Download finished but no .zip file was found in data/.")
            sys.exit(1)
        zip_path = os.path.join(DATA_DIR, candidates[0])

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    print("Done. Expected structure:")
    print(f"  {DATA_DIR}/Training/<class_name>/*.jpg")
    print(f"  {DATA_DIR}/Testing/<class_name>/*.jpg")
    print("\nIf the extracted folders have different names/nesting, either rename")
    print("them to match config.yaml's data.train_dir / data.test_dir, or edit")
    print("those two config values to match what you actually have.")


if __name__ == "__main__":
    main()
