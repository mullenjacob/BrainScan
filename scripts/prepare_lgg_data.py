"""
prepare_lgg_data.py
--------------------
Processes the raw LGG Segmentation Dataset (downloaded by
download_lgg_dataset.py) into this project's class-folder format, using each
slice's segmentation mask to derive a label automatically:

    mask has any tumor pixels  -> glioma
    mask is entirely empty     -> notumor

Splits by PATIENT (not by individual slice) into a training portion and a
held-out "external validation" portion. This matters: multiple slices from
the same patient look very similar, so splitting by slice instead of patient
would leak information between train and validation, making any accuracy
measured on the validation portion an overestimate of true generalization.

Output structure:
    data/Training/glioma/lgg_<patient>_<slice>.jpg       (added to existing training data)
    data/Training/notumor/lgg_<patient>_<slice>.jpg
    data/ExternalValidation/glioma/lgg_<patient>_<slice>.jpg   (held out, FLAIR-only)
    data/ExternalValidation/notumor/lgg_<patient>_<slice>.jpg

The ExternalValidation folder is kept separate from the existing Kaggle
"Testing" folder on purpose — it's a genuinely different data source (FLAIR,
different patients, different imaging protocol), so evaluating on it
specifically measures whether the model generalizes to FLAIR, rather than
just re-confirming performance on the original T1-contrast-enhanced
distribution it already does well on.

Usage:
    python scripts/prepare_lgg_data.py
"""

import os
import random
import sys

import numpy as np
from PIL import Image

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "lgg_raw"))
TRAINING_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Training"))
EXTERNAL_VAL_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ExternalValidation"))

VAL_FRACTION = 0.2
SEED = 42


def find_mask_pairs(raw_dir):
    """Recursively finds (image_path, mask_path, patient_id) triples."""
    pairs = []
    for root, _dirs, files in os.walk(raw_dir):
        for fname in files:
            if fname.endswith("_mask.tif") or fname.endswith("_mask.tiff"):
                mask_path = os.path.join(root, fname)
                image_fname = fname.replace("_mask.tif", ".tif").replace("_mask.tiff", ".tiff")
                image_path = os.path.join(root, image_fname)
                if os.path.exists(image_path):
                    patient_id = os.path.basename(root)
                    pairs.append((image_path, mask_path, patient_id))
    return pairs


def mask_has_tumor(mask_path: str) -> bool:
    mask = np.array(Image.open(mask_path).convert("L"))
    return bool((mask > 10).any())


def main():
    if not os.path.isdir(RAW_DIR):
        print(f"ERROR: {RAW_DIR} not found. Run scripts/download_lgg_dataset.py first.")
        sys.exit(1)

    pairs = find_mask_pairs(RAW_DIR)
    if not pairs:
        print(f"ERROR: No image/mask pairs found under {RAW_DIR}.")
        print("The dataset's internal folder structure may differ from what this script expects —")
        print("check the extracted contents and adjust find_mask_pairs() if needed.")
        sys.exit(1)

    print(f"Found {len(pairs)} image/mask slice pairs.")

    patient_ids = sorted(set(p[2] for p in pairs))
    rng = random.Random(SEED)
    rng.shuffle(patient_ids)

    n_val_patients = max(1, int(len(patient_ids) * VAL_FRACTION))
    val_patients = set(patient_ids[:n_val_patients])
    train_patients = set(patient_ids[n_val_patients:])

    print(f"Patients: {len(patient_ids)} total -> {len(train_patients)} train, {len(val_patients)} external-val")

    for cls in ("glioma", "notumor"):
        os.makedirs(os.path.join(TRAINING_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(EXTERNAL_VAL_DIR, cls), exist_ok=True)

    counts = {"train": {"glioma": 0, "notumor": 0}, "val": {"glioma": 0, "notumor": 0}}

    for idx, (image_path, mask_path, patient_id) in enumerate(pairs):
        label = "glioma" if mask_has_tumor(mask_path) else "notumor"
        split = "val" if patient_id in val_patients else "train"
        out_dir = EXTERNAL_VAL_DIR if split == "val" else TRAINING_DIR

        out_name = f"lgg_{patient_id}_{idx}.jpg"
        out_path = os.path.join(out_dir, label, out_name)

        img = Image.open(image_path).convert("RGB")
        img.save(out_path, quality=95)

        counts[split][label] += 1

        if (idx + 1) % 500 == 0:
            print(f"  processed {idx + 1}/{len(pairs)} slices...")

    print("\nDone. Slice counts:")
    print(f"  Added to data/Training:         glioma={counts['train']['glioma']}  notumor={counts['train']['notumor']}")
    print(f"  data/ExternalValidation (FLAIR): glioma={counts['val']['glioma']}  notumor={counts['val']['notumor']}")
    print("\nNext step: retrain with 'python src/train.py --config config.yaml'")
    print("Then check FLAIR generalization specifically with:")
    print("  python src/evaluate_external.py --checkpoint models/best_model.pth")


if __name__ == "__main__":
    main()
