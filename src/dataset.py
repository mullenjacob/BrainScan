"""
dataset.py
----------
Data loading utilities for the brain tumor MRI dataset.

Expects the Kaggle "Brain Tumor MRI Dataset" folder layout:

    data/Training/<class_name>/*.jpg
    data/Testing/<class_name>/*.jpg

Class folder names are whatever the dataset ships with (commonly
glioma, meningioma, notumor, pituitary) — torchvision.datasets.ImageFolder
picks them up automatically, so nothing here is hardcoded to specific names.
"""

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_brain_crop_box(image: Image.Image):
    """
    Finds the bounding box of the largest bright connected region (the
    skull/brain) in `image`. Returns (x0, y0, x1, y1) in the image's own
    pixel coordinates, or None if no clear dominant region is found (caller
    should treat None as "don't crop, use the image as-is").

    Split out from auto_crop_to_brain() so callers that need to map
    coordinates (e.g. Grad-CAM box overlays) back through the crop can do
    so — auto_crop_to_brain() itself just returns the cropped image, which
    is all the training/eval transform pipeline needs.
    """
    gray = np.array(image.convert("L"))
    h, w = gray.shape

    _thresh_val, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Some raw scanner/viewer exports have a thin bright border/frame running
    # around the ENTIRE image. Without this step, that frame connects to the
    # head into one giant contour spanning almost the whole image, so the
    # crop ends up doing nothing at all. Eroding first breaks thin connecting
    # lines (the frame, typically only a few pixels wide) while barely
    # affecting the much thicker head blob — this is a standard trick for
    # separating "thin connector" from "solid region" in a binary mask.
    erosion_kernel_size = 9
    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # If the largest bright region is only a small sliver of the image, this
    # probably isn't a real "brain blob vs scanner furniture" situation —
    # bail out rather than risk cropping something real. Real skull/brain
    # content in this dataset typically fills a large fraction of the frame.
    if area < 0.15 * h * w:
        return None

    x, y, bw, bh = cv2.boundingRect(largest)

    # Padding: the usual small margin, PLUS extra to compensate for the
    # erosion above shrinking the detected region by roughly half the
    # erosion kernel's size on each side.
    erosion_compensation = erosion_kernel_size // 2 + 2
    pad = int(0.02 * max(w, h)) + erosion_compensation
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)

    return (x0, y0, x1, y1)


def auto_crop_to_brain(image: Image.Image) -> Image.Image:
    """
    Crops the image down to the largest bright connected region (the
    skull/brain), discarding anything outside it — scale rulers, patient
    info text, DICOM parameter overlays, and other burned-in annotations
    common in raw clinical exports. These sit outside the skull and are
    disconnected from it, so isolating the largest connected bright region
    reliably separates "brain" from "scanner furniture."

    Falls back to returning the original image unchanged if no clear
    dominant region is found — this is a safety net, not a guess.

    Applied before clahe_normalize, and identically at training and
    inference time, for the same reason CLAHE must match: mismatched
    preprocessing between training and inference makes things worse, not
    better.
    """
    crop_box = find_brain_crop_box(image)
    if crop_box is None:
        return image
    return image.crop(crop_box)


def clahe_normalize(image: Image.Image) -> Image.Image:
    """
    Applies CLAHE (Contrast-Limited Adaptive Histogram Equalization) to
    standardize contrast/brightness across images from different sources.

    Why this matters: the training data (a clean, consistently-processed
    Kaggle dataset) has a narrow, consistent contrast/brightness profile. A
    real MRI image pulled from an arbitrary webpage, PDF, or screenshot can
    have very different windowing/contrast/compression characteristics. A
    model trained only on the former can misfire on the latter — not because
    it hasn't seen enough tumor examples, but because the *visual style* of
    the input differs from what it learned on. CLAHE re-normalizes local
    contrast so both the training data and real-world uploads get mapped
    into a more consistent representation before the model ever sees them.

    Applied identically at training time (via build_transforms) and at
    inference time (inference.py, app/server.py) — it MUST be applied both
    places consistently, or the model sees a different distribution at
    inference than it trained on, which would make things worse, not better.
    """
    gray = np.array(image.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def build_transforms(img_size: int):
    # Augmentation is deliberately moderate: MRI scans are consistently
    # framed/oriented (unlike, say, photos of objects), so aggressive crops
    # or rotations can create unrealistic training examples. Horizontal flip
    # is fine (brains are roughly left-right symmetric). RandomErasing helps
    # prevent the model from over-relying on one small region of the image.
    # auto_crop_to_brain and CLAHE normalization run first, on the original
    # untouched image, before any resizing.
    train_tf = transforms.Compose([
        transforms.Lambda(auto_crop_to_brain),
        transforms.Lambda(clahe_normalize),
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.08)),
    ])

    eval_tf = transforms.Compose([
        transforms.Lambda(auto_crop_to_brain),
        transforms.Lambda(clahe_normalize),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tf, eval_tf


def build_datasets(train_dir: str, test_dir: str, img_size: int, val_split: float, seed: int = 42):
    """
    Returns (train_ds, val_ds, test_ds, class_names).

    The train/val split is stratified (proportional class balance preserved
    in both splits) rather than purely random — with 4 classes and a modest
    dataset size, a plain random split can otherwise leave val with a skewed
    class mix, making val_acc a noisier signal for early stopping/model
    selection than it needs to be.

    train_ds gets augmented transforms; val_ds and test_ds get clean eval
    transforms. test_ds is the untouched Kaggle "Testing" folder, used only
    for final reporting.
    """
    train_tf, eval_tf = build_transforms(img_size)

    # Two ImageFolder instances over the same directory: one for augmented
    # training samples, one for clean eval-transform samples (used for val).
    full_train_aug = datasets.ImageFolder(train_dir, transform=train_tf)
    full_train_eval = datasets.ImageFolder(train_dir, transform=eval_tf)
    class_names = full_train_aug.classes

    targets = [label for _, label in full_train_aug.samples]
    indices = list(range(len(full_train_aug)))

    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, random_state=seed, stratify=targets
    )

    train_subset = Subset(full_train_aug, train_idx)
    val_subset = Subset(full_train_eval, val_idx)

    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

    return train_subset, val_subset, test_ds, class_names
