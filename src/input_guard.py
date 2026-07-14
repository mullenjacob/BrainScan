"""
input_guard.py
---------------
A lightweight heuristic guard that checks whether an uploaded image plausibly
looks like a brain MRI scan, as opposed to an ordinary color photo,
screenshot, or other unrelated image.

This is NOT a trained classifier — it's a handful of simple pixel statistics
that catch clearly-wrong uploads without needing extra training data or a
whole separate model:

  - Real color photos (faces, scenery, screenshots) have hue that varies
    across the image — skin tones, sky, clothing, etc. are all different
    colors. A colorized MRI render (e.g. a common cyan or "hot iron" tint
    applied to an otherwise grayscale scan) has high saturation too, but a
    single, consistent hue throughout, since it's just one color mapped onto
    grayscale intensity. So the check is: high saturation AND highly varied
    hue -> real color photo. High saturation with one consistent hue is left
    alone, since that's a legitimate (if stylized) way scans get shared.
  - MRI scans are cropped against a black background, so the image corners
    should be dark. A bright/busy corner (a wall, background clutter, a
    photo's edge) is a second, independent signal something's off. Uses the
    MEDIAN brightness of each corner patch rather than the mean, so a few
    small bright scanner reference markers (dots/crosses that many clinical
    exports have scattered near the image edges) don't skew a mostly-black
    corner into registering as "bright."

It can still be fooled by unusual images, and could in rare cases flag a
legitimate scan — treat it as a helpful guardrail, not a guarantee.
"""

import numpy as np
from PIL import Image


def _circular_hue_std(hue_values: np.ndarray) -> float:
    """
    Standard deviation of a set of hues, treated as circular (0 and 255 wrap
    around to the same point) rather than linear — otherwise, e.g., a mix of
    hues near 0 and near 255 (which are visually almost the same color) would
    register as maximally spread out, when they're actually consistent.
    Returns a value on roughly the same 0-255 scale as PIL's HSV hue channel.
    """
    if hue_values.size == 0:
        return 0.0
    angles = hue_values / 255.0 * 2 * np.pi
    mean_cos = np.mean(np.cos(angles))
    mean_sin = np.mean(np.sin(angles))
    resultant_length = np.sqrt(mean_cos**2 + mean_sin**2)
    circular_std = np.sqrt(max(0.0, -2 * np.log(max(resultant_length, 1e-6))))
    return circular_std * (255 / (2 * np.pi))


def looks_like_mri(
    image: Image.Image,
    saturation_threshold: float = 25.0,
    hue_std_threshold: float = 25.0,
    corner_brightness_threshold: float = 60.0,
):
    """
    Returns (is_mri_like: bool, reason: str). `reason` is empty when
    is_mri_like is True; otherwise it's a short, user-facing explanation of
    which check failed.
    """
    rgb = np.array(image.convert("RGB")).astype(np.float32)
    hsv = np.array(image.convert("HSV")).astype(np.float32)
    saturation = hsv[:, :, 1]
    hue = hsv[:, :, 0]

    mean_saturation = float(saturation.mean())

    # Hue is meaningless noise on near-gray pixels, so only measure hue spread
    # where there's enough saturation for hue to actually mean something.
    saturated_hues = hue[saturation > 15]
    hue_spread = _circular_hue_std(saturated_hues)

    h, w, _ = rgb.shape
    patch = max(4, min(h, w) // 12)
    corners = [
        rgb[:patch, :patch], rgb[:patch, -patch:],
        rgb[-patch:, :patch], rgb[-patch:, -patch:],
    ]
    # Median, not mean: a handful of bright reference-marker pixels in an
    # otherwise-black corner patch would drag a mean up substantially, but
    # barely move a median at all.
    corner_brightness = float(np.median([np.median(c) for c in corners]))

    if mean_saturation > saturation_threshold and hue_spread > hue_std_threshold:
        return False, "This looks like a multi-color photo, not an MRI scan."

    if corner_brightness > corner_brightness_threshold:
        return False, "This image doesn't have the dark cropped background typical of an MRI scan."

    return True, ""
