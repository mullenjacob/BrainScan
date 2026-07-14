"""
gradcam.py
----------
Grad-CAM heatmap generation + bounding-box extraction for the tumor classifier.

The classifier only predicts a class — it has no native concept of "where" in
the image that class comes from. Grad-CAM approximates this after the fact by
looking at which spatial regions of the last convolutional feature map most
influenced the predicted class's score (via gradients), producing a coarse
heatmap. We then threshold that heatmap and box the hottest connected
region(s).

Important: this is a well-known interpretability technique showing "where the
model focused," not a validated tumor boundary/segmentation. Treat boxes as
an explanation aid, not a clinical measurement.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Conv2d:
    """
    Finds the literal last Conv2d layer in the model. Kept as a fallback —
    prefer find_target_conv_layer() below for actual Grad-CAM use, since the
    very last conv layer in most CNN backbones has been downsampled so much
    (e.g. a 7x7 grid for a 224x224 input) that Grad-CAM boxes come out coarse
    and blocky, often missing the visually obvious lesion.
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found in model — can't compute Grad-CAM.")
    return last_conv


def find_target_conv_layer(model: torch.nn.Module, input_size: int = 224, min_spatial: int = 14) -> torch.nn.Conv2d:
    """
    Picks the LAST Conv2d layer whose output spatial resolution is still at
    least `min_spatial` x `min_spatial`, instead of the literal last conv
    layer. This trades a bit of semantic depth for much better localization
    precision: the final conv layer of most backbones (mobilenet, resnet,
    efficientnet) has been downsampled to a 7x7 grid for a 224x224 input,
    where each heatmap cell covers a 32x32 pixel block — too coarse to box a
    lesion tightly. Backing up to a ~14x14-or-better layer roughly doubles
    resolution while staying deep enough in the network to still reflect the
    class decision reasonably well.

    Works across backbones automatically (mobilenet_v3_*, resnet18,
    efficientnet_b0, ...) by empirically probing spatial sizes with a dummy
    forward pass, rather than hardcoding layer names per architecture.
    """
    spatial_sizes = {}

    def make_hook(name):
        def hook(module, inp, out):
            spatial_sizes[name] = tuple(out.shape[-2:])
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(1, 3, input_size, input_size, device=device))
    model.train(was_training)

    for h in handles:
        h.remove()

    name_to_module = dict(model.named_modules())

    target_name = None
    for name, _module in model.named_modules():
        if name in spatial_sizes:
            h, w = spatial_sizes[name]
            if min(h, w) >= min_spatial:
                target_name = name  # keep overwriting -> ends up as the LAST qualifying layer

    if target_name is None:
        # Nothing meets the bar (unusually aggressive downsampling) — fall back
        # to whichever conv layer has the largest spatial map available.
        target_name = max(spatial_sizes, key=lambda n: min(spatial_sizes[n]))

    return name_to_module[target_name]


class GradCAM:
    """
    Wraps a trained classifier to produce Grad-CAM heatmaps. Register once at
    startup (it attaches forward/backward hooks to the target layer) and reuse
    across requests — do not create a fresh instance per request.
    """

    def __init__(self, model: torch.nn.Module, input_size: int = 224, target_layer: torch.nn.Module = None):
        self.model = model
        self.target_layer = target_layer or find_target_conv_layer(model, input_size=input_size)
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        input_tensor: 1x3xHxW (already normalized, on the right device).
        Returns a HxW float32 heatmap normalized to [0, 1], resized to match
        input_tensor's spatial size.

        Uses Grad-CAM++ rather than vanilla Grad-CAM. Vanilla Grad-CAM weighs
        each channel by the plain average of its gradients, which tends to
        produce noisier, less tightly-localized maps on architectures built
        from depthwise-separable convolutions (MobileNetV3, EfficientNet) —
        exactly the backbones this project uses. Grad-CAM++ instead weighs
        each *spatial location* by how much it specifically contributes to
        the class score (via a closed-form weighting derived from higher-
        order gradient terms), which generally produces sharper, better-
        localized heatmaps, especially when there's more than one salient
        region in the image. This uses only first-order gradients under the
        hood (no actual second/third-order autograd), so it costs the same
        single backward pass as vanilla Grad-CAM.
        """
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            output = self.model(input_tensor)
            score = output[0, class_idx]
            score.backward()

        activations = self.activations  # 1xCxh'xw'
        gradients = self.gradients      # 1xCxh'xw'

        grads_power_2 = gradients ** 2
        grads_power_3 = grads_power_2 * gradients
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)  # 1xCx1x1

        eps = 1e-6
        denom = 2 * grads_power_2 + sum_activations * grads_power_3 + eps
        alpha = grads_power_2 / denom
        alpha = torch.where(gradients != 0, alpha, torch.zeros_like(alpha))

        weights = (F.relu(gradients) * alpha).sum(dim=(2, 3), keepdim=True)  # 1xCx1x1

        cam = (weights * activations).sum(dim=1, keepdim=True)  # 1x1xh'xw'
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy().astype(np.float32)
        if cam.max() > 0:
            cam = cam / cam.max()

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w, h))
        return cam


def heatmap_to_boxes(heatmap: np.ndarray, threshold: float = 0.55, min_area_frac: float = 0.003, max_boxes: int = 1):
    """
    Thresholds a 0-1 heatmap and returns bounding boxes (x, y, w, h) for the
    `max_boxes` most confident connected regions above `threshold` that each
    cover at least `min_area_frac` of the image (filters out small noisy
    hotspots). "Most confident" is ranked by each region's mean heatmap
    intensity, not just its size — a small, intensely-hot region beats a
    larger, weakly-hot one.

    max_boxes defaults to 1: in practice, Grad-CAM often lights up one clear
    region matching the visible lesion plus a couple of small, weak secondary
    spots (image corners/edges, midline structures) that just add visual
    clutter without adding real information. Set max_boxes higher if you
    want to see multiple distinct regions (e.g. for a case with genuinely
    multifocal findings).

    min_area_frac default is tuned for the ~14x14 Grad-CAM grid this project
    uses (see find_target_conv_layer): each grid cell covers roughly a 16x16
    pixel block, i.e. ~0.5% of a 224x224 image, so the threshold needs to sit
    below that or single-cell (but genuine) activations get filtered out
    entirely as "noise" — which is exactly what happened before this was tuned.
    """
    h, w = heatmap.shape
    mask = (heatmap >= threshold).astype(np.uint8)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    min_area = min_area_frac * h * w
    candidates = []
    for i in range(1, num_labels):  # label 0 is background
        x, y, bw, bh, area = stats[i]
        if area >= min_area:
            region_mean_intensity = float(heatmap[labels == i].mean())
            candidates.append((region_mean_intensity, (int(x), int(y), int(bw), int(bh))))

    candidates.sort(key=lambda c: c[0], reverse=True)
    if max_boxes is not None:
        candidates = candidates[:max_boxes]

    return [box for _, box in candidates]


def scale_boxes(boxes, from_size, to_size):
    """Rescales a list of (x, y, w, h) boxes from `from_size` to `to_size` ((w, h) tuples)."""
    fw, fh = from_size
    tw, th = to_size
    sx, sy = tw / fw, th / fh
    return [(int(x * sx), int(y * sy), int(w * sx), int(h * sy)) for (x, y, w, h) in boxes]


def is_no_tumor_class(class_name: str) -> bool:
    """Recognizes common 'no tumor' class-name spellings without hardcoding one exact string."""
    normalized = class_name.lower().replace("_", "").replace("-", "").replace(" ", "")
    return normalized in {"notumor", "normal", "healthy", "none", "negative"}
