"""
model.py
--------
Defines the CNN used to classify brain MRI scans into tumor categories.

We use transfer learning with a lightweight torchvision backbone. Lightweight
backbones (MobileNetV3) are the default because the Jetson Orin Nano needs to
run inference in real time on limited compute/power. ResNet18 is offered as a
higher-accuracy / heavier alternative.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, backbone: str = "mobilenet_v3_small", pretrained: bool = True) -> nn.Module:
    """
    Build a classifier head on top of a torchvision backbone.

    Args:
        num_classes: number of output classes (e.g. 4 for glioma/meningioma/notumor/pituitary)
        backbone: one of "mobilenet_v3_small", "mobilenet_v3_large", "resnet18"
        pretrained: whether to start from ImageNet-pretrained weights

    Returns:
        an nn.Module ready to train/finetune
    """
    backbone = backbone.lower()

    if backbone == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)

    elif backbone == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)

    elif backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)

    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose from: mobilenet_v3_small, "
            f"mobilenet_v3_large, resnet18, efficientnet_b0"
        )

    return net


def load_checkpoint_for_inference(checkpoint_path: str, device: torch.device):
    """
    Loads a checkpoint saved by train.py and rebuilds the model + class list.

    Returns:
        (model, class_names) — model is in eval() mode on `device`
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    class_names = ckpt["class_names"]
    backbone = ckpt.get("backbone", "mobilenet_v3_small")
    img_size = ckpt.get("img_size", 224)

    model = build_model(num_classes=len(class_names), backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, img_size
