"""
inference.py
------------
Command-line single-image inference.

Usage:
    python src/inference.py --image path/to/scan.jpg --checkpoint models/best_model.pth
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import load_checkpoint_for_inference
from dataset import IMAGENET_MEAN, IMAGENET_STD, clahe_normalize, auto_crop_to_brain


def preprocess(image: Image.Image, img_size: int) -> torch.Tensor:
    # auto_crop_to_brain and CLAHE MUST match what training used (see
    # dataset.py) — applying different preprocessing at inference than
    # training was done with would make predictions worse, not better.
    tf = transforms.Compose([
        transforms.Lambda(auto_crop_to_brain),
        transforms.Lambda(clahe_normalize),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(image.convert("RGB")).unsqueeze(0)


def predict(model, class_names, img_size, image: Image.Image, device):
    """
    Uses test-time augmentation (TTA): averages softmax probabilities over the
    original image and its horizontal flip, a free accuracy boost that needs
    no retraining.
    """
    tensor = preprocess(image, img_size).to(device)
    flipped_tensor = torch.flip(tensor, dims=[3])

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze(0)
        flipped_probs = F.softmax(model(flipped_tensor), dim=1).squeeze(0)
        avg_probs = ((probs + flipped_probs) / 2.0).cpu()

    ranked = sorted(zip(class_names, avg_probs.tolist()), key=lambda x: x[1], reverse=True)
    return ranked  # list of (class_name, probability), most likely first


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an MRI image (jpg/png)")
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, img_size = load_checkpoint_for_inference(args.checkpoint, device)

    image = Image.open(args.image)
    ranked = predict(model, class_names, img_size, image, device)

    print(f"\nImage: {args.image}")
    print("-" * 40)
    for cls, prob in ranked:
        bar = "#" * int(prob * 30)
        print(f"{cls:>15s}: {prob*100:5.1f}%  {bar}")
    print("-" * 40)
    top_cls, top_prob = ranked[0]
    print(f"Prediction: {top_cls} ({top_prob*100:.1f}% confidence)\n")
    print("NOTE: research/educational tool only — not a medical device, "
          "not a substitute for diagnosis by a qualified radiologist.")


if __name__ == "__main__":
    main()
