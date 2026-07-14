"""
evaluate_external.py
---------------------
Evaluates the trained model specifically on data/ExternalValidation — the
held-out FLAIR slices from the LGG dataset that were NOT used in training
(see scripts/prepare_lgg_data.py for the patient-level split).

This is kept separate from the accuracy reported by train.py on purpose:
that number reflects performance on the original T1-contrast-enhanced
distribution the model has always done reasonably well on. This script
answers a different, more honest question — does the model actually
generalize to a genuinely different MRI sequence it saw some (but held-out)
examples of during training?

Only glioma/notumor are evaluated here, since the LGG dataset doesn't include
meningioma or pituitary examples — this measures the sequence-generalization
gap specifically, not overall accuracy.

Usage:
    python src/evaluate_external.py --checkpoint models/best_model.pth
"""

import argparse
import os
import sys

import torch
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import load_checkpoint_for_inference
from dataset import IMAGENET_MEAN, IMAGENET_STD, clahe_normalize, auto_crop_to_brain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    parser.add_argument(
        "--data_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ExternalValidation"),
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: {args.data_dir} not found.")
        print("Run scripts/download_lgg_dataset.py and scripts/prepare_lgg_data.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, img_size = load_checkpoint_for_inference(args.checkpoint, device)

    eval_tf = transforms.Compose([
        transforms.Lambda(auto_crop_to_brain),
        transforms.Lambda(clahe_normalize),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    dataset = datasets.ImageFolder(args.data_dir, transform=eval_tf)
    external_classes = dataset.classes  # should be ['glioma', 'notumor']
    print(f"External validation set: {len(dataset)} images across classes {external_classes}")

    # Map the external dataset's class indices to this model's class indices,
    # since the model has 4 output classes but this validation set only
    # covers 2 of them.
    model_class_to_idx = {name: i for i, name in enumerate(class_names)}

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            image_tensor, external_label_idx = dataset[i]
            external_label_name = external_classes[external_label_idx]

            logits = model(image_tensor.unsqueeze(0).to(device))
            pred_idx = int(logits.argmax(dim=1).item())
            pred_name = class_names[pred_idx]

            all_preds.append(pred_name)
            all_labels.append(external_label_name)

            if (i + 1) % 200 == 0:
                print(f"  evaluated {i + 1}/{len(dataset)}...")

    print("\n" + "=" * 60)
    print("External (FLAIR) validation results — glioma vs notumor only")
    print("=" * 60)

    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    accuracy = correct / len(all_labels) if all_labels else 0.0
    print(f"Accuracy: {accuracy:.4f}  ({correct}/{len(all_labels)})")

    # Full transparency: show every actual-vs-predicted combination, including
    # cases where the model predicted meningioma/pituitary on these
    # glioma/notumor-only images (which only exist here as "wrong answers").
    print("\nActual -> Predicted breakdown:")
    for actual_label in ["glioma", "notumor"]:
        subset_preds = [p for p, l in zip(all_preds, all_labels) if l == actual_label]
        if not subset_preds:
            continue
        print(f"  Actual={actual_label} (n={len(subset_preds)}):")
        for pred_label in class_names:
            count = subset_preds.count(pred_label)
            if count > 0:
                pct = 100 * count / len(subset_preds)
                print(f"    -> predicted {pred_label}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
