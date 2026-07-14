"""
train.py
--------
Trains the brain tumor MRI classifier.

Usage:
    python src/train.py --config config.yaml

Run this either on the Jetson Orin Nano (it has enough GPU for this small
dataset/model) or on a laptop/PC with a CUDA GPU if you'd rather train off-device
and just copy models/best_model.pth over afterward.
"""

import argparse
import copy
import os
import sys
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import build_datasets
from model import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False, desc="train" if train else "eval"):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["train"]["seed"])

    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)

    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = build_datasets(
        train_dir=cfg["data"]["train_dir"],
        test_dir=cfg["data"]["test_dir"],
        img_size=cfg["data"]["img_size"],
        val_split=cfg["data"]["val_split"],
        seed=cfg["train"]["seed"],
    )
    print(f"Classes found: {class_names}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # Class counts + inverse-frequency weights. Matters especially once extra
    # data (e.g. the LGG FLAIR glioma/notumor set) has been mixed in — that
    # tends to make some classes much larger than others, and an unweighted
    # loss would let the model just get good at the oversized classes rather
    # than actually learning to tell all four apart.
    train_targets = [train_ds.dataset.targets[i] for i in train_ds.indices]
    class_counts = torch.bincount(torch.tensor(train_targets), minlength=len(class_names)).float()
    print("Class counts in training set:")
    for name, count in zip(class_names, class_counts.tolist()):
        print(f"  {name}: {int(count)}")

    class_weights = (1.0 / class_counts.clamp(min=1))
    class_weights = class_weights / class_weights.sum() * len(class_names)  # normalize, mean weight ~1
    class_weights = class_weights.to(device)
    print(f"Class weights (loss): {dict(zip(class_names, [round(w, 3) for w in class_weights.tolist()]))}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        num_classes=len(class_names),
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=cfg["train"].get("label_smoothing", 0.0)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None
    epochs_since_improve = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{cfg['train']['epochs']} "
            f"({dt:.1f}s) | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= cfg["train"]["early_stop_patience"]:
            print(f"No val improvement for {epochs_since_improve} epochs, stopping early.")
            break

    # Restore best weights before final test evaluation + saving
    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nEvaluating best model on held-out Testing set...")
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"Test loss={test_loss:.4f}  Test accuracy={test_acc:.4f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "backbone": cfg["model"]["backbone"],
        "img_size": cfg["data"]["img_size"],
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
    }
    torch.save(checkpoint, cfg["paths"]["best_checkpoint"])
    print(f"Saved checkpoint to {cfg['paths']['best_checkpoint']}")

    _plot_history(history, cfg["paths"]["history_plot"])
    _plot_confusion_matrix(model, test_loader, class_names, device, cfg["paths"]["confusion_matrix_plot"])


def _plot_history(history, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved training curves to {out_path}")


def _plot_confusion_matrix(model, loader, class_names, device, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Testing set)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
