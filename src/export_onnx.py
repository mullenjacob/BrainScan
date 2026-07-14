"""
export_onnx.py
--------------
Exports the trained PyTorch checkpoint to ONNX for faster/lighter inference
(and as the input format for building a TensorRT engine on the Jetson).

Usage:
    python src/export_onnx.py --checkpoint models/best_model.pth --out models/brain_tumor_model.onnx

After exporting, you can (optionally, for max speed on the Jetson) build a
TensorRT engine with the `trtexec` CLI tool that ships with JetPack:

    trtexec --onnx=models/brain_tumor_model.onnx \\
            --saveEngine=models/brain_tumor_model.engine \\
            --fp16

That .engine file can then be loaded with TensorRT's Python API for the
fastest possible inference. This is optional — plain ONNX Runtime or even
plain PyTorch is fast enough for the still-image / few-fps use cases this
project targets.
"""

import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import load_checkpoint_for_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    parser.add_argument("--out", default="models/brain_tumor_model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    device = torch.device("cpu")  # export on CPU is fine and avoids device quirks
    model, class_names, img_size = load_checkpoint_for_inference(args.checkpoint, device)

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        args.out,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=args.opset,
    )

    print(f"Exported ONNX model to {args.out}")
    print(f"Classes (output order): {class_names}")
    print(f"Expected input size: 1x3x{img_size}x{img_size} (RGB, ImageNet-normalized)")
    print("\nOptional next step for max speed on Jetson (run ON the Jetson):")
    print(f"  trtexec --onnx={args.out} "
          f"--saveEngine={args.out.replace('.onnx', '.engine')} --fp16")


if __name__ == "__main__":
    main()
