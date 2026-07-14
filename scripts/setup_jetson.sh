#!/bin/bash
# ============================================================
# setup_jetson.sh
# Sets up a Python virtual environment on the Jetson Orin Nano
# (JetPack 6.x, Python 3.10) with a CUDA-enabled PyTorch build.
#
# Run this ON the Jetson itself (e.g. through your VS Code Remote-SSH
# terminal), not on your laptop.
#
# IMPORTANT: Generic `pip install torch` does NOT give you a CUDA build on
# Jetson's ARM64 (aarch64) architecture — it silently installs a CPU-only
# build from PyPI. You need Jetson-specific wheels, which is what this
# script sets up.
# ============================================================
set -e

echo "== Checking JetPack / L4T version =="
if [ -f /etc/nv_tegra_release ]; then
    cat /etc/nv_tegra_release
else
    echo "Warning: /etc/nv_tegra_release not found — are you sure this is a Jetson?"
fi

echo ""
echo "== Creating virtual environment (.venv) =="
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python3 -m pip install --upgrade pip

echo ""
echo "== Installing PyTorch + torchvision for JetPack 6 (CUDA-enabled) =="
echo "Using the Jetson AI Lab pip index, which hosts prebuilt wheels matched"
echo "to JetPack 6.x / Python 3.10 / CUDA 12.6:"
echo ""

# This index serves wheels built specifically for JetPack 6.x + Python 3.10.
# If your JetPack version is older (6.0/6.1) and this fails, see the fallback
# instructions printed below.
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 || {
    echo ""
    echo "!! The jetson-ai-lab index install failed. This usually means your"
    echo "!! JetPack version doesn't match jp6/cu126 exactly, or there's no"
    echo "!! network access to pypi.jetson-ai-lab.io."
    echo "!!"
    echo "!! FALLBACK: download a wheel directly from NVIDIA matched to your exact"
    echo "!! JetPack version from:"
    echo "!!   https://developer.download.nvidia.com/compute/redist/jp/"
    echo "!! (pick the vXX folder matching your JetPack version, e.g. v60 for"
    echo "!! JetPack 6.0, v61 for 6.1, then grab the cp310 (Python 3.10) wheel),"
    echo "!! then run:  pip install --no-cache <path-or-url-to-wheel>.whl"
    echo "!!"
    echo "!! Also check the NVIDIA Jetson forums thread 'PyTorch for Jetson' for"
    echo "!! the latest links, since these change with each JetPack release."
    exit 1
}

echo ""
echo "== Installing the rest of the project's Python dependencies =="
pip install -r requirements.txt

echo ""
echo "== Verifying CUDA is visible to PyTorch =="
python3 -c "import torch; print('torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo ""
echo "Setup complete. Activate this environment in future sessions with:"
echo "    source .venv/bin/activate"
