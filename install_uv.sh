#!/bin/bash
set -e

echo "=== Installing kmol with uv ==="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment with specific Python version
echo "Creating Python 3.9.17 environment..."
uv venv --python 3.9.17

# Activate the environment
source .venv/bin/activate

# Install PyTorch with CUDA 11.7 first
echo "Installing PyTorch with CUDA 11.7..."
uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --index-url https://download.pytorch.org/whl/cu117

# Install build dependencies for PyG extensions
echo "Installing build dependencies..."
uv pip install setuptools wheel ninja

# Install PyTorch Geometric dependencies
echo "Installing PyTorch Geometric extensions..."
uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html \
    --no-build-isolation

# Install other dependencies
echo "Installing project dependencies..."
uv pip install -e .

echo ""
echo "=== Installation complete! ==="
echo ""
echo "Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then you can run kmol commands directly:"
echo "  python -m kmol.run train data/configs/model/tox21.json"
