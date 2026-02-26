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

# Install PyTorch Geometric dependencies with exact versions (pre-built wheels)
echo "Installing PyTorch Geometric extensions..."
uv pip install \
    torch-scatter==2.1.1 \
    torch-sparse==0.6.16 \
    torch-cluster==1.6.0 \
    torch-spline-conv==1.2.2 \
    torch-geometric==2.3.0 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

# Install numpy and cython for packages that need them at build time
echo "Installing build dependencies..."
uv pip install "numpy==1.24.4" "cython==0.29.33"

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
