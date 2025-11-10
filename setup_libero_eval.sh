#!/bin/bash
# Setup script for LIBERO evaluation
# This installs the necessary dependencies for running LIBERO evaluation

set -e

cd /home/kimkj/openpi
export PATH="$HOME/.local/bin:$PATH"

echo "=========================================="
echo "Setting up LIBERO Evaluation Environment"
echo "=========================================="

# Check if libero submodule is initialized
if [ ! -d "third_party/libero" ] || [ -z "$(ls -A third_party/libero 2>/dev/null)" ]; then
    echo "Initializing LIBERO submodule..."
    git submodule update --init --recursive third_party/libero
fi

# Install libero package
echo "Installing LIBERO package..."
uv pip install -e third_party/libero

# Install robosuite and other LIBERO dependencies
echo "Installing robosuite and LIBERO dependencies..."
# Install robosuite (key dependency for LIBERO)
uv pip install robosuite==1.4.1 --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match

# Install other key dependencies (use compatible versions, don't force old numpy)
uv pip install imageio matplotlib pyyaml tqdm bddl easydict gym || true

# Install openpi-client if not already installed
echo "Installing openpi-client..."
uv pip install -e packages/openpi-client

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "You can now run the evaluation with:"
echo "  ./run_custom_libero_evaluation.sh"
echo ""
echo "Make sure the policy server is running first:"
echo "  ./run_inference_libero_gpu0.sh"

