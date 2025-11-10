#!/bin/bash
# Script to run π 0.5 -LIBERO inference on GPU 0
# Usage: ./run_inference_libero_gpu0.sh [port]

set -e

# Set GPU 0 as visible device
export CUDA_VISIBLE_DEVICES=0

# Add uv to PATH if not already there
export PATH="$HOME/.local/bin:$PATH"

# Set port (default: 8000)
PORT=${1:-8000}

echo "=========================================="
echo "Running π 0.5 -LIBERO Inference"
echo "=========================================="
echo "GPU Device: GPU 0 (CUDA_VISIBLE_DEVICES=0)"
echo "Checkpoint: gs://openpi-assets/checkpoints/pi05_libero"
echo "Port: $PORT"
echo "=========================================="
echo ""

# Run the policy server
uv run scripts/serve_policy.py \
    --port $PORT \
    policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_libero

