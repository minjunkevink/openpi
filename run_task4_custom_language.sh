#!/bin/bash
# Script to run Task 4 with custom language prompts
# Make sure the policy server is running first: ./run_inference_libero_gpu0.sh

set -e

# Add uv to PATH if not already there
export PATH="$HOME/.local/bin:$PATH"

# Set up LIBERO environment if needed
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Check if libero is installed, if not run setup
if ! uv run python -c "import libero" 2>/dev/null; then
    echo "LIBERO not found. Running setup..."
    ./setup_libero_eval.sh
fi

echo "=========================================="
echo "Running Task 4 with Custom Language Prompts"
echo "=========================================="
echo "Task: Put the white mug on the left plate and put the yellow and white mug on the right plate"
echo ""
echo "Custom prompts are defined in run_task4_custom_language.py"
echo "Videos will be logged to wandb (NOT saved to disk)"
echo ""
echo "Configurable options:"
echo "  --num-rollouts-per-prompt N    (default: 10)"
echo "  --use-wandb false              (to disable wandb)"
echo "  --wandb-project PROJECT_NAME  (default: openpi-libero-task4-custom-language)"
echo "=========================================="
echo ""

# Check if policy server is running
if ! nc -z localhost 8000 2>/dev/null; then
    echo "WARNING: Policy server doesn't seem to be running on port 8000"
    echo "Make sure to start it first: ./run_inference_libero_gpu0.sh"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run the script
uv run python run_task4_custom_language.py \
    --host localhost \
    --port 8000 \
    "${@}"  # Pass through any additional arguments

echo ""
echo "=========================================="
echo "Task 4 evaluation complete!"
echo "Check wandb for videos with language prompts"
echo "=========================================="

