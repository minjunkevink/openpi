#!/bin/bash
# Script to run custom LIBERO evaluation with specific language prompts
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
echo "Running Custom LIBERO Evaluation"
echo "=========================================="
echo "Prompts:"
echo "  1. Turn on the stove and put the moka pot on it"
echo "  2. Put the black bowl in the bottom drawer of the cabinet and close it"
echo "  3. Put the yellow and white mug in the microwave and close it"
echo "  4. Put the white mug on the left plate and put the yellow and white mug on the right plate"
echo "  5. Pick up the book and place it in the back compartment of the caddy"
echo ""
echo "Rollouts per prompt: 10"
echo "Videos will be saved to: data/libero/custom_prompts"
echo "Wandb logging: ENABLED (use --args.use-wandb=false to disable)"
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

# Run the evaluation
# Add --args.use-wandb to enable wandb logging
uv run python run_custom_libero_prompts.py \
    --args.host localhost \
    --args.port 8000 \
    --args.task-suite-name libero_10 \
    --args.num-trials-per-prompt 10 \
    --args.video-out-path data/libero/custom_prompts \
    "${@}"  # Pass through any additional arguments

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Videos saved to: data/libero/custom_prompts"
echo "=========================================="

