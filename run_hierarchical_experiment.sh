#!/bin/bash
# Script to run hierarchical language experiments
# Compares subset (original) vs superset (abstract) language prompts
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
echo "Running Hierarchical Language Experiment"
echo "=========================================="
echo "Experiment: Comparing subset (original) vs superset (abstract) language"
echo ""
echo "Tasks:"
echo "  1. Task 2: 'Turn on the stove...' vs 'Prepare the stove'"
echo "  2. Task 3: 'Put the black bowl...' vs 'Access the cabinet'"
echo "  3. Task 9: 'Put the yellow and white mug...' vs 'Arrange the tableware'"
echo "  4. Task 4: 'Put the white mug...' vs 'Organize the dishes'"
echo "  5. Task 5: 'Pick up the book...' vs 'Store the wine bottle'"
echo ""
echo "For each task:"
echo "  - Subset (original): 1 rollout"
echo "  - Superset (abstract): 10 rollouts"
echo ""
echo "Videos will be saved to: data/libero/hierarchical_experiments"
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

# Run the hierarchical experiment
uv run python run_custom_libero_prompts.py \
    --args.experiment-type hierarchical \
    --args.host localhost \
    --args.port 8000 \
    --args.task-suite-name libero_10 \
    --args.video-out-path data/libero/hierarchical_experiments \
    "${@}"  # Pass through any additional arguments

echo ""
echo "=========================================="
echo "Hierarchical experiment complete!"
echo "Videos saved to: data/libero/hierarchical_experiments"
echo "Check wandb for side-by-side video comparisons and metrics"
echo "=========================================="

