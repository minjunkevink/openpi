#!/bin/bash
# Script to run any language augmentation experiment type
# Usage: ./run_augmented_experiments.sh [experiment_type]
#   experiment_type: hierarchical, task_completion_stop, kinematic, goal_state_synonym,
#                    logical_constraint_negation, undo, task_decomposition, or run_all
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

# Get experiment type from first argument or default to "run_all"
EXPERIMENT_TYPE=${1:-run_all}

# Validate experiment type
VALID_TYPES=("hierarchical" "task_completion_stop" "kinematic" "goal_state_synonym" 
             "logical_constraint_negation" "undo" "task_decomposition" "run_all" "custom")

if [[ ! " ${VALID_TYPES[@]} " =~ " ${EXPERIMENT_TYPE} " ]]; then
    echo "ERROR: Invalid experiment type: $EXPERIMENT_TYPE"
    echo "Valid types: ${VALID_TYPES[*]}"
    exit 1
fi

echo "=========================================="
echo "Running Language Augmentation Experiment"
echo "=========================================="
echo "Experiment type: $EXPERIMENT_TYPE"
echo ""
echo "Available experiment types:"
echo "  - hierarchical: Compare subset (original) vs superset (abstract) prompts"
echo "  - task_completion_stop: Test prompts with explicit stop commands"
echo "  - kinematic: Test prompts with speed/force modifiers"
echo "  - goal_state_synonym: Test different ways of describing goal states"
echo "  - logical_constraint_negation: Test prompts with negative constraints"
echo "  - undo: Test prompts for reversing actions"
echo "  - task_decomposition: Test prompts breaking tasks into steps"
echo "  - run_all: Run all experiment types sequentially"
echo ""
echo "Videos will be saved to: data/libero/custom_prompts/$EXPERIMENT_TYPE"
echo "Wandb logging: ENABLED (use --args.use-wandb=false to disable)"
echo ""
echo "Configurable options:"
echo "  --args.num-rollouts-per-prompt N        (default: 10, for list-based experiments)"
echo "  --args.hierarchical-subset-rollouts N   (default: 1, for hierarchical)"
echo "  --args.hierarchical-superset-rollouts N (default: 10, for hierarchical)"
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

# Shift to remove experiment_type from arguments before passing to Python script
# (only if an argument was provided)
if [[ -n "$1" ]]; then
    shift
fi

# Run the experiment
uv run python run_custom_libero_prompts.py \
    --args.experiment-type "$EXPERIMENT_TYPE" \
    --args.host localhost \
    --args.port 8000 \
    --args.task-suite-name libero_10 \
    --args.video-out-path data/libero/custom_prompts \
    "${@}"  # Pass through any additional arguments

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "Videos saved to: data/libero/custom_prompts/$EXPERIMENT_TYPE"
echo "Check wandb for visualizations and metrics"
echo "=========================================="

