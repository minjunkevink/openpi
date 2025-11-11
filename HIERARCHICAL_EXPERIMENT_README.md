# Hierarchical Language Experiment

This experiment compares how the π 0.5 -LIBERO model performs with original (subset) language prompts versus abstract (superset) language prompts on the same tasks.

## Experiment Design

For each of 5 tasks from libero_10:
- **Subset (Original)**: 1 rollout using the original LIBERO task language (specific, detailed)
- **Superset (Abstract)**: 10 rollouts using abstract/general language (hierarchical, high-level)

The same initial state (index 0) is used for the subset rollout to ensure fair comparison.

## Tasks Tested

1. **Task 2**: "Turn on the stove and put the moka pot on it" vs "Prepare the stove"
2. **Task 3**: "Put the black bowl in the bottom drawer of the cabinet and close it" vs "Access the cabinet"
3. **Task 9**: "Put the yellow and white mug in the microwave and close it" vs "Arrange the tableware"
4. **Task 4**: "Put the white mug on the left plate and put the yellow and white mug on the right plate" vs "Organize the dishes"
5. **Task 5**: "Pick up the book and place it in the back compartment of the caddy" vs "Store the wine bottle"

## Running the Experiment

### Prerequisites

1. **Policy server must be running** (Terminal 1):
   ```bash
   ./run_inference_libero_gpu0.sh
   ```

2. **LIBERO dependencies installed** (if not already):
   ```bash
   ./setup_libero_eval.sh
   ```

### Quick Start

```bash
./run_hierarchical_experiment.sh
```

### Manual Run

```bash
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

uv run python run_custom_libero_prompts.py \
    --args.experiment-type hierarchical \
    --args.host localhost \
    --args.port 8000 \
    --args.task-suite-name libero_10 \
    --args.video-out-path data/libero/hierarchical_experiments
```

## Output

### Local Videos

Videos are saved to `data/libero/hierarchical_experiments/` organized by task:

```
data/libero/hierarchical_experiments/
├── task_2_turn_on_the_stove.../
│   ├── subset_success.mp4 (or subset_failure.mp4)
│   ├── superset_rollout_01_success.mp4
│   ├── superset_rollout_01_failure.mp4
│   ├── superset_rollout_02_success.mp4
│   └── ... (10 superset rollouts)
├── task_3_put_the_black_bowl.../
└── ...
```

### Wandb Visualizations

The experiment automatically logs to wandb with:

1. **Comparison Table** (`hierarchical/comparison_table`):
   - Side-by-side videos: Subset video | Successful Superset video
   - Task information and prompts
   - Success metrics for each task

2. **Bar Chart** (`hierarchical/success_rate_comparison`):
   - Compares subset vs superset success rates across all tasks
   - Shows percentage labels on bars

3. **Individual Metrics**:
   - `hierarchical/task_{N}/subset_success`: 0 or 1
   - `hierarchical/task_{N}/superset_success_rate`: X/10
   - `hierarchical/task_{N}/superset_success_count`: number of successes
   - All individual videos logged separately

4. **Summary Metrics**:
   - Overall subset success rate
   - Average superset success rate
   - Total tasks completed

## Interpreting Results

- **Subset success**: Whether the model succeeded with the original, specific language (1 rollout)
- **Superset success rate**: Fraction of successful rollouts with abstract language (out of 10)
- **Comparison**: Shows if abstract language helps or hurts performance

The side-by-side video comparison in wandb allows you to visually compare:
- How the model behaves with specific vs abstract instructions
- Whether abstract language leads to similar or different behaviors
- Success patterns across different language formulations

## Customization

### Change video output location:
```bash
--args.video-out-path /path/to/your/videos
```

### Disable wandb:
```bash
--args.use-wandb=false
```

### Custom wandb project:
```bash
--args.wandb-project my-project-name
```

## Notes

- Total rollouts: 5 tasks × (1 subset + 10 superset) = 55 rollouts
- Each rollout uses the same task/scene but different language prompts
- Videos are automatically organized by task for easy comparison
- Wandb table shows successful superset videos side-by-side with subset videos

