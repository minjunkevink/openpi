<!-- 395941dc-ef82-44be-8bdd-90b6606bcf49 46d71a79-1c02-47df-b103-d23201c7beb8 -->
# Modular Language Augmentation Experiments

## Overview

Refactor `run_custom_libero_prompts.py` to support multiple experiment types from `AUGMENTED_EXPERIMENTS` dictionary, with modular execution, configurable rollout counts, and type-specific wandb visualizations.

## Key Changes

### 1. Replace HIERARCHICAL_EXPERIMENTS with AUGMENTED_EXPERIMENTS

- **File**: `run_custom_libero_prompts.py` (lines 54-79)
- Replace `HIERARCHICAL_EXPERIMENTS` dictionary with the new `AUGMENTED_EXPERIMENTS` structure
- Structure: `{task_idx: {experiment_type: {prompt_key: prompt_value}}}`

### 2. Update Args dataclass

- **File**: `run_custom_libero_prompts.py` (lines 82-123)
- Add `experiment_type: str` with options: `"hierarchical"`, `"task_completion_stop"`, `"kinematic"`, `"goal_state_synonym"`, `"logical_constraint_negation"`, `"undo"`, `"task_decomposition"`, or `"run_all"`
- Add `num_rollouts_per_prompt: int = 10` for list-based experiments
- Add `hierarchical_subset_rollouts: int = 1` for hierarchical subset rollouts
- Add `hierarchical_superset_rollouts: int = 10` for hierarchical superset rollouts
- Update `video_out_path` default to include experiment type in path

### 3. Create modular experiment runner functions

- **File**: `run_custom_libero_prompts.py`
- Create `run_hierarchical_experiment()` - refactor existing function to use `AUGMENTED_EXPERIMENTS[task_idx]["hierarchical"]`
- Create `run_list_based_experiment()` - generic function for list-based types (task_completion_stop, kinematic, goal_state_synonym, logical_constraint_negation, undo, task_decomposition)
- Create `run_all_experiments()` - orchestrates running all experiment types sequentially
- Update `main()` to route to appropriate function based on `experiment_type`

### 4. Implement list-based experiment runner

- **File**: `run_custom_libero_prompts.py`
- Function: `run_list_based_experiment(experiment_type: str, args: Args)`
- For each task in `AUGMENTED_EXPERIMENTS`:
- Extract prompt list from `AUGMENTED_EXPERIMENTS[task_idx][experiment_type]`
- Run `num_rollouts_per_prompt` rollouts per prompt
- Save all videos organized as: `{experiment_type}/task_{idx}/{prompt_idx}_{prompt_safe_name}/rollout_{n}.mp4`
- Track success rates per prompt
- Log all videos to wandb with appropriate grouping

### 5. Update hierarchical experiment runner

- **File**: `run_custom_libero_prompts.py` (lines 234-602)
- Modify to read from `AUGMENTED_EXPERIMENTS[task_idx]["hierarchical"]["subset"]` and `["superset"]`
- Use configurable rollout counts from args
- Keep existing side-by-side visualization logic

### 6. Create type-specific visualizations

- **File**: `run_custom_libero_prompts.py`
- **Hierarchical**: Keep existing 3-panel side-by-side comparison (subset, superset success, superset failure)
- **List-based types**: Create visualization showing:
- Success rate bar chart per prompt
- Table with all videos (one row per prompt, columns for sample success/failure videos)
- Summary statistics (total prompts, total rollouts, overall success rate)

### 7. Update wandb logging

- **File**: `run_custom_libero_prompts.py`
- Set `wandb_run_name` to include experiment type: `f"{experiment_type}-experiment-{timestamp}"` if not provided
- Group wandb logs by experiment type: `{experiment_type}/task_{idx}/...`
- For list-based: Log all videos under `{experiment_type}/task_{idx}/prompt_{n}/videos/`
- Log summary metrics: success rates, counts, etc.

### 8. Update data organization

- **File**: `run_custom_libero_prompts.py`
- Base path: `{args.video_out_path}/{experiment_type}/`
- Hierarchical: `{base}/task_{idx}_{task_name}/subset_{success|failure}.mp4`, `superset_{success|failure}.mp4`
- List-based: `{base}/task_{idx}_{task_name}/prompt_{n}_{prompt_safe_name}/rollout_{m}_{success|failure}.mp4`

### 9. Update shell scripts

- **File**: `run_hierarchical_experiment.sh`
- Update to use new experiment type argument: `--args.experiment-type hierarchical`
- **File**: `run_custom_libero_evaluation.sh`
- Keep for backward compatibility, but update to use new structure
- **New File**: `run_augmented_experiments.sh`
- Create script that accepts experiment type as argument and runs appropriate experiment

## Implementation Details

### Experiment Type Detection

- Check if `args.experiment_type == "run_all"` → call `run_all_experiments()`
- Check if `args.experiment_type == "hierarchical"` → call `run_hierarchical_experiment()`
- Otherwise → call `run_list_based_experiment(args.experiment_type)`

### Prompt Safety

- Create helper function `_make_prompt_safe(prompt: str) -> str` to sanitize prompts for file/directory names
- Limit length, replace special characters, etc.

### Error Handling

- Validate that `experiment_type` exists in `AUGMENTED_EXPERIMENTS` for selected tasks
- Handle missing experiment types gracefully (skip or warn)

## Implementation Status

- Starting implementation: replacing HIERARCHICAL_EXPERIMENTS with AUGMENTED_EXPERIMENTS

## Files to Modify

1. `run_custom_libero_prompts.py` - Main refactoring
2. `run_hierarchical_experiment.sh` - Update argument format
3. `run_custom_libero_evaluation.sh` - Update for compatibility
4. `run_augmented_experiments.sh` - New script for running any experiment type

## Testing Considerations

- Verify all experiment types can be run independently
- Verify "run_all" executes all types correctly
- Verify wandb logging groups correctly by experiment type
- Verify videos are saved in correct directory structure
- Verify configurable rollout counts work for all types