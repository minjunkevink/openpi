# Custom LIBERO Evaluation with Language Prompts

This guide explains how to run π 0.5 -LIBERO inference with custom language prompts on libero_10 tasks.

## Overview

This evaluation runs 5 custom language prompts (from libero_10 temporal/sequential tasks) with 10 rollouts each, generating videos for every rollout.

**Prompts:**
1. "Turn on the stove and put the moka pot on it"
2. "Put the black bowl in the bottom drawer of the cabinet and close it"
3. "Put the yellow and white mug in the microwave and close it"
4. "Put the white mug on the left plate and put the yellow and white mug on the right plate"
5. "Pick up the book and place it in the back compartment of the caddy"

## Prerequisites

1. **Policy server must be running** (in Terminal 1):
   ```bash
   ./run_inference_libero_gpu0.sh
   ```

2. **LIBERO dependencies** (will be installed automatically on first run)

## Quick Start

### Step 1: Setup (First time only)

```bash
./setup_libero_eval.sh
```

This will:
- Initialize LIBERO submodule if needed
- Install LIBERO package
- Install required dependencies

### Step 2: Run Evaluation

```bash
./run_custom_libero_evaluation.sh
```

This will:
- Run 10 rollouts for each of the 5 prompts (50 total rollouts)
- Save videos to `data/libero/custom_prompts/`
- Show progress and success rates

## Manual Run

If you prefer to run manually:

```bash
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

uv run python run_custom_libero_prompts.py \
    --host localhost \
    --port 8000 \
    --task-suite-name libero_10 \
    --num-trials-per-prompt 10 \
    --video-out-path data/libero/custom_prompts
```

## Customization

### Change number of rollouts per prompt:

```bash
uv run python run_custom_libero_prompts.py \
    --num-trials-per-prompt 20 \
    --video-out-path data/libero/custom_prompts
```

### Use different prompts:

Edit `run_custom_libero_prompts.py` and modify the `CUSTOM_PROMPTS` list, or pass them via command line:

```bash
uv run python run_custom_libero_prompts.py \
    --custom-prompts "Prompt 1" "Prompt 2" "Prompt 3" \
    --num-trials-per-prompt 10
```

### Change video output location:

```bash
uv run python run_custom_libero_prompts.py \
    --video-out-path /path/to/your/videos
```

## Output Structure

Videos are saved in the following structure:

```
data/libero/custom_prompts/
├── turn_on_the_stove_and_put_the_moka_pot_on_it/
│   ├── prompt_01_rollout_01_success.mp4
│   ├── prompt_01_rollout_01_failure.mp4
│   ├── prompt_01_rollout_02_success.mp4
│   └── ...
├── put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it/
│   └── ...
└── ...
```

Each video filename includes:
- `prompt_XX`: The prompt number (01-05)
- `rollout_XX`: The rollout number (01-10)
- `success` or `failure`: Whether the task succeeded

## Troubleshooting

### "ModuleNotFoundError: No module named 'libero'"

Run the setup script:
```bash
./setup_libero_eval.sh
```

### "Connection refused" or "Cannot connect to server"

Make sure the policy server is running:
```bash
./run_inference_libero_gpu0.sh
```

### "Mujoco EGL errors"

Set the environment variable before running:
```bash
export MUJOCO_GL=glx
./run_custom_libero_evaluation.sh
```

### Videos not saving

Check that the output directory is writable:
```bash
mkdir -p data/libero/custom_prompts
chmod -R 755 data/libero/custom_prompts
```

## Script Details

- **`run_custom_libero_prompts.py`**: Main evaluation script
- **`run_custom_libero_evaluation.sh`**: Convenience wrapper script
- **`setup_libero_eval.sh`**: Setup script for LIBERO dependencies

## Notes

- Each rollout uses the same initial state from the LIBERO task suite
- Videos are saved at 10 FPS
- The script uses the first 5 tasks from libero_10 and overrides their language prompts
- Total runtime depends on task complexity (typically 5-15 minutes per rollout)

