#!/usr/bin/env python3
"""
Run π 0.5 -LIBERO inference on Task 4 with custom language prompts.

Task 4: "Put the white mug on the left plate and put the yellow and white mug on the right plate"

This script runs Task 4 with custom language prompts defined below.
Videos are logged to wandb but NOT saved to disk.
"""

import collections
import dataclasses
import logging
import math
import pathlib
import sys
import tempfile

import imageio
import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

# Task 4 index in libero_10 suite
TASK_4_IDX = 4

# ============================================================================
# CUSTOM LANGUAGE PROMPTS
# ============================================================================
# Define your custom language prompts here.
# These will be used instead of the original task language.
CUSTOM_PROMPTS = [
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "put the white mug on the right plate and put the yellow and white mug on the left plate",
    "put the put the yellow and white mug on the right plate and put the white mug on the left plate",
    "put the yellow and white mug on the left plate and put the white mug on the right plate",
]


@dataclasses.dataclass
class Args:
    """Arguments for Task 4 custom language evaluation."""

    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "localhost"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_rollouts_per_prompt: int = 1  # Number of rollouts per prompt

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7  # Random Seed (for reproducibility)
    max_steps: int = 500  # Maximum steps per rollout

    #################################################################################################################
    # Wandb logging
    #################################################################################################################
    use_wandb: bool = True  # Whether to log to wandb
    wandb_project: str = "openpi-libero-task4-custom-language"  # Wandb project name
    wandb_run_name: str | None = None  # Wandb run name (auto-generated if None)


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment."""
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env


def run_single_rollout(
    env,
    initial_state,
    custom_prompt: str,
    client: _websocket_client_policy.WebsocketClientPolicy,
    args: Args,
    max_steps: int,
) -> tuple[bool, list]:
    """Run a single rollout with the given prompt and return success status and images."""
    # Reset environment
    env.reset()
    action_plan = collections.deque()

    # Set initial states
    obs = env.set_init_state(initial_state)

    # Setup
    t = 0
    replay_images = []

    while t < max_steps + args.num_steps_wait:
        try:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            # Get preprocessed image
            # IMPORTANT: rotate 180 degrees to match train preprocessing
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
            )
            wrist_img = image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            wrist_img = image_tools.convert_to_uint8(wrist_img)

            # Save preprocessed image for replay video
            replay_images.append(img)

            if not action_plan:
                # Finished executing previous action chunk -- compute new chunk
                # Prepare observations dict
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": custom_prompt,  # Use custom prompt instead of task.language
                }

                # Query model to get action
                action_chunk = client.infer(element)["actions"]
                assert (
                    len(action_chunk) >= args.replan_steps
                ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                return True, replay_images
            t += 1

        except Exception as e:
            logging.error(f"Caught exception: {e}")
            break

    return False, replay_images


def main(args: Args) -> None:
    """Main function to run Task 4 with custom language prompts."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Validate custom prompts
    if not CUSTOM_PROMPTS:
        logging.error("CUSTOM_PROMPTS list is empty! Please add your custom language prompts to the list.")
        sys.exit(1)

    # Initialize wandb if requested
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"task4_custom_language_{wandb.util.generate_id()}",
                config={
                    "task_idx": TASK_4_IDX,
                    "task_name": "Put the white mug on the left plate and put the yellow and white mug on the right plate",
                    "num_prompts": len(CUSTOM_PROMPTS),
                    "num_rollouts_per_prompt": args.num_rollouts_per_prompt,
                    "seed": args.seed,
                    "max_steps": args.max_steps,
                },
            )
        except ImportError:
            logging.warning("wandb not installed. Install with: pip install wandb")
            args.use_wandb = False
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {e}")
            args.use_wandb = False

    # Load LIBERO task suite
    logging.info("Loading LIBERO task suite...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    num_tasks_in_suite = task_suite.n_tasks

    if TASK_4_IDX >= num_tasks_in_suite:
        logging.error(f"Task index {TASK_4_IDX} is out of range. Suite has {num_tasks_in_suite} tasks.")
        sys.exit(1)

    # Get Task 4
    task = task_suite.get_task(TASK_4_IDX)
    logging.info(f"Task {TASK_4_IDX}: {task.language}")

    # Get default LIBERO initial states
    initial_states = task_suite.get_task_init_states(TASK_4_IDX)

    # Initialize LIBERO environment
    env = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    # Connect to policy server
    logging.info(f"Connecting to policy server at {args.host}:{args.port}...")
    client = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    logging.info(f"\n{'='*60}")
    logging.info(f"Running Task 4 with {len(CUSTOM_PROMPTS)} custom language prompts")
    logging.info(f"Number of rollouts per prompt: {args.num_rollouts_per_prompt}")
    logging.info(f"{'='*60}\n")

    # Track results
    all_results = []

    # Run rollouts for each custom prompt
    for prompt_idx, custom_prompt in enumerate(tqdm.tqdm(CUSTOM_PROMPTS, desc="Processing prompts")):
        logging.info(f"\n{'='*60}")
        logging.info(f"Prompt {prompt_idx + 1}/{len(CUSTOM_PROMPTS)}: {custom_prompt}")
        logging.info(f"{'='*60}")

        prompt_successes = 0
        prompt_results = []

        # Run multiple rollouts for this prompt
        for rollout_idx in tqdm.tqdm(
            range(args.num_rollouts_per_prompt),
            desc=f"  Rollouts for prompt {prompt_idx+1}",
            leave=False,
        ):
            # Use first initial state (or cycle through if multiple)
            initial_state = initial_states[rollout_idx % len(initial_states)]

            # Run rollout
            success, replay_images = run_single_rollout(
                env, initial_state, custom_prompt, client, args, args.max_steps
            )

            if success:
                prompt_successes += 1

            # Log video to wandb (save temporarily, log, then delete)
            if args.use_wandb and wandb_run is not None:
                try:
                    import wandb
                    # Create temporary file for video
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                        tmp_video_path = tmp_file.name
                    
                    # Save video to temporary file
                    imageio.mimwrite(
                        tmp_video_path,
                        [np.asarray(x) for x in replay_images],
                        fps=10,
                    )
                    
                    # Log video from file path (no moviepy needed)
                    wandb_run.log({
                        f"videos/prompt_{prompt_idx+1}/rollout_{rollout_idx+1}": wandb.Video(
                            tmp_video_path,
                            format="mp4",
                            caption=custom_prompt
                        ),
                    })
                    
                    # Delete temporary file
                    pathlib.Path(tmp_video_path).unlink(missing_ok=True)
                except Exception as e:
                    logging.warning(f"Failed to log to wandb: {e}")
                    # Clean up temp file if it exists
                    if 'tmp_video_path' in locals():
                        pathlib.Path(tmp_video_path).unlink(missing_ok=True)

            prompt_results.append({
                "rollout_idx": rollout_idx,
                "success": success,
                "video_images": replay_images,  # Keep for potential later use
            })

        # Calculate success rate for this prompt
        success_rate = prompt_successes / args.num_rollouts_per_prompt

        logging.info(f"\nPrompt '{custom_prompt}' results:")
        logging.info(f"  Successes: {prompt_successes}/{args.num_rollouts_per_prompt} ({success_rate*100:.1f}%)")

        all_results.append({
            "prompt": custom_prompt,
            "prompt_idx": prompt_idx,
            "success_rate": success_rate,
            "successes": prompt_successes,
            "total_rollouts": args.num_rollouts_per_prompt,
            "results": prompt_results,
        })

    # Final summary
    logging.info(f"\n{'='*60}")
    logging.info("FINAL SUMMARY")
    logging.info(f"{'='*60}")
    for result in all_results:
        logging.info(
            f"Prompt {result['prompt_idx']+1}: {result['success_rate']*100:.1f}% "
            f"({result['successes']}/{result['total_rollouts']}) - {result['prompt']}"
        )

    overall_success_rate = sum(r["successes"] for r in all_results) / sum(r["total_rollouts"] for r in all_results)
    logging.info(f"\nOverall success rate: {overall_success_rate*100:.1f}%")

    # Cleanup
    env.close()
    if wandb_run is not None:
        wandb_run.finish()

    logging.info("\nDone!")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)

