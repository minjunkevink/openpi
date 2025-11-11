#!/usr/bin/env python3
"""
Run π 0.5 -LIBERO inference with custom language prompts on libero_10 tasks.

This script runs specific tasks from libero_10 with custom language prompts,
generating 10 rollouts per prompt and saving videos for each.
"""

import collections
import dataclasses
import logging
import math
import pathlib
import sys

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import matplotlib.pyplot as plt
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

# Custom language prompts for libero_10 tasks (Temporal/Sequential)
# These correspond to specific tasks in libero_10:
# Prompt 1 -> Task 2: "turn on the stove and put the moka pot on it"
# Prompt 2 -> Task 3: "put the black bowl in the bottom drawer of the cabinet and close it"
# Prompt 3 -> Task 9: "put the yellow and white mug in the microwave and close it"
# Prompt 4 -> Task 4: "put the white mug on the left plate and put the yellow and white mug on the right plate"
# Prompt 5 -> Task 5: "pick up the book and place it in the back compartment of the caddy"
CUSTOM_PROMPTS = [
    "Turn on the stove and put the moka pot on it",
    "Put the black bowl in the bottom drawer of the cabinet and close it",
    "Put the yellow and white mug in the microwave and close it",
    "Put the white mug on the left plate and put the yellow and white mug on the right plate",
    "Pick up the book and place it in the back compartment of the caddy",
]

# Map custom prompts to their corresponding libero_10 task indices
CUSTOM_PROMPT_TO_TASK_IDX = {
    0: 2,  # "Turn on the stove..." -> Task 2
    1: 3,  # "Put the black bowl..." -> Task 3
    2: 9,  # "Put the yellow and white mug..." -> Task 9
    3: 4,  # "Put the white mug..." -> Task 4
    4: 5,  # "Pick up the book..." -> Task 5
}

# Augmented experiment configuration
# Maps task indices to experiment types, each containing different language augmentation prompts
AUGMENTED_EXPERIMENTS = {
    2: {  # Task: Turn on stove + place moka pot
        "hierarchical": {
            "subset": "Turn on the stove and put the moka pot on it",
            "superset": "Prepare the stove",
        },
        "task_completion_stop": [
            "Turn on the stove and place the moka pot on it, then stop",
            "Turn on the stove and stop",
            "Place the moka pot on the stove and stop",
            "Put the moka pot on the front-left burner and stop",
            "Do nothing",
            "Turn on the stove",  # no explicit stop—should it stop naturally?
        ],
        "kinematic": [
            "Slowly turn on the stove and gently place the moka pot on it",
            "Quickly turn on the stove, then carefully set the moka pot on the burner",
            "Gently place the moka pot on the stove after you turn it on",
            "Carefully align the moka pot before setting it down on the burner",
        ],
        "goal_state_synonym": [
            "Achieve a state where the stove is on and the moka pot is on a burner",
            "Ensure the stove is powered and the moka pot rests on the cooktop",
            "Make the moka pot be on the stove with the stove turned on",
            "Execute turning the stove on and staging the moka pot on top",
        ],
        "logical_constraint_negation": [
            "Place the moka pot on the stove but not on the back burners",
            "Turn on the stove but do not touch the kettle",
            "Put the moka pot on any burner except the rightmost one",
            "Turn on only one burner and avoid moving other cookware",
        ],
        "undo": [
            "Turn off the stove and remove the moka pot from the burner",
            "Lift the moka pot off the stove and power the stove off",
            "Undo the placement by taking the moka pot off the stove",
        ],
        "task_decomposition": [
            "Reach toward the stove controls",
            "Rotate the control to turn the stove on",
            "Reach for the moka pot",
            "Grasp the moka pot",
            "Place the moka pot on the selected burner",
            "Release the moka pot",
        ],
    },
    3: {  # Task: Black bowl -> bottom drawer + close
        "hierarchical": {
            "subset": "Put the black bowl in the bottom drawer of the cabinet and close it",
            "superset": "Access the cabinet",
        },
        "task_completion_stop": [
            "Open the bottom drawer, place the black bowl inside, close it, then stop",
            "Place the black bowl in the bottom drawer and stop",
            "Open the bottom drawer and stop",
            "Close the bottom drawer and stop",
            "Do nothing",
            "Put the black bowl in the bottom drawer",  # no explicit stop
        ],
        "kinematic": [
            "Slowly open the bottom drawer and gently place the black bowl inside",
            "Carefully slide the drawer out and softly close it after placing the bowl",
            "Quickly open the drawer but gently set the black bowl in",
            "Gently close the drawer without slamming it",
        ],
        "goal_state_synonym": [
            "Ensure the black bowl ends up inside the bottom drawer with the drawer closed",
            "Achieve the drawer-closed state with the black bowl contained inside",
            "Make the black bowl be in the bottom drawer and the drawer be shut",
            "Execute placing the bowl in the bottom drawer, then finish with it closed",
        ],
        "logical_constraint_negation": [
            "Place the black bowl in the bottom drawer, not the middle or top drawers",
            "Use the bottom drawer but do not touch the white bowl",
            "Put the black bowl inside and avoid leaving the drawer open",
            "Avoid placing any other item in the drawer besides the black bowl",
        ],
        "undo": [
            "Open the bottom drawer and remove the black bowl",
            "Undo the close by reopening the drawer",
            "Take the black bowl out of the bottom drawer and leave it outside",
        ],
        "task_decomposition": [
            "Reach for the bottom drawer handle",
            "Pull to open the bottom drawer",
            "Reach for the black bowl",
            "Grasp the black bowl",
            "Place the black bowl inside the bottom drawer",
            "Release the black bowl",
            "Push to close the bottom drawer",
        ],
    },
    9: {  # Task: Yellow-and-white mug -> microwave + close
        "hierarchical": {
            "subset": "Put the yellow and white mug in the microwave and close it",
            "superset": "Microwave the cup",
        },
        "task_completion_stop": [
            "Open the microwave, place the yellow and white mug inside, close it, then stop",
            "Put the yellow and white mug in the microwave and stop",
            "Close the microwave and stop",
            "Open the microwave and stop",
            "Do nothing",
            "Place the mug in the microwave",  # no explicit stop
        ],
        "kinematic": [
            "Slowly open the microwave and gently set the yellow and white mug inside",
            "Carefully center the mug on the turntable and close the door softly",
            "Quickly open the door but gently place the mug before closing",
            "Gently close the microwave door without slamming",
        ],
        "goal_state_synonym": [
            "Ensure the yellow and white mug is inside the microwave with the door closed",
            "Achieve the state where the mug is in the microwave and the door is shut",
            "Make the mug be inside and the microwave door be closed",
            "Execute inserting the mug and finishing with a closed microwave",
        ],
        "logical_constraint_negation": [
            "Place the yellow and white mug in the microwave, not the oven",
            "Put the mug inside while avoiding the white mug",
            "Insert the mug but do not leave the microwave door open",
            "Use the microwave and avoid moving any plates on the counter",
        ],
        "undo": [
            "Open the microwave and remove the yellow and white mug",
            "Undo the close by opening the microwave door",
            "Take the mug back out of the microwave and place it outside",
        ],
        "task_decomposition": [
            "Reach for the microwave handle",
            "Pull to open the microwave door",
            "Reach for the yellow and white mug",
            "Grasp the mug",
            "Place the mug on the microwave turntable",
            "Release the mug",
            "Push to close the microwave door",
        ],
    },
    4: {  # Task: White mug -> left plate; yellow&white mug -> right plate
        "hierarchical": {
            "subset": "Put the white mug on the left plate and put the yellow and white mug on the right plate",
            "superset": "Organize the tableware",
        },
        "task_completion_stop": [
            "Place the white mug on the left plate and the yellow and white mug on the right plate, then stop",
            "Put the white mug on the left plate and stop",
            "Put the yellow and white mug on the right plate and stop",
            "Swap any misplaced mugs to the correct plates and stop",
            "Do nothing",
            "Place the white mug on the left plate",  # no explicit stop
        ],
        "kinematic": [
            "Gently place the white mug on the left plate and carefully set the yellow and white mug on the right plate",
            "Slowly align each mug with its plate before releasing",
            "Carefully lift and softly set each mug on its designated plate",
            "Quickly transport but gently place each mug onto its assigned plate",
        ],
        "goal_state_synonym": [
            "Ensure the white mug ends up on the left plate and the yellow and white mug on the right plate",
            "Achieve the state where each mug rests on its correct plate",
            "Make the white mug be on the left plate and the yellow-and-white mug be on the right plate",
            "Execute positioning both mugs onto their designated plates",
        ],
        "logical_constraint_negation": [
            "Place the white mug on the left plate, not the right",
            "Place the yellow and white mug on the right plate, not the left",
            "Move only the two mugs and avoid relocating the plates",
            "Do not stack mugs; set each on its correct plate",
        ],
        "undo": [
            "Remove both mugs from the plates and place them back on the table",
            "Take the white mug off the left plate and the yellow and white mug off the right plate",
            "Undo the arrangement by clearing both plates",
        ],
        "task_decomposition": [
            "Reach for the white mug",
            "Grasp the white mug",
            "Place the white mug on the left plate",
            "Release the white mug",
            "Reach for the yellow and white mug",
            "Grasp the yellow and white mug",
            "Place the yellow and white mug on the right plate",
            "Release the yellow and white mug",
        ],
    },
    5: {  # Task: Book -> back compartment of caddy
        "hierarchical": {
            "subset": "Pick up the book and place it in the back compartment of the caddy",
            "superset": "Organize the items in the caddy.",
        },
        "task_completion_stop": [
            "Pick up the book and place it in the back compartment of the caddy, then stop",
            "Pick up the book and stop",
            "Open space in the back compartment if needed, place the book, and stop",
            "Remove clutter from the caddy, place the book in back, then stop",
            "Do nothing",
            "Place the book in the back compartment",  # no explicit stop
        ],
        "kinematic": [
            "Slowly pick up the book and gently set it in the back compartment",
            "Carefully align the book flush against the back divider before releasing",
            "Quickly grasp the book but softly place it into the back compartment",
            "Gently lower the book to avoid bending pages",
        ],
        "goal_state_synonym": [
            "Ensure the book is located in the caddy's back compartment",
            "Achieve placement of the book in the rear slot of the caddy",
            "Make the book be inside the back compartment",
            "Execute picking up the book and depositing it in the back section",
        ],
        "logical_constraint_negation": [
            "Place the book in the back compartment, not the front",
            "Move only the book and do not relocate other caddy items",
            "Avoid placing the notebook; use the book",
            "Do not leave the book on the table—use the caddy's back compartment",
        ],
        "undo": [
            "Remove the book from the back compartment of the caddy",
            "Take the book out of the caddy and place it back on the table",
            "Undo the placement by retrieving the book from the back slot",
        ],
        "task_decomposition": [
            "Reach for the book",
            "Grasp the book",
            "Transport the book toward the caddy",
            "Insert the book into the back compartment",
            "Release the book",
            "Optionally adjust the book to sit upright",
        ],
    },
}


@dataclasses.dataclass
class Args:
    """Arguments for the custom LIBERO evaluation script."""

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
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_prompt: int = 10  # Number of rollouts per prompt

    #################################################################################################################
    # Custom prompts
    #################################################################################################################
    # If provided, will use these prompts instead of the default CUSTOM_PROMPTS
    custom_prompts: list[str] | None = None

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/custom_prompts"  # Path to save videos (will be extended with experiment_type)
    seed: int = 7  # Random Seed (for reproducibility)
    
    #################################################################################################################
    # Wandb logging
    #################################################################################################################
    use_wandb: bool = True  # Whether to log to wandb (default: True)
    wandb_project: str = "openpi-libero-custom-prompts"  # Wandb project name
    wandb_run_name: str | None = None  # Wandb run name (auto-generated if None)
    
    #################################################################################################################
    # Experiment type and rollout configuration
    #################################################################################################################
    experiment_type: str = "custom"  # "custom", "hierarchical", "task_completion_stop", "kinematic", "goal_state_synonym", "logical_constraint_negation", "undo", "task_decomposition", or "run_all"
    num_rollouts_per_prompt: int = 10  # Number of rollouts per prompt for list-based experiments
    hierarchical_subset_rollouts: int = 1  # Number of rollouts for hierarchical subset (original) prompts
    hierarchical_superset_rollouts: int = 10  # Number of rollouts for hierarchical superset (abstract) prompts


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


def _make_prompt_safe(prompt: str, max_length: int = 50) -> str:
    """Sanitize a prompt string for use in file/directory names."""
    # Replace special characters with underscores
    safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in prompt)
    # Replace spaces with underscores
    safe = safe.replace(" ", "_").lower()
    # Limit length
    safe = safe[:max_length]
    return safe


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
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )

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


def run_hierarchical_experiment(args: Args) -> None:
    """Run hierarchical language experiments comparing subset (original) vs superset (abstract) prompts."""
    import datetime
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Update video path to include experiment type
    experiment_type = "hierarchical"
    video_out_path = pathlib.Path(args.video_out_path) / experiment_type
    video_out_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if requested
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"{experiment_type}-experiment-{timestamp}",
                config={
                    "experiment_type": experiment_type,
                    "task_suite": args.task_suite_name,
                    "num_tasks": len([idx for idx in AUGMENTED_EXPERIMENTS.keys() if "hierarchical" in AUGMENTED_EXPERIMENTS[idx]]),
                    "subset_rollouts": args.hierarchical_subset_rollouts,
                    "superset_rollouts": args.hierarchical_superset_rollouts,
                    "seed": args.seed,
                    "host": args.host,
                    "port": args.port,
                }
            )
            logging.info(f"Initialized wandb run: {wandb_run.name}")
        except ImportError:
            logging.warning("wandb not installed. Install with: uv pip install wandb")
            args.use_wandb = False
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {e}")
            args.use_wandb = False

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    
    # Set max steps for libero_10
    max_steps = 520
    
    # Connect to policy server
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info(f"Connected to policy server at {args.host}:{args.port}")
    
    # Store results for each task
    task_results = {}
    
    # Process each task in hierarchical experiments
    task_indices = sorted([idx for idx in AUGMENTED_EXPERIMENTS.keys() if "hierarchical" in AUGMENTED_EXPERIMENTS[idx]])
    
    for task_idx in tqdm.tqdm(task_indices, desc="Processing hierarchical tasks"):
        if "hierarchical" not in AUGMENTED_EXPERIMENTS[task_idx]:
            logging.warning(f"Skipping task {task_idx}: no hierarchical experiment defined")
            continue
        
        subset_prompt = AUGMENTED_EXPERIMENTS[task_idx]["hierarchical"]["subset"]
        superset_prompt = AUGMENTED_EXPERIMENTS[task_idx]["hierarchical"]["superset"]
        task = task_suite.get_task(task_idx)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Task {task_idx}: {task.language}")
        logging.info(f"Subset (Original): {subset_prompt}")
        logging.info(f"Superset (Abstract): {superset_prompt}")
        logging.info(f"{'='*60}")
        
        # Get initial states
        initial_states = task_suite.get_task_init_states(task_idx)
        
        # Initialize environment
        env = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        
        # Create task-specific directory
        task_name_safe = _make_prompt_safe(task.language)
        task_video_dir = video_out_path / f"task_{task_idx}_{task_name_safe}"
        task_video_dir.mkdir(parents=True, exist_ok=True)
        
        # Run subset rollouts - but only save first success and first failure videos
        logging.info(f"\nRunning subset (original language): {args.hierarchical_subset_rollouts} rollout(s)")
        subset_successes = []
        successful_subset_video_path = None
        failed_subset_video_path = None
        successful_subset_found = False
        failed_subset_found = False
        
        for rollout_idx in range(args.hierarchical_subset_rollouts):
            initial_state = initial_states[rollout_idx % len(initial_states)]
            success, replay_images = run_single_rollout(
                env, initial_state, subset_prompt, client, args, max_steps
            )
            subset_successes.append(success)
            
            # Only save first success and first failure videos
            if success and not successful_subset_found:
                successful_subset_video_path = task_video_dir / f"subset_success.mp4"
                imageio.mimwrite(
                    successful_subset_video_path,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
                successful_subset_found = True
                logging.info(f"  Subset rollout {rollout_idx + 1}: ✓ SUCCESS (saved)")
            elif not success and not failed_subset_found:
                failed_subset_video_path = task_video_dir / f"subset_failure.mp4"
                imageio.mimwrite(
                    failed_subset_video_path,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
                failed_subset_found = True
                logging.info(f"  Subset rollout {rollout_idx + 1}: ✗ FAILURE (saved)")
            else:
                logging.info(f"  Subset rollout {rollout_idx + 1}: {'✓ SUCCESS' if success else '✗ FAILURE'} (not saved)")
        
        subset_success_count = sum(subset_successes)
        subset_success_rate = float(subset_success_count) / args.hierarchical_subset_rollouts if args.hierarchical_subset_rollouts > 0 else 0.0
        subset_video_path = successful_subset_video_path if successful_subset_video_path else failed_subset_video_path
        
        # Run superset rollouts - but only save first success and first failure videos
        logging.info(f"\nRunning superset (abstract language): {args.hierarchical_superset_rollouts} rollouts")
        superset_successes = []
        successful_superset_video_path = None
        failed_superset_video_path = None
        successful_superset_found = False
        failed_superset_found = False
        
        for rollout_idx in tqdm.tqdm(range(args.hierarchical_superset_rollouts), desc=f"Superset rollouts for task {task_idx}"):
            # Use initial states 0-9 (cycling if needed)
            initial_state = initial_states[rollout_idx % len(initial_states)]
            
            success, replay_images = run_single_rollout(
                env, initial_state, superset_prompt, client, args, max_steps
            )
            superset_successes.append(success)
            
            # Only save first success and first failure videos
            if success and not successful_superset_found:
                successful_superset_video_path = task_video_dir / f"superset_success.mp4"
                imageio.mimwrite(
                    successful_superset_video_path,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
                successful_superset_found = True
                logging.info(f"  Rollout {rollout_idx + 1}: ✓ SUCCESS (saved)")
            elif not success and not failed_superset_found:
                failed_superset_video_path = task_video_dir / f"superset_failure.mp4"
                imageio.mimwrite(
                    failed_superset_video_path,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
                failed_superset_found = True
                logging.info(f"  Rollout {rollout_idx + 1}: ✗ FAILURE (saved)")
            else:
                # Count but don't save video
                logging.info(f"  Rollout {rollout_idx + 1}: {'✓ SUCCESS' if success else '✗ FAILURE'} (not saved)")
        
        # Calculate metrics
        superset_success_count = sum(superset_successes)
        superset_success_rate = float(superset_success_count) / args.hierarchical_superset_rollouts if args.hierarchical_superset_rollouts > 0 else 0.0
        
        # Store results
        task_results[task_idx] = {
            "task_name": task.language,
            "subset_prompt": subset_prompt,
            "superset_prompt": superset_prompt,
            "subset_success_count": subset_success_count,
            "subset_success_rate": subset_success_rate,
            "subset_video": subset_video_path,
            "successful_subset_video": successful_subset_video_path,
            "failed_subset_video": failed_subset_video_path,
            "superset_success_count": superset_success_count,
            "superset_success_rate": superset_success_rate,
            "successful_superset_video": successful_superset_video_path,
            "failed_superset_video": failed_superset_video_path,
        }
        
        # Log metrics to wandb
        if args.use_wandb and wandb_run is not None:
            try:
                import wandb
                wandb_run.log({
                    f"{experiment_type}/task_{task_idx}/subset_success_rate": subset_success_rate,
                    f"{experiment_type}/task_{task_idx}/subset_success_count": subset_success_count,
                    f"{experiment_type}/task_{task_idx}/superset_success_rate": superset_success_rate,
                    f"{experiment_type}/task_{task_idx}/superset_success_count": superset_success_count,
                    f"{experiment_type}/task_{task_idx}/subset_prompt": subset_prompt,
                    f"{experiment_type}/task_{task_idx}/superset_prompt": superset_prompt,
                })
                
                # Log individual videos (for reference, but main visualization is in comparison plots)
                if successful_subset_video_path:
                    wandb_run.log({
                        f"{experiment_type}/task_{task_idx}/subset_success_video": wandb.Video(
                            str(successful_subset_video_path),
                            format="mp4",
                            caption=f"Subset Success: {subset_prompt}"
                        ),
                    })
                
                if failed_subset_video_path:
                    wandb_run.log({
                        f"{experiment_type}/task_{task_idx}/subset_failure_video": wandb.Video(
                            str(failed_subset_video_path),
                            format="mp4",
                            caption=f"Subset Failure: {subset_prompt}"
                        ),
                    })
                
                # Log saved superset videos only
                if successful_superset_video_path:
                    wandb_run.log({
                        f"{experiment_type}/task_{task_idx}/superset_success_video": wandb.Video(
                            str(successful_superset_video_path),
                            format="mp4",
                            caption=f"Superset Success: {superset_prompt}"
                        ),
                    })
                
                if failed_superset_video_path:
                    wandb_run.log({
                        f"{experiment_type}/task_{task_idx}/superset_failure_video": wandb.Video(
                            str(failed_superset_video_path),
                            format="mp4",
                            caption=f"Superset Failure: {superset_prompt}"
                        ),
                    })
            except Exception as e:
                logging.warning(f"Failed to log to wandb: {e}")
        
        logging.info(f"\nTask {task_idx} Results:")
        logging.info(f"  Subset success rate: {subset_success_count}/{args.hierarchical_subset_rollouts} ({subset_success_rate*100:.1f}%)")
        logging.info(f"  Superset success rate: {superset_success_count}/{args.hierarchical_superset_rollouts} ({superset_success_rate*100:.1f}%)")
    
    # Create side-by-side comparison plots for each task
    if args.use_wandb and wandb_run is not None:
        try:
            import wandb
            
            # Create a plot/visualization for each task showing 3 videos side-by-side
            for task_idx in sorted(task_results.keys()):
                result = task_results[task_idx]
                
                # Prepare videos with captions
                subset_video = wandb.Video(
                    str(result["successful_subset_video"] or result["failed_subset_video"]),
                    format="mp4",
                    caption=f"Original (Subset): {result['subset_prompt']}"
                ) if (result.get("successful_subset_video") or result.get("failed_subset_video")) else None
                
                success_video = wandb.Video(
                    str(result["successful_superset_video"]),
                    format="mp4",
                    caption=f"Superset Success: {result['superset_prompt']}"
                ) if result["successful_superset_video"] else None
                
                failure_video = wandb.Video(
                    str(result["failed_superset_video"]),
                    format="mp4",
                    caption=f"Superset Failure: {result['superset_prompt']}"
                ) if result["failed_superset_video"] else None
                
                # Create a table with the 3 videos side-by-side
                # This displays videos side-by-side in wandb
                video_table = wandb.Table(
                    columns=["Original (Subset)", "Superset Success", "Superset Failure"],
                    data=[[subset_video, success_video, failure_video]],
                )
                
                # Create a visual plot showing the layout and success count
                fig = plt.figure(figsize=(18, 7))
                
                # Create 3 subplots side by side
                ax1 = plt.subplot(1, 3, 1)
                ax2 = plt.subplot(1, 3, 2)
                ax3 = plt.subplot(1, 3, 3)
                
                # Remove axes
                ax1.axis('off')
                ax2.axis('off')
                ax3.axis('off')
                
                # Title with success count at the top
                fig.suptitle(
                    f"Task {task_idx}: {result['task_name']}\n"
                    f"Subset Success: {result['subset_success_count']}/{args.hierarchical_subset_rollouts} ({result['subset_success_rate']*100:.1f}%) | "
                    f"Superset Success: {result['superset_success_count']}/{args.hierarchical_superset_rollouts} ({result['superset_success_rate']*100:.1f}%)",
                    fontsize=14,
                    fontweight='bold',
                    y=0.98
                )
                
                # Subset video panel (left)
                ax1.text(0.5, 0.95, "Original (Subset)", ha='center', va='top', 
                        transform=ax1.transAxes, fontsize=14, fontweight='bold')
                # Wrap text for long prompts
                subset_text = result['subset_prompt']
                if len(subset_text) > 60:
                    # Split into multiple lines
                    words = subset_text.split()
                    lines = []
                    current_line = []
                    current_len = 0
                    for word in words:
                        if current_len + len(word) + 1 > 60:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                            current_len = len(word)
                        else:
                            current_line.append(word)
                            current_len += len(word) + 1
                    if current_line:
                        lines.append(' '.join(current_line))
                    subset_text = '\n'.join(lines)
                ax1.text(0.5, 0.15, subset_text, ha='center', va='bottom',
                        transform=ax1.transAxes, fontsize=11, wrap=True,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
                subset_status = "✓ SUCCESS" if result['subset_success_rate'] > 0 else "✗ FAILURE"
                ax1.text(0.5, 0.5, subset_status, 
                        ha='center', va='center', transform=ax1.transAxes, fontsize=18,
                        color='green' if result['subset_success_rate'] > 0 else 'red', fontweight='bold')
                
                # Successful superset video panel (middle)
                if result['successful_superset_video']:
                    ax2.text(0.5, 0.95, "Superset Success", ha='center', va='top',
                            transform=ax2.transAxes, fontsize=14, fontweight='bold', color='green')
                    superset_text = result['superset_prompt']
                    if len(superset_text) > 60:
                        words = superset_text.split()
                        lines = []
                        current_line = []
                        current_len = 0
                        for word in words:
                            if current_len + len(word) + 1 > 60:
                                lines.append(' '.join(current_line))
                                current_line = [word]
                                current_len = len(word)
                            else:
                                current_line.append(word)
                                current_len += len(word) + 1
                        if current_line:
                            lines.append(' '.join(current_line))
                        superset_text = '\n'.join(lines)
                    ax2.text(0.5, 0.15, superset_text, ha='center', va='bottom',
                            transform=ax2.transAxes, fontsize=11, wrap=True,
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
                    ax2.text(0.5, 0.5, "✓ SUCCESS", ha='center', va='center',
                            transform=ax2.transAxes, fontsize=18, color='green', fontweight='bold')
                else:
                    ax2.text(0.5, 0.5, "No successful\nsuperset rollout", ha='center', va='center',
                            transform=ax2.transAxes, fontsize=14, color='gray', style='italic')
                    ax2.text(0.5, 0.15, result['superset_prompt'], ha='center', va='bottom',
                            transform=ax2.transAxes, fontsize=11, wrap=True,
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
                
                # Failed superset video panel (right)
                if result['failed_superset_video']:
                    ax3.text(0.5, 0.95, "Superset Failure", ha='center', va='top',
                            transform=ax3.transAxes, fontsize=14, fontweight='bold', color='red')
                    superset_text = result['superset_prompt']
                    if len(superset_text) > 60:
                        words = superset_text.split()
                        lines = []
                        current_line = []
                        current_len = 0
                        for word in words:
                            if current_len + len(word) + 1 > 60:
                                lines.append(' '.join(current_line))
                                current_line = [word]
                                current_len = len(word)
                            else:
                                current_line.append(word)
                                current_len += len(word) + 1
                        if current_line:
                            lines.append(' '.join(current_line))
                        superset_text = '\n'.join(lines)
                    ax3.text(0.5, 0.15, superset_text, ha='center', va='bottom',
                            transform=ax3.transAxes, fontsize=11, wrap=True,
                            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
                    ax3.text(0.5, 0.5, "✗ FAILURE", ha='center', va='center',
                            transform=ax3.transAxes, fontsize=18, color='red', fontweight='bold')
                else:
                    ax3.text(0.5, 0.5, "No failed\nsuperset rollout", ha='center', va='center',
                            transform=ax3.transAxes, fontsize=14, color='gray', style='italic')
                    ax3.text(0.5, 0.15, result['superset_prompt'], ha='center', va='bottom',
                            transform=ax3.transAxes, fontsize=11, wrap=True,
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
                
                plt.tight_layout()
                
                # Log both the visual plot and the video table
                wandb_run.log({
                    f"{experiment_type}/task_{task_idx}/comparison_plot": wandb.Image(fig),
                    f"{experiment_type}/task_{task_idx}/videos": video_table,
                })
                plt.close(fig)
            
            # Create bar chart comparing success rates
            fig, ax = plt.subplots(figsize=(12, 6))
            
            task_labels = [f"Task {idx}" for idx in sorted(task_results.keys())]
            subset_rates = [task_results[idx]["subset_success_rate"] for idx in sorted(task_results.keys())]
            superset_rates = [task_results[idx]["superset_success_rate"] for idx in sorted(task_results.keys())]
            
            x = np.arange(len(task_labels))
            width = 0.35
            
            ax.bar(x - width/2, subset_rates, width, label="Subset (Original)", alpha=0.8)
            ax.bar(x + width/2, superset_rates, width, label="Superset (Abstract)", alpha=0.8)
            
            ax.set_xlabel("Task")
            ax.set_ylabel("Success Rate")
            ax.set_title("Hierarchical Language Experiment: Subset vs Superset Success Rates")
            ax.set_xticks(x)
            ax.set_xticklabels(task_labels)
            ax.legend()
            ax.set_ylim([0, 1.1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (subset_val, superset_val) in enumerate(zip(subset_rates, superset_rates)):
                ax.text(i - width/2, subset_val + 0.02, f"{subset_val:.0%}", ha='center', va='bottom')
                ax.text(i + width/2, superset_val + 0.02, f"{superset_val:.0%}", ha='center', va='bottom')
            
            plt.tight_layout()
            wandb_run.log({f"{experiment_type}/success_rate_comparison": wandb.Image(fig)})
            plt.close(fig)
            
            # Log summary metrics
            avg_subset_rate = np.mean([r["subset_success_rate"] for r in task_results.values()])
            avg_superset_rate = np.mean([r["superset_success_rate"] for r in task_results.values()])
            
            wandb_run.log({
                f"{experiment_type}/summary/total_tasks": len(task_results),
                f"{experiment_type}/summary/avg_subset_success_rate": avg_subset_rate,
                f"{experiment_type}/summary/avg_superset_success_rate": avg_superset_rate,
            })
            
            wandb_run.finish()
            logging.info("Wandb run completed and logged.")
        except Exception as e:
            logging.warning(f"Failed to create wandb visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    logging.info(f"\n{'='*60}")
    logging.info("HIERARCHICAL EXPERIMENT SUMMARY")
    logging.info(f"{'='*60}")
    avg_subset_rate = np.mean([r["subset_success_rate"] for r in task_results.values()])
    avg_superset_rate = np.mean([r["superset_success_rate"] for r in task_results.values()])
    
    logging.info(f"Total tasks: {len(task_results)}")
    logging.info(f"Average subset success rate: {avg_subset_rate*100:.1f}%")
    logging.info(f"Average superset success rate: {avg_superset_rate*100:.1f}%")
    logging.info(f"Videos saved to: {video_out_path}")


def run_list_based_experiment(experiment_type: str, args: Args) -> None:
    """Run list-based language augmentation experiments (task_completion_stop, kinematic, etc.)."""
    import datetime
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Update video path to include experiment type
    video_out_path = pathlib.Path(args.video_out_path) / experiment_type
    video_out_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if requested
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"{experiment_type}-experiment-{timestamp}",
                config={
                    "experiment_type": experiment_type,
                    "task_suite": args.task_suite_name,
                    "num_rollouts_per_prompt": args.num_rollouts_per_prompt,
                    "seed": args.seed,
                    "host": args.host,
                    "port": args.port,
                }
            )
            logging.info(f"Initialized wandb run: {wandb_run.name}")
        except ImportError:
            logging.warning("wandb not installed. Install with: uv pip install wandb")
            args.use_wandb = False
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {e}")
            args.use_wandb = False

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    
    # Set max steps for libero_10
    max_steps = 520
    
    # Connect to policy server
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info(f"Connected to policy server at {args.host}:{args.port}")
    
    # Store results for each task
    task_results = {}
    
    # Process each task that has this experiment type
    task_indices = sorted([idx for idx in AUGMENTED_EXPERIMENTS.keys() if experiment_type in AUGMENTED_EXPERIMENTS[idx]])
    
    if not task_indices:
        logging.warning(f"No tasks found for experiment type: {experiment_type}")
        return
    
    for task_idx in tqdm.tqdm(task_indices, desc=f"Processing {experiment_type} tasks"):
        if experiment_type not in AUGMENTED_EXPERIMENTS[task_idx]:
            logging.warning(f"Skipping task {task_idx}: no {experiment_type} experiment defined")
            continue
        
        prompts = AUGMENTED_EXPERIMENTS[task_idx][experiment_type]
        if not isinstance(prompts, list):
            logging.warning(f"Task {task_idx}: {experiment_type} prompts must be a list")
            continue
        
        task = task_suite.get_task(task_idx)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Task {task_idx}: {task.language}")
        logging.info(f"Experiment type: {experiment_type}")
        logging.info(f"Number of prompts: {len(prompts)}")
        logging.info(f"{'='*60}")
        
        # Get initial states
        initial_states = task_suite.get_task_init_states(task_idx)
        
        # Initialize environment
        env = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        
        # Create task-specific directory
        task_name_safe = _make_prompt_safe(task.language)
        task_video_dir = video_out_path / f"task_{task_idx}_{task_name_safe}"
        task_video_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results for each prompt
        prompt_results = []
        
        for prompt_idx, prompt in enumerate(tqdm.tqdm(prompts, desc=f"Prompts for task {task_idx}")):
            prompt_safe = _make_prompt_safe(prompt)
            prompt_dir = task_video_dir / f"prompt_{prompt_idx:02d}_{prompt_safe}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            
            # Run rollouts for this prompt
            prompt_successes = []
            prompt_videos = []
            
            for rollout_idx in range(args.num_rollouts_per_prompt):
                initial_state = initial_states[rollout_idx % len(initial_states)]
                
                success, replay_images = run_single_rollout(
                    env, initial_state, prompt, client, args, max_steps
                )
                prompt_successes.append(success)
                
                # Save all videos
                suffix = "success" if success else "failure"
                video_path = prompt_dir / f"rollout_{rollout_idx+1:02d}_{suffix}.mp4"
                imageio.mimwrite(
                    video_path,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
                prompt_videos.append((video_path, success))
                
                logging.info(f"  Prompt {prompt_idx+1}, Rollout {rollout_idx+1}: {'✓ SUCCESS' if success else '✗ FAILURE'} - Saved to {video_path}")
                
                # Log to wandb
                if args.use_wandb and wandb_run is not None:
                    try:
                        import wandb
                        wandb_run.log({
                            f"{experiment_type}/task_{task_idx}/prompt_{prompt_idx}/rollout_{rollout_idx+1}": wandb.Video(
                                str(video_path),
                                format="mp4",
                                caption=prompt
                            ),
                            f"{experiment_type}/task_{task_idx}/prompt_{prompt_idx}/success": 1.0 if success else 0.0,
                        })
                    except Exception as e:
                        logging.warning(f"Failed to log to wandb: {e}")
            
            # Calculate metrics for this prompt
            prompt_success_count = sum(prompt_successes)
            prompt_success_rate = float(prompt_success_count) / args.num_rollouts_per_prompt if args.num_rollouts_per_prompt > 0 else 0.0
            
            prompt_results.append({
                "prompt": prompt,
                "prompt_idx": prompt_idx,
                "success_count": prompt_success_count,
                "success_rate": prompt_success_rate,
                "videos": prompt_videos,
            })
            
            logging.info(f"\nPrompt '{prompt}' results:")
            logging.info(f"  Successes: {prompt_success_count}/{args.num_rollouts_per_prompt} ({prompt_success_rate*100:.1f}%)")
        
        # Store task results
        task_results[task_idx] = {
            "task_name": task.language,
            "prompt_results": prompt_results,
        }
        
        # Log task summary to wandb
        if args.use_wandb and wandb_run is not None:
            try:
                import wandb
                # Create bar chart for this task
                fig, ax = plt.subplots(figsize=(12, 6))
                
                prompt_labels = [f"P{i+1}" for i in range(len(prompt_results))]
                success_rates = [pr["success_rate"] for pr in prompt_results]
                
                bars = ax.bar(prompt_labels, success_rates, alpha=0.8)
                ax.set_xlabel("Prompt")
                ax.set_ylabel("Success Rate")
                ax.set_title(f"Task {task_idx}: {task.language} - Success Rates by Prompt")
                ax.set_ylim([0, 1.1])
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, rate) in enumerate(zip(bars, success_rates)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f"{rate:.0%}", ha='center', va='bottom')
                
                plt.tight_layout()
                wandb_run.log({f"{experiment_type}/task_{task_idx}/success_rate_chart": wandb.Image(fig)})
                plt.close(fig)
                
                # Create table with sample videos
                sample_videos = []
                for pr in prompt_results:
                    # Get first success and first failure videos
                    success_video = next((v for v, s in pr["videos"] if s), None)
                    failure_video = next((v for v, s in pr["videos"] if not s), None)
                    
                    success_wandb = wandb.Video(str(success_video), format="mp4", caption=f"Success: {pr['prompt']}") if success_video else None
                    failure_wandb = wandb.Video(str(failure_video), format="mp4", caption=f"Failure: {pr['prompt']}") if failure_video else None
                    
                    sample_videos.append([pr["prompt"], pr["success_rate"], success_wandb, failure_wandb])
                
                video_table = wandb.Table(
                    columns=["Prompt", "Success Rate", "Sample Success Video", "Sample Failure Video"],
                    data=sample_videos
                )
                wandb_run.log({f"{experiment_type}/task_{task_idx}/videos_table": video_table})
                
            except Exception as e:
                logging.warning(f"Failed to create wandb visualizations: {e}")
    
    # Create overall summary visualization
    if args.use_wandb and wandb_run is not None:
        try:
            import wandb
            
            # Overall success rate bar chart across all tasks
            fig, ax = plt.subplots(figsize=(14, 8))
            
            task_labels = [f"Task {idx}" for idx in sorted(task_results.keys())]
            avg_success_rates = []
            
            for task_idx in sorted(task_results.keys()):
                prompt_rates = [pr["success_rate"] for pr in task_results[task_idx]["prompt_results"]]
                avg_rate = np.mean(prompt_rates) if prompt_rates else 0.0
                avg_success_rates.append(avg_rate)
            
            bars = ax.bar(task_labels, avg_success_rates, alpha=0.8)
            ax.set_xlabel("Task")
            ax.set_ylabel("Average Success Rate")
            ax.set_title(f"{experiment_type.replace('_', ' ').title()} Experiment: Average Success Rate by Task")
            ax.set_ylim([0, 1.1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, rate in zip(bars, avg_success_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f"{rate:.0%}", ha='center', va='bottom')
            
            plt.tight_layout()
            wandb_run.log({f"{experiment_type}/overall/success_rate_by_task": wandb.Image(fig)})
            plt.close(fig)
            
            # Log summary metrics
            all_success_rates = []
            total_rollouts = 0
            total_successes = 0
            
            for task_idx in sorted(task_results.keys()):
                for pr in task_results[task_idx]["prompt_results"]:
                    all_success_rates.append(pr["success_rate"])
                    total_rollouts += args.num_rollouts_per_prompt
                    total_successes += pr["success_count"]
            
            overall_success_rate = float(total_successes) / total_rollouts if total_rollouts > 0 else 0.0
            avg_success_rate = np.mean(all_success_rates) if all_success_rates else 0.0
            
            wandb_run.log({
                f"{experiment_type}/summary/total_tasks": len(task_results),
                f"{experiment_type}/summary/total_prompts": sum(len(tr["prompt_results"]) for tr in task_results.values()),
                f"{experiment_type}/summary/total_rollouts": total_rollouts,
                f"{experiment_type}/summary/total_successes": total_successes,
                f"{experiment_type}/summary/overall_success_rate": overall_success_rate,
                f"{experiment_type}/summary/avg_success_rate": avg_success_rate,
            })
            
            wandb_run.finish()
            logging.info("Wandb run completed and logged.")
        except Exception as e:
            logging.warning(f"Failed to create summary visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    logging.info(f"\n{'='*60}")
    logging.info(f"{experiment_type.upper().replace('_', ' ')} EXPERIMENT SUMMARY")
    logging.info(f"{'='*60}")
    
    total_rollouts = 0
    total_successes = 0
    
    for task_idx in sorted(task_results.keys()):
        task_result = task_results[task_idx]
        logging.info(f"\nTask {task_idx}: {task_result['task_name']}")
        for pr in task_result["prompt_results"]:
            logging.info(f"  Prompt '{pr['prompt'][:50]}...': {pr['success_count']}/{args.num_rollouts_per_prompt} ({pr['success_rate']*100:.1f}%)")
            total_rollouts += args.num_rollouts_per_prompt
            total_successes += pr["success_count"]
    
    overall_success_rate = float(total_successes) / total_rollouts if total_rollouts > 0 else 0.0
    logging.info(f"\nTotal rollouts: {total_rollouts}")
    logging.info(f"Total successes: {total_successes}")
    logging.info(f"Overall success rate: {overall_success_rate*100:.1f}%")
    logging.info(f"Videos saved to: {video_out_path}")


def eval_custom_prompts(args: Args) -> None:
    """Run evaluation with custom prompts."""
    # Set random seed
    np.random.seed(args.seed)
    
    # Initialize wandb if requested
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "task_suite": args.task_suite_name,
                    "num_prompts": len(CUSTOM_PROMPTS),
                    "num_trials_per_prompt": args.num_trials_per_prompt,
                    "seed": args.seed,
                    "host": args.host,
                    "port": args.port,
                }
            )
            logging.info(f"Initialized wandb run: {wandb_run.name}")
        except ImportError:
            logging.warning("wandb not installed. Install with: uv pip install wandb")
            args.use_wandb = False
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {e}")
            args.use_wandb = False

    # Get prompts to use
    prompts_to_use = args.custom_prompts if args.custom_prompts is not None else CUSTOM_PROMPTS
    # Check if we're using the default CUSTOM_PROMPTS (by comparing content, not reference)
    is_using_default_prompts = (
        args.custom_prompts is None or 
        (len(prompts_to_use) == len(CUSTOM_PROMPTS) and 
         all(p.lower() == c.lower() for p, c in zip(prompts_to_use, CUSTOM_PROMPTS)))
    )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(f"Number of tasks in suite: {num_tasks_in_suite}")
    logging.info(f"Number of custom prompts: {len(prompts_to_use)}")
    logging.info(f"Rollouts per prompt: {args.num_trials_per_prompt}")

    # Check if we have enough tasks
    if len(prompts_to_use) > num_tasks_in_suite:
        logging.warning(
            f"Warning: More prompts ({len(prompts_to_use)}) than tasks ({num_tasks_in_suite}). "
            f"Will only use first {num_tasks_in_suite} prompts."
        )
        prompts_to_use = prompts_to_use[:num_tasks_in_suite]

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Set max steps for libero_10
    max_steps = 520  # longest training demo has 505 steps

    # Connect to policy server
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info(f"Connected to policy server at {args.host}:{args.port}")

    # Start evaluation
    total_episodes, total_successes = 0, 0

    for prompt_idx, custom_prompt in enumerate(tqdm.tqdm(prompts_to_use, desc="Processing prompts")):
        # Map custom prompt to the correct libero_10 task
        # If using default CUSTOM_PROMPTS, use the predefined mapping
        # Otherwise, try to find matching task by language, or fall back to sequential
        if is_using_default_prompts and prompt_idx in CUSTOM_PROMPT_TO_TASK_IDX:
            task_idx = CUSTOM_PROMPT_TO_TASK_IDX[prompt_idx]
        else:
            # Try to find matching task by language
            task_idx = None
            for idx in range(num_tasks_in_suite):
                task = task_suite.get_task(idx)
                if custom_prompt.lower() == task.language.lower():
                    task_idx = idx
                    break
            # Fall back to sequential if no match found
            if task_idx is None:
                task_idx = prompt_idx % num_tasks_in_suite
        
        task = task_suite.get_task(task_idx)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_idx)

        # Initialize LIBERO environment
        env = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        logging.info(f"\n{'='*60}")
        logging.info(f"Prompt {prompt_idx + 1}/{len(prompts_to_use)}: {custom_prompt}")
        logging.info(f"Using task {task_idx}: {task.language}")
        logging.info(f"{'='*60}")

        # Run rollouts for this prompt
        prompt_episodes, prompt_successes = 0, 0

        for episode_idx in tqdm.tqdm(
            range(args.num_trials_per_prompt), desc=f"Rollouts for prompt {prompt_idx + 1}"
        ):
            # Use the corresponding initial state (cycle if needed)
            initial_state = initial_states[episode_idx % len(initial_states)]

            # Run single rollout
            success, replay_images = run_single_rollout(
                env, initial_state, custom_prompt, client, args, max_steps
            )

            prompt_episodes += 1
            total_episodes += 1
            if success:
                prompt_successes += 1
                total_successes += 1

            # Save video
            suffix = "success" if success else "failure"
            # Create safe filename from prompt
            prompt_safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in custom_prompt)
            prompt_safe = prompt_safe.replace(" ", "_").lower()[:50]  # Limit length
            video_filename = f"prompt_{prompt_idx+1:02d}_rollout_{episode_idx+1:02d}_{suffix}.mp4"
            video_path = pathlib.Path(args.video_out_path) / prompt_safe / video_filename
            video_path.parent.mkdir(parents=True, exist_ok=True)

            imageio.mimwrite(
                video_path,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            logging.info(
                f"  Rollout {episode_idx + 1}: {'✓ SUCCESS' if success else '✗ FAILURE'} - Saved to {video_path}"
            )
            
            # Log to wandb if enabled
            if args.use_wandb and wandb_run is not None:
                try:
                    import wandb
                    # Log video to wandb - use file path instead of raw array to avoid moviepy requirement
                    wandb_run.log({
                        f"videos/prompt_{prompt_idx+1}_{prompt_safe}/rollout_{episode_idx+1}": wandb.Video(
                            str(video_path),
                            format="mp4"
                        ),
                        f"metrics/prompt_{prompt_idx+1}/success": 1.0 if success else 0.0,
                        f"metrics/prompt_{prompt_idx+1}/rollout_num": episode_idx + 1,
                    }, step=total_episodes)
                except Exception as e:
                    logging.warning(f"Failed to log to wandb: {e}")

        # Log results for this prompt
        prompt_success_rate = float(prompt_successes) / float(prompt_episodes) if prompt_episodes > 0 else 0.0
        logging.info(f"\nPrompt '{custom_prompt}' results:")
        logging.info(f"  Successes: {prompt_successes}/{prompt_episodes} ({prompt_success_rate*100:.1f}%)")
        logging.info(f"  Videos saved to: {pathlib.Path(args.video_out_path) / prompt_safe}")
        
        # Log prompt summary to wandb
        if args.use_wandb and wandb_run is not None:
            try:
                import wandb
                wandb_run.log({
                    f"prompt_summary/prompt_{prompt_idx+1}_success_rate": prompt_success_rate,
                    f"prompt_summary/prompt_{prompt_idx+1}_successes": prompt_successes,
                    f"prompt_summary/prompt_{prompt_idx+1}_total": prompt_episodes,
                    f"prompt_summary/prompt_{prompt_idx+1}_name": custom_prompt,
                })
            except Exception as e:
                logging.warning(f"Failed to log prompt summary to wandb: {e}")

    # Final summary
    overall_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    logging.info(f"\n{'='*60}")
    logging.info("FINAL SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Total successes: {total_successes}")
    logging.info(f"Overall success rate: {overall_success_rate*100:.1f}%")
    logging.info(f"Videos saved to: {args.video_out_path}")
    
    # Log final summary to wandb
    if args.use_wandb and wandb_run is not None:
        try:
            import wandb
            wandb_run.log({
                "summary/total_episodes": total_episodes,
                "summary/total_successes": total_successes,
                "summary/overall_success_rate": overall_success_rate,
            })
            wandb_run.finish()
            logging.info("Wandb run completed and logged.")
        except Exception as e:
            logging.warning(f"Failed to log final summary to wandb: {e}")


def run_all_experiments(args: Args) -> None:
    """Run all experiment types sequentially."""
    # Get all available experiment types from AUGMENTED_EXPERIMENTS
    all_experiment_types = set()
    for task_data in AUGMENTED_EXPERIMENTS.values():
        all_experiment_types.update(task_data.keys())
    
    # Remove "hierarchical" as it's handled separately
    list_based_types = sorted([et for et in all_experiment_types if et != "hierarchical"])
    
    logging.info(f"Running all experiment types: hierarchical, {', '.join(list_based_types)}")
    
    # Run hierarchical first
    logging.info("\n" + "="*60)
    logging.info("RUNNING HIERARCHICAL EXPERIMENT")
    logging.info("="*60)
    run_hierarchical_experiment(args)
    
    # Then run all list-based types
    for experiment_type in list_based_types:
        logging.info("\n" + "="*60)
        logging.info(f"RUNNING {experiment_type.upper().replace('_', ' ')} EXPERIMENT")
        logging.info("="*60)
        run_list_based_experiment(experiment_type, args)


def main(args: Args) -> None:
    """Main entry point that routes to the appropriate experiment type."""
    if args.experiment_type == "run_all":
        run_all_experiments(args)
    elif args.experiment_type == "hierarchical":
        run_hierarchical_experiment(args)
    elif args.experiment_type in ["task_completion_stop", "kinematic", "goal_state_synonym", 
                                   "logical_constraint_negation", "undo", "task_decomposition"]:
        run_list_based_experiment(args.experiment_type, args)
    else:
        # Default to custom prompts evaluation
        eval_custom_prompts(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    tyro.cli(main)

