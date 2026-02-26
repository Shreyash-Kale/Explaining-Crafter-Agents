# environment.py - Updated to save MIXTAPE JSON files in MIXTAPE_Results folder

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import gym
import crafter
import numpy as np
import pandas as pd
import cv2
from dreamer.policy import DreamerPolicy
from pathlib import Path
import json
import base64
import imageio.v2 as iio
from io import BytesIO
from PIL import Image

# Configuration constants
BASE_CHECKPOINT_DIR = './saved_checkpoints/ckpt_'
RESULTS_DIR = './results'
MIXTAPE_RESULTS_DIR = './MIXTAPE_Results'  # New folder for MIXTAPE JSON files
DEFAULT_NUM_EPISODES = 50
DEFAULT_CHECKPOINT = 530000

# Crafter action mapping for MIXTAPE
CRAFTER_ACTION_MAPPING = {
    "0": "noop",
    "1": "move_left", 
    "2": "move_right",
    "3": "move_up",
    "4": "move_down",
    "5": "do",
    "6": "sleep",
    "7": "place_stone",
    "8": "place_table",
    "9": "place_furnace",
    "10": "place_plant",
    "11": "make_wood_pickaxe",
    "12": "make_stone_pickaxe",
    "13": "make_iron_pickaxe",
    "14": "make_wood_sword",
    "15": "make_stone_sword",
    "16": "make_iron_sword"
}

def create_environment(output_dir: str = "./results"):
    """Crafter with dense reward + recorder + safe metadata for Gym 0.25+."""
    # 1) Base env with dense reward
    env = gym.make("CrafterReward-v1")
    
    # 2) Initialize metadata if not present
    if env.metadata is None:
        env.metadata = {}
    env.metadata["render_modes"] = ["rgb_array"]
    
    # 3) Ensure unwrapped env also has metadata
    if hasattr(env.unwrapped, "metadata"):
        if env.unwrapped.metadata is None:
            env.unwrapped.metadata = {}
        env.unwrapped.metadata.setdefault("render_modes", ["rgb_array"])
    else:
        env.unwrapped.metadata = {"render_modes": ["rgb_array"]}
    
    # 4) Wrap with Crafter's recorder
    env = crafter.Recorder(
        env,
        output_dir,
        save_stats=True,
        save_episode=True,
        save_video=True,
    )
    
    return env


def encode_image_to_base64(image_array):
    """Convert numpy image array to base64 string."""
    if image_array is None:
        return None
    
    try:
        # Convert to PIL Image
        if len(image_array.shape) == 3:
            image = Image.fromarray(image_array.astype('uint8'))
        else:
            return None
            
        # Save to bytes
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_base64
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def create_observation_space(step_info, env_state=None):
    """Convert rich game state to simplified 1D observation array with ALL achievements."""
    obs = []
    
    # Basic survival stats (4 values)
    stats = ["health", "food", "drink", "energy"]
    for stat in stats:
        if stat in step_info:
            obs.append(float(step_info[stat]))
        elif hasattr(env_state, stat) if env_state else False:
            obs.append(float(getattr(env_state, stat)))
        else:
            obs.append(9.0)  # Default starting value
    
    # Inventory items (12 values)
    inventory_items = [
        "sapling", "wood", "stone", "coal", "iron", "diamond",
        "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
        "wood_sword", "stone_sword", "iron_sword"
    ]
    
    for item in inventory_items:
        if "inventory" in step_info and item in step_info["inventory"]:
            obs.append(float(step_info["inventory"][item]))
        elif item in step_info:
            obs.append(float(step_info[item]))
        elif hasattr(env_state, item) if env_state else False:
            obs.append(float(getattr(env_state, item)))
        else:
            obs.append(0.0)
    
    # ALL 22 achievements as binary flags (22 values)
    all_achievements = [
        "collect_coal", "collect_diamond", "collect_drink", "collect_iron", 
        "collect_sapling", "collect_stone", "collect_wood",
        "defeat_skeleton", "defeat_zombie", "eat_cow", "eat_plant",
        "make_iron_pickaxe", "make_iron_sword", "make_stone_pickaxe", 
        "make_stone_sword", "make_wood_pickaxe", "make_wood_sword",
        "place_furnace", "place_plant", "place_stone", "place_table",
        "wake_up"
    ]
    
    for ach in all_achievements:
        if "achievements" in step_info and ach in step_info["achievements"]:
            obs.append(float(step_info["achievements"][ach]))
        elif ach in step_info:
            obs.append(float(step_info[ach]))
        else:
            obs.append(0.0)
    
    return obs

def run_episode_mixtape(
    env,
    policy_fn,
    max_frames: int = 1_000,
    output_dir: str = "./results",
    episode_id: int = 0,
    checkpoint_number: int = None,
    total_training_steps: int = 500000,
    include_images: bool = False,
):
    """
    Execute ONE episode and save in MIXTAPE JSON format.
    """
    # Keep original output_dir for CSV/video files (compatibility)
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create MIXTAPE_Results directory structure
    mixtape_base_dir = Path(MIXTAPE_RESULTS_DIR).expanduser()
    mixtape_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint-specific directory within MIXTAPE_Results
    if checkpoint_number:
        mixtape_checkpoint_dir = mixtape_base_dir / f"checkpoint_{checkpoint_number}"
        mixtape_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        base_filename = f"ckpt{checkpoint_number}_episode{episode_id:03d}_mixtape"
    else:
        mixtape_checkpoint_dir = mixtape_base_dir / "default_checkpoint"
        mixtape_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        base_filename = f"episode_{episode_id:03d}_mixtape"
    
    # JSON file goes to MIXTAPE_Results folder
    json_path = mixtape_checkpoint_dir / f"{base_filename}.json"
    
    # Resolve the policy object for decision attribution
    policy_obj = getattr(policy_fn, "__self__", policy_fn)
    
    # Initialize MIXTAPE data structure
    mixtape_data = {
        "action_mapping": CRAFTER_ACTION_MAPPING,
        "training": {
            "environment": "Crafter-v1",
            "algorithm": "DreamerV2",
            "parallel": False,
            "num_gpus": 0.0,
            "iterations": total_training_steps,
            "config": {
                "discrete_size": 32,
                "discrete_classes": 32,
                "mixed_precision": True,
                "max_frames": max_frames
            }
        },
        "inference": {
            "parallel": False,
            "config": {},
            "steps": []
        }
    }
    
    # Episode execution
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs, info = reset_result, {}
    
    cumulative_r = 0.0
    last_achievements = {}
    every_n_steps = 10  # Capture image every 10 steps

    for t in range(max_frames):
        # 1) Get action
        action = policy_fn(obs)
        
        # 2) Get decision attribution if available
        attribution = {}
        if hasattr(policy_obj, "decision_attribution"):
            try:
                attribution = policy_obj.decision_attribution(obs) or {}
            except Exception as e:
                print(f"Attribution error @t={t}: {e}")
        
        # 3) Environment step
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, r, terminated, truncated, step_info = step_result
            done = terminated or truncated
        else:
            next_obs, r, done, step_info = step_result
        
        cumulative_r += r
        
        # 4) Image capture logic
        image_b64 = None
        if include_images:
            should_capture = False
            
            # Initialize achievements tracker after first step
            if t == 0 and "achievements" in step_info:
                last_achievements = {k: False for k in step_info["achievements"]}
            
            # Check for achievement unlocks
            if "achievements" in step_info:
                # Check if any new achievement was unlocked
                for ach in step_info["achievements"]:
                    if step_info["achievements"][ach] and not last_achievements.get(ach, False):
                        should_capture = True
                # Update last achievements
                last_achievements = dict(step_info["achievements"])
            
            # Check for agent death
            if ("inventory" in step_info and step_info["inventory"].get("health", 1) == 0) or done:
                should_capture = True
            
            # Capture every N steps
            if t % every_n_steps == 0:
                should_capture = True
            
            if should_capture:
                try:
                    frame = env.render()
                    image_b64 = encode_image_to_base64(frame)
                except Exception as e:
                    print(f"Image capture failed at step {t}: {str(e)}")
        
        # 5) Create observation space
        observation_space = create_observation_space(step_info, getattr(env, 'state', None))
        
        # 6) Create MIXTAPE step
        step_data = {
            "number": t,
            "agent_steps": [{
                "agent": "dreamer_agent",
                "action": int(action) if np.isscalar(action) else int(action[0]),
                "reward": float(r),
                "observation_space": observation_space
            }]
        }
        
        # Add image if available
        if image_b64:
            step_data["image"] = image_b64
        
        # Add decision attribution as custom fields
        if attribution:
            step_data["agent_steps"][0].update({
                "action_probability": float(attribution.get("action_probability", 0.0)),
                "value_estimate": float(attribution.get("value_estimate", 0.0)),
                "exploration_bonus": float(attribution.get("exploration_bonus", 0.0)),
                "world_model_score": float(attribution.get("world_model_score", 0.0)),
                "cumulative_reward": float(cumulative_r)
            })
        
        # Add achievements and inventory as custom fields
        if "achievements" in step_info:
            step_data["agent_steps"][0]["achievements"] = {
                k: bool(v) for k, v in step_info["achievements"].items()
            }
        
        if "inventory" in step_info:
            step_data["agent_steps"][0]["inventory"] = {
                k: int(v) for k, v in step_info["inventory"].items()
            }
        
        # Add to steps
        mixtape_data["inference"]["steps"].append(step_data)
        
        # Advance
        obs = next_obs
        if done:
            break
    
    # Save MIXTAPE JSON to the new directory
    with json_path.open("w") as f:
        json.dump(mixtape_data, f, indent=2)
    
    print(f"Episode completed with total reward: {cumulative_r:.2f}")
    print(f"MIXTAPE JSON saved: {json_path}")
    
    return cumulative_r, json_path

def find_latest_checkpoint(checkpoint_dir):
    """Helper function to find the latest checkpoint number."""
    import os
    import re
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('ckpt-') and f.endswith('.index')]
    if not checkpoint_files:
        return None
    
    checkpoint_numbers = [int(re.search(r'ckpt-(\d+)', f).group(1)) for f in checkpoint_files]
    return max(checkpoint_numbers)

def run_with_dreamer_mixtape(
    checkpoint_number=270000, 
    checkpoint=None, 
    num_episodes=DEFAULT_NUM_EPISODES, 
    results_dir=RESULTS_DIR,
    include_images=False
):
    """Run multiple episodes with DreamerV2 and save in MIXTAPE format"""
    
    # Construct the full checkpoint directory path
    checkpoint_dir = f"{BASE_CHECKPOINT_DIR}{checkpoint_number}"
    
    # Find the latest checkpoint if not specified
    if checkpoint is None:
        checkpoint = find_latest_checkpoint(checkpoint_dir)
        if checkpoint is None:
            print(f"No checkpoints found in {checkpoint_dir}. Cannot run evaluation.")
            return None
    
    print(f"Using checkpoint: {checkpoint} from {checkpoint_dir}")
    
    # Create algorithm directory structure (for CSV/video files)
    algo_dir = os.path.join(results_dir, 'dreamer_v2_mixtape')
    checkpoint_dir_output = os.path.join(algo_dir, f'checkpoint_{checkpoint}')
    os.makedirs(checkpoint_dir_output, exist_ok=True)
    
    # Run episodes
    results = []
    for episode in range(1, num_episodes + 1):
        # Create episode-specific directory for CSV/video files
        episode_dir = os.path.join(checkpoint_dir_output, f'episode_{episode:03d}')
        os.makedirs(episode_dir, exist_ok=True)
        
        # Create environment
        env = create_environment(episode_dir)
        
        # Create agent
        agent = DreamerPolicy(
            env,
            training=False,
            checkpoint_dir=checkpoint_dir,
            load_checkpoint=True,
            checkpoint_number=checkpoint
        )
        
        def policy_func(obs):
            return agent(obs)
        
        policy_func.__self__ = agent
        
        # Run episode with MIXTAPE format
        total_reward, json_path = run_episode_mixtape(
            env, 
            policy_func, 
            output_dir=episode_dir,  # CSV/video files go here
            episode_id=episode,
            checkpoint_number=checkpoint,
            total_training_steps=checkpoint_number,
            include_images=include_images
        )
        
        results.append((total_reward, json_path))
        print(f"Completed episode {episode}/{num_episodes} with reward: {total_reward:.2f}")
    
    print(f"\nAll MIXTAPE JSON files saved in: {MIXTAPE_RESULTS_DIR}/checkpoint_{checkpoint}/")
    return results

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run evaluation episodes with DreamerV2 and save in MIXTAPE format")
    parser.add_argument("--checkpoint-number", type=int, default=DEFAULT_CHECKPOINT,
                       help="Checkpoint folder number (e.g., 270000 for ckpt_270000)")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES,
                       help="Number of episodes to run")
    parser.add_argument("--include-images", action="store_true",
                       help="Include base64-encoded images in the output")
    
    args = parser.parse_args()
    
    # Run evaluation
    checkpoint_number = args.checkpoint_number
    latest_checkpoint = find_latest_checkpoint(f"{BASE_CHECKPOINT_DIR}{checkpoint_number}")

    if latest_checkpoint:
        print(f"Running evaluation with checkpoint {latest_checkpoint} from folder ckpt_{checkpoint_number}")
        results = run_with_dreamer_mixtape(
            checkpoint_number=checkpoint_number,
            checkpoint=latest_checkpoint,
            num_episodes=args.num_episodes,
            include_images=args.include_images
        )
        print(f"Evaluation complete! MIXTAPE JSON files saved in {MIXTAPE_RESULTS_DIR}/checkpoint_{latest_checkpoint}/")
    else:
        print(f"No checkpoints found in {BASE_CHECKPOINT_DIR}{checkpoint_number}. Train a model first.")
