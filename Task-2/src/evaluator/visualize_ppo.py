"""
Visualize a trained PPO agent on the custom manipulator task.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed


# -----------------------------------------------------------------------------
# Handle imports safely for this folder layout:
# Task-2/src/evaluator/visualize_ppo.py
# Task-2/src/environment/simple_manipulator_env.py
# Task-2/src/environment/manipulator_pixel_wrapper.py
# -----------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from environment.simple_manipulator_env import SimpleManipulatorEnv
    from environment.manipulator_pixel_wrapper import ManipulatorPixelWrapper
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import manipulator environment modules from src/environment."
    ) from exc


def build_visual_env(seed: int) -> tuple[gym.Env, str]:
    """
    Build the same wrapped setup used in training and attach a human-render
    compatibility wrapper so the MuJoCo scene is visible.
    """
    env_name = "SimpleManipulatorEnv"

    base_env = SimpleManipulatorEnv(render_mode="rgb_array")
    pixel_env = ManipulatorPixelWrapper(
        base_env,
        image_size=(84, 84),
        grayscale=True,
        frame_stack=4,
    )

    # Compatibility visualization path:
    # keep training-time observation pipeline (rgb_array + pixel wrapper),
    # then mirror frames to an on-screen window.
    env = gym.wrappers.HumanRendering(pixel_env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env, env_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained PPO model on SimpleManipulatorEnv."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained PPO model zip file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to visualize (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.02,
        help="Sleep time in seconds between steps (default: 0.02).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1.")
    if args.sleep < 0:
        raise ValueError("--sleep must be >= 0.")

    print("=" * 70)
    print("PPO Visualization Run (SimpleManipulatorEnv)")
    print(f"Model path : {model_path}")
    print(f"Episodes   : {args.episodes}")
    print(f"Seed       : {args.seed}")
    print(f"Step sleep : {args.sleep} sec")
    print("=" * 70)

    set_random_seed(args.seed)
    model = PPO.load(str(model_path))
    print("[INFO] PPO model loaded successfully.")

    env: gym.Env | None = None
    try:
        env, env_name = build_visual_env(seed=args.seed)
        print(f"[INFO] Environment selected: {env_name}")
        print(f"[INFO] Observation space: {env.observation_space}")
        print(f"[INFO] Action space: {env.action_space}")

        for episode_idx in range(1, args.episodes + 1):
            observation, _ = env.reset(seed=args.seed + episode_idx)
            episode_reward = 0.0
            step_count = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                step_count += 1

                if args.sleep > 0:
                    time.sleep(args.sleep)

            print(
                f"[EPISODE {episode_idx}/{args.episodes}] "
                f"reward={episode_reward:.4f}, steps={step_count}, "
                f"terminated={terminated}, truncated={truncated}"
            )

        print("[INFO] Visualization finished.")
    finally:
        if env is not None:
            env.close()
            print("[INFO] Environment closed.")


if __name__ == "__main__":
    main()
