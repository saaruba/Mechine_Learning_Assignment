"""
Visualize a trained TD3 agent on MuJoCo Reacher.

This script loads a trained TD3 model and runs evaluation episodes while
showing the robot movement in a human-visible window.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.utils import set_random_seed


# -----------------------------------------------------------------------------
# Handle imports safely for this folder layout:
# Task-2/src/evaluator/visualize_td3.py
# Task-2/src/environment/reacher_pixel_wrapper.py
# -----------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from environment.manipulator_pixel_wrapper import ManipulatorPixelWrapper
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import ReacherPixelWrapper from src/environment/reacher_pixel_wrapper.py. "
        "Please check your folder structure."
    ) from exc


def create_base_reacher_env(render_mode: str) -> tuple[gym.Env, str]:
    """
    Create Reacher-v5 first, with fallback to Reacher-v4.
    """
    candidates = ("Reacher-v5", "Reacher-v4")
    last_error: Exception | None = None

    for env_name in candidates:
        try:
            env = gym.make(env_name, render_mode=render_mode)
            return env, env_name
        except Exception as exc:  # noqa: BLE001 - useful for fallback behavior
            last_error = exc

    raise RuntimeError("Failed to create Reacher-v5 and Reacher-v4.") from last_error


def build_visual_env(seed: int) -> tuple[gym.Env, str]:
    """
    Build the same wrapped observation setup used in training:
    grayscale=True, frame_stack=4, image_size=84x84.

    Preferred path:
    - Base env with render_mode="human" + ReacherPixelWrapper

    Compatibility fallback (if wrapper needs rgb frames from render()):
    - Base env with render_mode="rgb_array" + ReacherPixelWrapper + HumanRendering
    """
    try:
        base_env, env_name = create_base_reacher_env(render_mode="human")
        env = ManipulatorPixelWrapper(
            base_env,
            image_size=(84, 84),
            grayscale=True,
            frame_stack=4,
        )
        env.reset(seed=seed)
        env.action_space.seed(seed)
        print(f"[INFO] Using direct human render path with env: {env_name}")
        return env, env_name
    except Exception as exc:  # noqa: BLE001 - fallback to compatibility rendering
        print("[WARN] Direct render_mode='human' path failed with wrapper.")
        print(f"[WARN] Reason: {exc}")
        print("[INFO] Falling back to HumanRendering wrapper compatibility mode.")

    base_env, env_name = create_base_reacher_env(render_mode="rgb_array")
    pixel_env = ManipulatorPixelWrapper(
        base_env,
        image_size=(84, 84),
        grayscale=True,
        frame_stack=4,
    )
    env = gym.wrappers.HumanRendering(pixel_env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    print(f"[INFO] Using HumanRendering compatibility path with env: {env_name}")
    return env, env_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained TD3 model on MuJoCo Reacher."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained TD3 model zip file.",
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
        help="Sleep time in seconds after each step (default: 0.02).",
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
    print("TD3 Visualization Run")
    print(f"Model path : {model_path}")
    print(f"Episodes   : {args.episodes}")
    print(f"Seed       : {args.seed}")
    print(f"Step sleep : {args.sleep} sec")
    print("=" * 70)

    set_random_seed(args.seed)
    model = TD3.load(str(model_path))
    print("[INFO] TD3 model loaded successfully.")

    env: gym.Env | None = None
    try:
        env, env_name = build_visual_env(seed=args.seed)
        print(f"[INFO] Environment selected: {env_name}")
        print(f"[INFO] Wrapped observation space: {env.observation_space}")
        print(f"[INFO] Action space: {env.action_space}")

        for episode_idx in range(1, args.episodes + 1):
            observation, _ = env.reset(seed=args.seed + episode_idx)
            terminated = False
            truncated = False
            episode_reward = 0.0
            step_count = 0

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
