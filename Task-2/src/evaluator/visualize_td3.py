"""
Visualize a trained TD3 model on MuJoCo Reacher.

What this script does:
1. Creates Reacher-v5 (fallback Reacher-v4) in rgb_array mode.
2. Wraps with ReacherPixelWrapper (84x84, grayscale, frame_stack=4).
3. Uses gym.wrappers.HumanRendering to show the scene in a window.
4. Loads a TD3 model and runs evaluation episodes deterministically.

How to run from Task-2:
python src/evaluator/visualize_td3.py --model-path results/td3_seed7/models/td3_reacher_pixels_seed_7.zip --episodes 5 --seed 7 --sleep 0.05

Output files:
- None (console output + visualization window only).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.utils import set_random_seed


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from environment.reacher_pixel_wrapper import (
        ReacherPixelWrapper,
        make_reacher_env_with_fallback,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import ReacherPixelWrapper from src/environment/reacher_pixel_wrapper.py."
    ) from exc


def build_visual_env(seed: int) -> tuple[gym.Env, str]:
    base_env, env_name = make_reacher_env_with_fallback(render_mode="rgb_array")
    pixel_env = ReacherPixelWrapper(
        base_env,
        image_size=(84, 84),
        grayscale=True,
        frame_stack=4,
    )
    env = gym.wrappers.HumanRendering(pixel_env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env, env_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize TD3 on pixel-based MuJoCo Reacher.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=0.05)
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
    print("TD3 Reacher Visualization")
    print(f"Model path : {model_path}")
    print(f"Episodes   : {args.episodes}")
    print(f"Seed       : {args.seed}")
    print(f"Step sleep : {args.sleep}")
    print("=" * 70)

    set_random_seed(args.seed)
    model = TD3.load(str(model_path))
    print("[INFO] TD3 model loaded.")

    env: gym.Env | None = None
    try:
        env, env_name = build_visual_env(seed=args.seed)
        print(f"[INFO] Environment selected: {env_name}")
        print(f"[INFO] Observation space: {env.observation_space}")
        print(f"[INFO] Action space: {env.action_space}")

        for episode_idx in range(1, args.episodes + 1):
            obs, _ = env.reset(seed=args.seed + episode_idx)
            terminated = False
            truncated = False
            episode_reward = 0.0
            step_count = 0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                step_count += 1
                if args.sleep > 0:
                    time.sleep(args.sleep)

            print(
                f"[EPISODE {episode_idx}/{args.episodes}] reward={episode_reward:.4f}, "
                f"steps={step_count}, terminated={terminated}, truncated={truncated}"
            )
    finally:
        if env is not None:
            env.close()
            print("[INFO] Environment closed.")


if __name__ == "__main__":
    main()
