"""
Debug SAC training script for MuJoCo Reacher with image observations.

This script:
1. Creates Reacher-v5 (falls back to Reacher-v4) with render_mode="rgb_array".
2. Wraps the environment using the existing pixel wrapper.
3. Trains a SAC agent with a CNN policy for a short debug run.
4. Saves the model, logs, and a small metadata JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv


# -----------------------------------------------------------------------------
# Handle imports safely for this folder layout:
# Task-2/src/training/train_sac.py
# Task-2/src/environment/reacher_pixel_wrapper.py
# -----------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from environment.reacher_pixel_wrapper import ReacherPixelWrapper
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import ReacherPixelWrapper from src/environment/reacher_pixel_wrapper.py. "
        "Please check your folder structure."
    ) from exc


def create_base_reacher_env(render_mode: str = "rgb_array") -> tuple[gym.Env, str]:
    """
    Create Reacher-v5 first, with fallback to Reacher-v4.
    """
    candidates = ("Reacher-v5", "Reacher-v4")
    last_error: Exception | None = None

    for env_name in candidates:
        try:
            env = gym.make(env_name, render_mode=render_mode)
            return env, env_name
        except Exception as exc:  # noqa: BLE001 - useful fallback logging
            last_error = exc

    raise RuntimeError(
        "Failed to create Reacher-v5 and Reacher-v4. "
        "Please verify MuJoCo/Gymnasium installation."
    ) from last_error


def build_wrapped_env(seed: int, log_dir: Path) -> tuple[DummyVecEnv, str]:
    """
    Build a vectorized wrapped environment for Stable-Baselines3.
    """
    env_name_holder: dict[str, str] = {}

    def _make_env():
        base_env, env_name = create_base_reacher_env(render_mode="rgb_array")
        env_name_holder["name"] = env_name

        wrapped_env = ReacherPixelWrapper(
            base_env,
            image_size=(84, 84),
            grayscale=True,
            frame_stack=4,
        )

        monitor_file = log_dir / f"monitor_seed_{seed}.csv"
        wrapped_env = Monitor(wrapped_env, filename=str(monitor_file))
        wrapped_env.reset(seed=seed)
        wrapped_env.action_space.seed(seed)
        return wrapped_env

    vec_env = DummyVecEnv([_make_env])
    selected_env_name = env_name_holder.get("name", "unknown")
    return vec_env, selected_env_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on pixel-based MuJoCo Reacher.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000,
        help="Total training timesteps (default: 10000, short debug run).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save model/logs/metadata. Default: Task-2/results/sac_debug",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]  # Task-2
    default_save_dir = project_root / "results" / "sac_debug"
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else default_save_dir

    model_dir = save_dir / "models"
    log_dir = save_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Starting SAC debug training (image-based Reacher)")
    print(f"Timesteps : {args.timesteps}")
    print(f"Seed      : {args.seed}")
    print(f"Save dir  : {save_dir}")
    print("=" * 70)

    set_random_seed(args.seed)

    vec_env = None
    try:
        vec_env, env_name = build_wrapped_env(seed=args.seed, log_dir=log_dir)
        print(f"[INFO] Environment selected: {env_name}")
        print(f"[INFO] Observation space: {vec_env.observation_space}")
        print(f"[INFO] Action space: {vec_env.action_space}")

        tensorboard_dir = log_dir / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        model = SAC(
            policy="CnnPolicy",
            env=vec_env,
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(tensorboard_dir),
            device="auto",
            buffer_size=10000,
            learning_starts=1000,
            batch_size=64,
        )

        print("[INFO] Training started...")
        train_start = time.perf_counter()
        model.learn(total_timesteps=args.timesteps)
        training_time_sec = time.perf_counter() - train_start
        print(f"[INFO] Training completed in {training_time_sec:.2f} seconds.")

        model_name = f"sac_reacher_pixels_seed_{args.seed}"
        model_base_path = model_dir / model_name
        model.save(str(model_base_path))
        model_path = model_base_path.with_suffix(".zip")
        print(f"[INFO] Model saved to: {model_path}")

        metadata = {
            "algorithm": "SAC",
            "env_name": env_name,
            "seed": args.seed,
            "timesteps": args.timesteps,
            "render_mode": "rgb_array",
            "image_width": 84,
            "image_height": 84,
            "grayscale": True,
            "frame_stack": 4,
            "training_time_sec": round(training_time_sec, 4),
            "model_path": str(model_path),
            "log_dir": str(log_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        metadata_path = save_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"[INFO] Metadata saved to: {metadata_path}")

        print("[INFO] Debug SAC training run finished successfully.")
    finally:
        if vec_env is not None:
            vec_env.close()
            print("[INFO] Environment closed.")


if __name__ == "__main__":
    main()
