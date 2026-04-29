"""
Train TD3 on MuJoCo Reacher using pixel observations.

What this script does:
1. Creates Reacher-v5 (fallback Reacher-v4) in rgb_array mode.
2. Wraps env with ReacherPixelWrapper for image observations.
3. Trains TD3 with CnnPolicy and memory-safe replay settings.
4. Saves model, logs, and metadata.json.

How to run from Task-2:
python src/training/train_td3.py --timesteps 20000 --seed 7 --save-dir results/td3_seed7

Output files:
- <save-dir>/models/td3_reacher_pixels_seed_<seed>.zip
- <save-dir>/logs/monitor_seed_<seed>.csv
- <save-dir>/logs/tensorboard/*
- <save-dir>/metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv


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


def build_wrapped_env(seed: int, log_dir: Path) -> tuple[DummyVecEnv, str]:
    env_name_holder: dict[str, str] = {}

    def _make_env():
        base_env, env_name = make_reacher_env_with_fallback(render_mode="rgb_array")
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
    return vec_env, env_name_holder.get("name", "unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TD3 on pixel-based MuJoCo Reacher.")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    default_save_dir = project_root / "results" / "td3_debug"
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else default_save_dir

    model_dir = save_dir / "models"
    log_dir = save_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Starting TD3 training (Reacher pixels)")
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

        model = TD3(
            policy="CnnPolicy",
            env=vec_env,
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(tensorboard_dir),
            device="auto",
            buffer_size=10_000,
            learning_starts=1_000,
            batch_size=64,
            learning_rate=3e-4,
        )

        print("[INFO] Training started...")
        t0 = time.perf_counter()
        model.learn(total_timesteps=args.timesteps)
        training_time_sec = time.perf_counter() - t0
        print(f"[INFO] Training completed in {training_time_sec:.2f} sec.")

        model_name = f"td3_reacher_pixels_seed_{args.seed}"
        model_base = model_dir / model_name
        model.save(str(model_base))
        model_path = model_base.with_suffix(".zip")
        print(f"[INFO] Model saved to: {model_path}")

        metadata = {
            "algorithm": "TD3",
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
    finally:
        if vec_env is not None:
            vec_env.close()
            print("[INFO] Environment closed.")


if __name__ == "__main__":
    main()
