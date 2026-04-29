"""
Reacher sanity test with human rendering.

What this script does:
1. Creates MuJoCo Reacher with fallback:
   Reacher-v5 first, then Reacher-v4.
2. Resets the env and prints observation/action spaces.
3. Runs random actions for ~200 steps so movement is visible.

How to run:
- Run from Task-2/src:
  python -m environment.test_mujoco_reacher

Output files:
- None (console output only).
"""

from __future__ import annotations

import time

import gymnasium as gym


def make_reacher_env_human() -> tuple[gym.Env, str]:
    for env_id in ("Reacher-v5", "Reacher-v4"):
        try:
            env = gym.make(env_id, render_mode="human")
            return env, env_id
        except Exception:
            continue
    raise RuntimeError("Could not create Reacher-v5 or Reacher-v4.")


def main() -> None:
    env: gym.Env | None = None
    try:
        env, env_name = make_reacher_env_human()
        obs, info = env.reset(seed=42)

        print(f"Using environment: {env_name}")
        print(f"Observation shape: {getattr(obs, 'shape', None)}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print(f"Reset info keys: {list(info.keys())}")

        for step in range(200):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            env.render()

            if step % 25 == 0:
                print(
                    f"Step {step:03d} | reward={reward:.4f} | "
                    f"terminated={terminated} truncated={truncated}"
                )

            time.sleep(0.02)

            if terminated or truncated:
                obs, _ = env.reset()
                _ = obs
    finally:
        if env is not None:
            env.close()
            print("Environment closed.")


if __name__ == "__main__":
    main()
