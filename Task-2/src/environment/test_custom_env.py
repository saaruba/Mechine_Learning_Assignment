"""
Minimal test script for the custom SimpleManipulatorEnv.

Run with:
python -m environment.test_custom_env
"""

from __future__ import annotations

import time

from environment.simple_manipulator_env import SimpleManipulatorEnv


def main() -> None:
    env = None
    try:
        env = SimpleManipulatorEnv(render_mode="human")
        obs, info = env.reset(seed=42)
        print(f"Initial observation shape: {obs.shape}")
        print(f"Reset info keys: {list(info.keys())}")

        for step in range(300):
            action = env.action_space.sample()
            _, reward, terminated, truncated, step_info = env.step(action)
            env.render()

            if step % 25 == 0:
                print(
                    f"Step {step:03d} | reward={reward:.4f} | "
                    f"distance={step_info['distance']:.4f}"
                )

            if terminated:
                print("Reached the ball!")
                break

            if truncated:
                print("Episode truncated.")
                break

            time.sleep(0.02)
        else:
            print("Did not reach the ball in this short random rollout.")
    finally:
        if env is not None:
            env.close()
            print("Environment closed.")


if __name__ == "__main__":
    main()
