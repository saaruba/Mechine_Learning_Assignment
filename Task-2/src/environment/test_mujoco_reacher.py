"""
Minimal sanity check for MuJoCo Reacher in Gymnasium.

What this script does:
1. Tries to create `Reacher-v4` first, then falls back to `Reacher-v5`.
2. Resets the environment and prints observation/action space details.
3. Takes one random action step and prints step outputs.
4. Renders one RGB frame and prints its shape.
5. Closes the environment cleanly.
"""

from __future__ import annotations

import gymnasium as gym


ENV_CANDIDATES = ("Reacher-v5", "Reacher-v4")


def create_reacher_env(render_mode: str = "human") -> tuple[gym.Env, str]:
    """
    Create a MuJoCo Reacher environment with a simple version fallback.

    Returns:
        (env, env_id): The environment instance and the selected environment ID.
    """
    last_error: Exception | None = None

    for env_id in ENV_CANDIDATES:
        try:
            env = gym.make(env_id, render_mode=render_mode)
            return env, env_id
        except Exception as exc:  # noqa: BLE001 - helpful for environment fallback
            last_error = exc

    raise RuntimeError(
        "Could not create Reacher environment. Tried Reacher-v4 and Reacher-v5."
    ) from last_error


def main() -> None:
    env: gym.Env | None = None

    try:
        env, env_id = create_reacher_env(render_mode="human")
        print(f"Using environment: {env_id}")

        observation, info = env.reset(seed=42)
        obs_shape = getattr(observation, "shape", None)
        print(f"Observation type: {type(observation)}")
        print(f"Observation shape: {obs_shape}")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        print(f"Reset info keys: {list(info.keys())}")

        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")

        frame = env.render()
        if frame is None:
            raise RuntimeError(
                "env.render() returned None. Ensure render_mode='human'."
            )
        print(f"Rendered frame shape: {frame.shape}")

    except Exception as exc:  # noqa: BLE001 - clearer debug for setup issues
        print(f"[ERROR] Reacher sanity check failed: {exc}")
        raise
    finally:
        if env is not None:
            env.close()
            print("Environment closed.")


if __name__ == "__main__":
    main()

