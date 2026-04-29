"""
Reacher pixel observation wrapper for Stable-Baselines3.

What this script does:
1. Creates a Gymnasium MuJoCo Reacher env with fallback:
   Reacher-v5 first, then Reacher-v4.
2. Wraps the env to produce image observations suitable for CNN policies:
   - resize to 84x84
   - optional grayscale
   - channel-first
   - optional frame stacking

How to run from Task-2 folder:
python src/environment/reacher_pixel_wrapper.py

Output files:
- None (console sanity output only).
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


def make_reacher_env_with_fallback(render_mode: str = "rgb_array") -> tuple[gym.Env, str]:
    """
    Create Reacher-v5 if available, otherwise Reacher-v4.
    """
    for env_id in ("Reacher-v5", "Reacher-v4"):
        try:
            env = gym.make(env_id, render_mode=render_mode)
            return env, env_id
        except Exception:
            continue
    raise RuntimeError("Could not create Reacher-v5 or Reacher-v4.")


class ReacherPixelWrapper(gym.Wrapper):
    """
    Convert Reacher observations to pixel-based observations.
    """

    def __init__(
        self,
        env: gym.Env,
        image_size: tuple[int, int] = (84, 84),
        grayscale: bool = False,
        frame_stack: int = 4,
    ) -> None:
        super().__init__(env)

        if frame_stack < 1:
            raise ValueError("frame_stack must be >= 1.")

        render_mode = getattr(env, "render_mode", None)
        if render_mode != "rgb_array":
            raise ValueError("Base env must use render_mode='rgb_array'.")

        self.width, self.height = image_size
        self.grayscale = grayscale
        self.frame_stack = frame_stack
        self.channels_per_frame = 1 if grayscale else 3
        self.total_channels = self.channels_per_frame * frame_stack
        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=frame_stack)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.total_channels, self.height, self.width),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        frame = self._render_frame()
        processed = self._preprocess_frame(frame)

        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed)

        return self._stacked_observation(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        frame = self._render_frame()
        processed = self._preprocess_frame(frame)
        self.frame_buffer.append(processed)
        return self._stacked_observation(), reward, terminated, truncated, info

    def _render_frame(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("env.render() returned None. Check render_mode='rgb_array'.")
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected np.ndarray frame, got {type(frame)}.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected frame shape (H, W, 3), got {frame.shape}.")
        return frame

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)  # (H, W)
            processed = np.expand_dims(gray, axis=0)  # (1, H, W)
        else:
            processed = np.transpose(resized, (2, 0, 1))  # (3, H, W)
        return processed.astype(np.uint8)

    def _stacked_observation(self) -> np.ndarray:
        if len(self.frame_buffer) == 0:
            raise RuntimeError("Frame buffer is empty. Call reset() first.")
        return np.concatenate(list(self.frame_buffer), axis=0)


if __name__ == "__main__":
    env: gym.Env | None = None
    try:
        base_env, env_id = make_reacher_env_with_fallback(render_mode="rgb_array")
        env = ReacherPixelWrapper(
            base_env,
            image_size=(84, 84),
            grayscale=True,
            frame_stack=4,
        )
        obs, _ = env.reset(seed=42)
        print(f"Using env: {env_id}")
        print(f"Wrapped observation shape: {obs.shape}")  # expected (4, 84, 84)
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
    finally:
        if env is not None:
            env.close()
