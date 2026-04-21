"""
Pixel observation wrapper for MuJoCo Reacher (Gymnasium).

This wrapper converts state observations into image observations by:
- Rendering RGB frames from the simulator
- Resizing frames to a target size (default 84x84)
- Optionally converting to grayscale
- Returning channel-first tensors for Stable-Baselines3
- Optionally stacking the last N frames (default 4)
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
    Create Reacher-v5 if available, otherwise Reacher-v5.
    """
    for env_id in ("Reacher-v5", "Reacher-v4"):
        try:
            env = gym.make(env_id, render_mode=render_mode)
            return env, env_id
        except Exception:
            continue
    raise RuntimeError("Could not create Reacher-v4 or Reacher-v5.")


class ReacherPixelWrapper(gym.Wrapper):
    """
    Convert MuJoCo Reacher observations to image observations.

    Args:
        env: A Gymnasium environment created with render_mode="rgb_array".
        image_size: (width, height) target image size.
        grayscale: If True, output single-channel images.
        frame_stack: Number of consecutive frames to stack.

    Output shape:
    - RGB + stack=1:  (3, H, W)
    - RGB + stack=4:  (12, H, W)
    - Gray + stack=1: (1, H, W)
    - Gray + stack=4: (4, H, W)
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
            raise ValueError(
                "Base env must use render_mode='rgb_array' so frames can be rendered."
            )

        self.width, self.height = image_size
        self.grayscale = grayscale
        self.frame_stack = frame_stack
        self.channels_per_frame = 1 if self.grayscale else 3
        self.total_channels = self.channels_per_frame * self.frame_stack
        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=self.frame_stack)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.total_channels, self.height, self.width),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        """
        Reset env, render first frame, and initialize frame stack.
        """
        _, info = self.env.reset(**kwargs)
        frame = self._render_frame()
        processed = self._preprocess_frame(frame)

        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed)

        return self._stacked_observation(), info

    def step(self, action):
        """
        Step env and return stacked image observation.
        """
        _, reward, terminated, truncated, info = self.env.step(action)
        frame = self._render_frame()
        processed = self._preprocess_frame(frame)
        self.frame_buffer.append(processed)

        return self._stacked_observation(), reward, terminated, truncated, info

    def _render_frame(self) -> np.ndarray:
        """
        Render one RGB frame from the base environment.
        """
        frame = self.env.render()
        if frame is None:
            raise RuntimeError(
                "env.render() returned None. Check render_mode='rgb_array'."
            )
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected np.ndarray from render(), got {type(frame)}.")
        return frame

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize and optionally grayscale a raw RGB frame.
        Returns channel-first uint8 frame.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected RGB frame with shape (H, W, 3), got {frame.shape}."
            )

        resized = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )

        if self.grayscale:
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)  # (H, W)
            processed = np.expand_dims(gray, axis=0)  # (1, H, W)
        else:
            processed = np.transpose(resized, (2, 0, 1))  # (3, H, W)

        return processed.astype(np.uint8)

    def _stacked_observation(self) -> np.ndarray:
        """
        Concatenate all frames along the channel axis.
        """
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
            grayscale=False,
            frame_stack=4,
        )

        observation, _ = env.reset(seed=42)
        print(f"Using environment: {env_id}")
        print(f"Wrapped observation shape: {observation.shape}")
        print(f"Wrapped observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        print(f"After one step, obs shape: {observation.shape}")
        print(
            f"Step outputs -> reward: {reward}, terminated: {terminated}, truncated: {truncated}"
        )
    finally:
        if env is not None:
            env.close()
            print("Wrapped environment closed.")

