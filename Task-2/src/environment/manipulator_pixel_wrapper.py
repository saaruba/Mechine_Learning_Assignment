"""
Pixel observation wrapper for the custom manipulator environment.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ManipulatorPixelWrapper(gym.Wrapper):
    """
    Wrap an existing environment and expose image observations.
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
        # Ignore numeric observation from base env, return processed pixels instead.
        _, info = self.env.reset(**kwargs)
        frame = self._render_frame()
        processed = self._preprocess_frame(frame)

        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed)

        return self._stacked_observation(), info

    def step(self, action):
        # Ignore numeric observation from base env, return processed pixels instead.
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

        # Some env setups may return None from render(); try renderer fallback.
        if frame is None and hasattr(self.env, "mujoco_renderer"):
            frame = self.env.mujoco_renderer.render(render_mode="rgb_array")

        if frame is None:
            raise RuntimeError(
                "Could not render RGB frame. Use base env with render_mode='rgb_array'."
            )
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected np.ndarray frame, got {type(frame)}.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected frame shape (H, W, 3), got {frame.shape}.")
        return frame

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            frame,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )

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
    from environment.simple_manipulator_env import SimpleManipulatorEnv

    env = None
    try:
        base_env = SimpleManipulatorEnv(render_mode="rgb_array")
        env = ManipulatorPixelWrapper(
            base_env,
            image_size=(84, 84),
            grayscale=True,
            frame_stack=4,
        )

        obs, _ = env.reset(seed=42)
        print(f"Wrapped observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")

        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"After one step -> obs shape: {obs.shape}")
        print(
            f"Step outputs -> reward: {reward:.4f}, "
            f"terminated: {terminated}, truncated: {truncated}"
        )
    finally:
        if env is not None:
            env.close()
            print("Wrapped environment closed.")
