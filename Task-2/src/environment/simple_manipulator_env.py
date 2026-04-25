"""
Custom Gymnasium MuJoCo environment for a simple 3-joint manipulator.

Task:
- Move the end-effector toward a ball target.
- Reward is negative distance.
- Episode terminates when the end-effector is close enough.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv


class SimpleManipulatorEnv(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 100,
    }

    def __init__(self, render_mode: str = "rgb_array") -> None:
        xml_path = Path(__file__).resolve().parent / "assets" / "simple_manipulator.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        temp_model = mujoco.MjModel.from_xml_path(str(xml_path))
        obs_dim = int(temp_model.nq + temp_model.nv + 4)  # qpos, qvel, ee_xy, ball_xy

        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        super().__init__(
            model_path=str(xml_path),
            frame_skip=5,
            observation_space=observation_space,
            render_mode=render_mode,
            width=640,
            height=480,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self.ee_body_id = self._find_body_id(
            ["ee", "end_effector", "fingertip", "tip"]
        )
        self.ball_body_id = self._find_body_id(
            ["ball", "target_ball", "target", "ball_target"]
        )
        self.ball_z = float(self.model.body_pos[self.ball_body_id, 2])

        # Episode bookkeeping for truncation and reward shaping.
        self.max_episode_steps = 100
        self.current_step = 0
        self.prev_distance: float | None = None

    def _find_body_id(self, candidates: list[str]) -> int:
        for name in candidates:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                return int(body_id)

        available = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(self.model.nbody)
        ]
        raise ValueError(
            f"Could not find body. Tried {candidates}. Available: {available}"
        )

    def _get_obs(self) -> np.ndarray:
        """
        Return numeric state observation (no rendered images).
        """
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        ee_xy = self.data.xpos[self.ee_body_id][:2].copy()
        ball_xy = self.data.xpos[self.ball_body_id][:2].copy()
        obs = np.concatenate([qpos, qvel, ee_xy, ball_xy]).astype(np.float32)
        return obs

    def reset_model(self) -> np.ndarray:
        """
        Reset joints to a more useful reaching posture and randomize ball position.
        """
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        if qpos.shape[0] >= 3:
            # Start from an upright/useful pose instead of a near-flat configuration.
            # [base_yaw, shoulder, elbow]
            base_pose = np.array([0.0, 1.2, 0.9], dtype=np.float32)
            noise = self.np_random.uniform(low=-0.08, high=0.08, size=3)
            qpos[:3] = base_pose + noise
            qpos[:3] = np.clip(qpos[:3], self.model.jnt_range[:3, 0], self.model.jnt_range[:3, 1])
        if qvel.shape[0] >= 3:
            qvel[:3] = 0.0

        self.set_state(qpos, qvel)

        # Randomize target in reachable XY workspace.
        radius = float(self.np_random.uniform(0.20, 0.55))
        angle = float(self.np_random.uniform(-np.pi, np.pi))
        ball_xy = np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32)
        self.model.body_pos[self.ball_body_id, 0] = float(ball_xy[0])
        self.model.body_pos[self.ball_body_id, 1] = float(ball_xy[1])
        self.model.body_pos[self.ball_body_id, 2] = self.ball_z

        mujoco.mj_forward(self.model, self.data)

        # Reset episode counters and initialize previous distance for progress reward.
        self.current_step = 0
        ee_xy = self.data.xpos[self.ee_body_id][:2].copy()
        ball_xy = self.data.xpos[self.ball_body_id][:2].copy()
        self.prev_distance = float(np.linalg.norm(ee_xy - ball_xy))
        return self._get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.current_step += 1
        self.do_simulation(action, self.frame_skip)

        ee_xy = self.data.xpos[self.ee_body_id][:2].copy()
        ball_xy = self.data.xpos[self.ball_body_id][:2].copy()
        distance = float(np.linalg.norm(ee_xy - ball_xy))

        prev_distance = self.prev_distance if self.prev_distance is not None else distance
        progress = float(prev_distance - distance)

        action_penalty = float(np.sum(np.square(action)))
        reward = 5.0 * progress - 0.1 * distance - 0.01 * action_penalty
        if distance < 0.07:
            reward += 10.0

        terminated = distance < 0.07
        truncated = self.current_step >= self.max_episode_steps
        self.prev_distance = distance

        observation = self._get_obs()
        info = {
            "distance": distance,
            "ee_pos": ee_xy.tolist(),
            "ball_pos": ball_xy.tolist(),
            "current_step": self.current_step,
        }
        return observation, reward, terminated, truncated, info
