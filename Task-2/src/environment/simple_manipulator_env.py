"""
Custom Gymnasium MuJoCo environment for a simple 2-joint manipulator.

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
            width=320,
            height=240,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.ee_body_id = self._find_body_id(
            ["ee", "end_effector", "fingertip", "tip"]
        )
        self.ball_body_id = self._find_body_id(
            ["ball", "target_ball", "target", "ball_target"]
        )
        self.ball_z = float(self.model.body_pos[self.ball_body_id, 2])

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
        Reset joints and randomize ball position in XY plane.
        """
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        if qpos.shape[0] >= 2:
            qpos[:2] = self.np_random.uniform(low=-0.1, high=0.1, size=2)
        if qvel.shape[0] >= 2:
            qvel[:2] = 0.0

        self.set_state(qpos, qvel)

        ball_xy = self.np_random.uniform(
            low=np.array([0.15, -0.25]),
            high=np.array([0.35, 0.25]),
        )
        self.model.body_pos[self.ball_body_id, 0] = float(ball_xy[0])
        self.model.body_pos[self.ball_body_id, 1] = float(ball_xy[1])
        self.model.body_pos[self.ball_body_id, 2] = self.ball_z

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.do_simulation(action, self.frame_skip)

        ee_xy = self.data.xpos[self.ee_body_id][:2].copy()
        ball_xy = self.data.xpos[self.ball_body_id][:2].copy()
        distance = float(np.linalg.norm(ee_xy - ball_xy))

        reward = -distance
        terminated = distance < 0.05
        truncated = False

        observation = self._get_obs()
        info = {
            "distance": distance,
            "ee_pos": ee_xy.tolist(),
            "ball_pos": ball_xy.tolist(),
        }
        return observation, reward, terminated, truncated, info
