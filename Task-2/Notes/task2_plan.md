# Task 2 Plan: Robot Learning with MuJoCo

## Objective
Task 2 focuses on deep reinforcement learning for robot control using MuJoCo, with image-based observations from the simulator. The goal is to build a clean and reproducible pipeline suitable for MSc-level reporting.

## Why Reacher
`Reacher` is a standard continuous-control MuJoCo task with manageable complexity. It is appropriate for comparing DRL algorithms because:
- it has meaningful robot dynamics,
- it supports fast experimentation,
- it is widely used in benchmark settings.

## Agents to Compare
The project will train and compare three DRL agents:
- PPO
- SAC
- TD3

## Planned Pipeline
1. Environment sanity check (`Reacher-v4`, fallback to `Reacher-v5`).
2. Pixel wrapper for image observations (resize, grayscale option, frame stacking).
3. Training scripts for PPO/SAC/TD3 with consistent configs.
4. Evaluation scripts for trained models.
5. Multi-seed experiments (at least 3 seeds per agent).
6. Result aggregation into tables and plots for report-ready analysis.

## Reported Metrics
The final comparison will include:
- average episode reward,
- average steps per episode,
- training time,
- test time.

## use to install the requireed liblaryes with this command 
   
   pip install numpy matplotlib opencv-python gymnasium mujoco stable-baselines3 pandas pillow imageio pygame tensorboard