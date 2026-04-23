import gymnasium as gym

env = gym.make("MuJoCoEnv", model_path="assets/simple_manipulator.xml", render_mode="human")

obs, _ = env.reset()

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()

env.close()