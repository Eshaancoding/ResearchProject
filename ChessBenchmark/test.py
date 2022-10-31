import gym
env = gym.make("Pendulum-v1", render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()