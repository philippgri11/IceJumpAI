import gym
import numpy as np
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2", render_mode="human")

env.reset()

print("Sample Movement: ", env.action_space.sample())

print("ObservationSampleSpaceShape: ", env.observation_space.shape)
print("ObservationSampleSpaceSample: ", env.observation_space.sample())

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

episodes = 1

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

env.close()