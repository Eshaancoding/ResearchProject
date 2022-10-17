import gym
from DQN import *
from torch import nn
import torch

# Declare model
model = nn.Sequential(
    nn.Linear(2, 100),
    nn.ReLU(),
    nn.Linear(100, 3),
)

# Declare trainer
trainer = DQN(
    model=model,
    replay_mem_max_len=500,
    epsilon=0.1,
    batch_size=64,
    gamma=0.99,
    lr=0.01,
    model_path="MountainCarDQNTest.pth",
    log_per_epi=500,
    update_target_model_per_epi=15,
    max_test_itr=100
)

# Make environment
env = gym.make('MountainCar-v0')

# Train model
#trainer.train(
        #    env=env,
        #num_episodes=5_000,
        #use_database=False, # Only if we are using chess database
        #use_tqdm=True,
        #render=False
        #)
trainer.test(env, 10000, render=True)
