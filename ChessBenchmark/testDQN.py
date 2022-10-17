import gym
from DQN import *
from torch import nn
import torch

# Declare model
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 3),
)

# Declare trainer
trainer = DQN(
    model=model,
    replay_mem_max_len=10_000,
    epsilon=0.99,
    batch_size=32,
    gamma=0.99,
    lr=0.01,
    model_path=None,
    log_per_epi=None,
    update_target_model_per_epi=15,
    max_test_itr=100
)

# Make environment
env = gym.make('MountainCar-v0')

# Train model
trainer.train(
    env=env,
    num_episodes=100_000,
    use_database=False, # Only if we are using chess database
    use_tqdm=True,
    render=False
)