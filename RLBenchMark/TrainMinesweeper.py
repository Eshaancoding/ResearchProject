from msilib.schema import CustomAction
import gym
from RL.DQN import *
from torch import nn
from random import randint

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)

# Declare trainer
trainer = DQN(
    model=model,

    replay_mem_max_len=1_000,
    batch_size=32,
    gamma=0.95,
    lr=0.001,

    update_target_model_per_epi=10,
    epsilon=1,
    epsilon_decay=0.995,
    epsilon_min=0.1,
    test_per_epi=1000,

    model_path=None,
    should_load_from_path=False, 
    save_per_epi=100,
)

env = gym.make("CartPole-v1")

trainer.train(
    env=env,
    num_episodes=100_000,
    use_database=False, # Only if we are using chess database
    use_tqdm=True,
    render=True
)