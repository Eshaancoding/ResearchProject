from msilib.schema import CustomAction
import gym
from DQN import *
from torch import nn

model = nn.Sequential(
    nn.Linear(2, 15),
    nn.ReLU(),
    nn.Linear(15, 30),
    nn.ReLU(),
    nn.Linear(30, 3),
)

# Declare trainer
trainer = DQN(
    model=model,
    replay_mem_max_len=1_000,
    epsilon=0.85,
    batch_size=16,
    gamma=0.98,
    lr=0.01,
    model_path=None,
    should_load_from_path=False, 
    save_per_epi=100,
    update_target_model_per_epi=1000,
    itr_limit=1_000
)

env = gym.make("MountainCar-v0")

# Train or test model
trainer.train(
    env=env,
    num_episodes=10_000,
    use_database=False, # Only if we are using chess database
    use_tqdm=True,
    render=False
)