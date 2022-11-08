import sys; sys.path.append("..\\")
import gym
from RL.DQN import *
from torch import nn
from random import randint

class WrapperModel (nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    # Ignore extra state
    def forward (self, x, extra_state):
        return self.model(x)

model = WrapperModel()

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

# Sadly, we have to create a wrapper for the gym class in order to support the extra_state functionality
class WrapperGym ():
    def __init__(self) -> None:
        self.env = gym.make("Acrobot-v1")
        self.action_space = self.env.action_space

    def reset (self):
        state = self.env.reset()
        state = torch.from_numpy(state)
        return state, torch.tensor([]), None

    def step (self, action):
        observation, reward, terminated, _ = self.env.step(action) 

        observation = torch.from_numpy(observation)
        
        done = False
        if terminated: 
            done = True
        
        return observation, torch.tensor([]), reward, done, None, None

env = WrapperGym()

trainer.train(
    env=env,
    num_episodes=100_000,
    use_tqdm=True,
)