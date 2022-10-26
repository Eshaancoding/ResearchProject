import gym
from DQN import *
from torch import nn
test = False

# Declare model
model = nn.Sequential(
    nn.Linear(4, 15),
    nn.ReLU(),
    nn.Linear(15, 30),
    nn.ReLU(),
    nn.Linear(30, 2),
)

# Declare trainer
trainer = DQN(
    model=model,
    replay_mem_max_len=1_000,
    epsilon=0.15,
    batch_size=16,
    gamma=0.98,
    lr=0.01,
    model_path=None,
    should_load_from_path=False, 
    save_per_epi=100,
    update_target_model_per_epi=1000,
)

# Make environment
class TestActionSpace ():
    def sample (self):
        return randint(0, 4)

class CustomEnv ():
    def __init__(self) -> None:
        self.itr = 0
        self.action_space = TestActionSpace()

    def reset (self):
        self.itr = 0
        self.random_integer = randint(0, 4)
        self.x = [0,0,0,0,0]
        self.x[self.random_integer] = 1

        return self.x
        
    def step (self, action):
        self.itr += 1
        reward = 4 - abs(self.random_integer - action)

        self.random_integer = randint(0, 4)
        self.x = [0,0,0,0,0]
        self.x[self.random_integer] = 1

        is_done = False
        if self.itr == 100:
            is_done = True
            self.itr = 0
        
        return self.x, reward, is_done, ""

# env = CustomEnv()
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Train or test model
if test:
    trainer.test(env, render=False)
else:
    trainer.train(
        env=env,
        num_episodes=10_000,
        # num_episodes=1,
        use_database=False, # Only if we are using chess database
        use_tqdm=True,
        render=True
    )