import gym
from DQN import *
from torch import nn
test = False

# Declare model
model = nn.Sequential(
    nn.Linear(4, 100),
    nn.ReLU(),
    nn.Linear(100, 2),
)

# Declare trainer
trainer = DQN(
    model=model,
    replay_mem_max_len=1_000,
    epsilon=0.5,
    batch_size=16,
    gamma=0.98,
    lr=0.01,
    model_path="CartpoleV1.pth",
    save_per_epi=100,
    update_target_model_per_epi=15,
)

# Make environment
env = gym.make('CartPole-v1')

# Train or test model
if test:
    trainer.test(env, 10000, render=True)
else:
    trainer.train(
        env=env,
        num_episodes=5_000,
        use_database=False, # Only if we are using chess database
        use_tqdm=True,
        render=False
    )