from PPO import *
from Agent import *

# Parameters
vnn_version = 1
device = "cuda"
min_size = 4
max_size = 6

policy = Policy(vnn_version, device)
value = Value(device)

agent = Trainer(policy, value, "Walker-v0", device, min_size, max_size)
agent.test(render=True, return_training_vars=True)