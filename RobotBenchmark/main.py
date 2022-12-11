from PPO import *

agent = PPO(None, "Walker-v0", 4, 6)
agent.test(100)