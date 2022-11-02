from ChessEnv import *

env = ChessEnv("")
state, possible_move, _ = env.reset()
env.step(1)