from re import S
import torch
from random import randint
from tqdm import trange
from ChessEnv import * 

class ReplayMemory ():
    def __init__(self, max_len) -> None:
        self.max_len = max_len 
        self.states = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def addToTensor (self, origTensor, addTensor):
        addTensor = addTensor.unqueeze(0)
        if origTensor.size(0) == 0: 
            return addTensor
        else:
            return torch.concat((origTensor, addTensor), dim=0)

    def add (self, state, next_state, reward, done):     
        self.states.append(state)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

        self.states = self.states[-self.max_len:]
        self.next_states = self.next_states[-self.max_len:]
        self.rewards = self.rewards[-self.max_len:]
        self.dones = self.dones[-self.max_len:]

    def get_batch (self, batch_size):
        states_return = torch.tensor([]).to(self.device)
        next_states_return = torch.tensor([]).to(self.device)
        rewards_return = torch.tensor([]).to(self.device)
        dones_return = []

        for _ in range(batch_size):
            random_index = randint(0, len(self.states))
            
            states_return = self.addToTensor(states_return, self.states[random_index])
            next_states_return = self.addToTensor(next_states_return, self.next_states[random_index])
            rewards_return = self.addToTensor(rewards_return, self.rewards[random_index])
            dones_return.append(self.dones[random_index])
    
        return states_return, next_states_return, rewards_return, dones_return

class DQN:
    def __init__(self, model, max_len, epsilon) -> None:
        self.model = model 
        self.env = ChessEnv()
        self.replay_mem = ReplayMemory(max_len=max_len)
        self.epsilon = epsilon

    def train (self, num_episodes, use_tqdm=False):
        if use_tqdm: progress_bar = trange(num_episodes)
        else: progress_bar = range(num_episodes)

        for episode in range(progress_bar):
            state, len_moves = self.env.reset()     
            done = False

            learn_grand_master = randint(0, 2)
            if learn_grand_master == 0: 
                # 1/3 of the chance to draw from master database
                self.replay_mem.add(*self.env.get_game_from_master_db())
            else:
                # Go to the actual environment
                while not done: 
                    rand_num = randint(0, 3) 
                    if rand_num < self.epsilon * 100: 
                        # Do random action
                        move = randint(0, len_moves-1)
                    else:
                        move = self.model(state)
                    
                    next_state, reward, done = self.env.step(move)
                    self.replay_mem.add(state, next_state, reward, done) 
                    state = next_state

            # HI ODION
            
            