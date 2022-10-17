from re import S
import torch
from random import randint
from tqdm import trange
from ChessEnv import * 
from os.path import exists

class ReplayMemory ():
    def __init__(self, max_len) -> None:
        self.max_len = max_len 
        self.states = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def addToTensor (self, origTensor, addTensor):
        if isinstance(addTensor, bool):
            if addTensor: 
                addTensor = torch.tensor([1])
            else:
                addTensor = torch.tensor([0])
        else:
            addTensor = torch.tensor(addTensor).unqueeze(0)
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
        dones_return = torch.tensor([]).to(self.device)

        for _ in range(batch_size):
            random_index = randint(0, len(self.states))
            
            states_return = self.addToTensor(states_return, self.states[random_index])
            next_states_return = self.addToTensor(next_states_return, self.next_states[random_index])
            rewards_return = self.addToTensor(rewards_return, self.rewards[random_index])
            dones_return = self.addToTensor(dones_return, self.dones[random_index])
    
        return states_return, next_states_return, rewards_return, dones_return

class DQN:
    def __init__(self, model, max_len, epsilon, batch_size, gamma, lr, model_path, log_per_itr) -> None:
        self.model = model # The model will act as a value function but will output the value of each output in the output layer of the neural network, thus acting as a q-value function
        self.env = ChessEnv()
        self.replay_mem = ReplayMemory(max_len=max_len)
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.gamma = gamma
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.mse_loss = torch.nn.MSELoss()

        # Device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model saving
        self.model_path = model_path
        self.log_per_itr = log_per_itr

        # Check if model path exists, and if so load the model
        if exists(model_path):
            self.model = torch.load(model_path, map_location=torch.device(device))

        self.to(device)

    def train (self, num_episodes, use_database=True, use_tqdm=False):
        if use_tqdm: progress_bar = trange(num_episodes)
        else: progress_bar = range(num_episodes)
        last_saved_itr = "Not saved" 

        for episode in range(progress_bar):
            state, len_moves = self.env.reset()     
            done = False

            if use_database: database_rand = randint(0, 2)
            else: database_rand = 1 

            if database_rand == 0: 
                # 1/3 of the chance to draw from master database
                self.replay_mem.add(*self.env.get_game_from_master_db())
            else:
                # Go to the actual environment
                while not done: 
                    rand_num = randint(0, 100) 
                    if rand_num < self.epsilon * 100: 
                        # Do random action
                        move = randint(0, len_moves-1)
                    else:
                        move = self.model(state)
                    
                    next_state, reward, done = self.env.step(torch.argmax(move, dim=0))
                    self.replay_mem.add(state, next_state, reward, done) 
                    state = next_state

            # Sample random minibatch of transitions from D, and get expected y value
            state_batch, next_state_batch, rewards_batch, done_batch = self.replay_mem(self.batch_size)
            y = rewards_batch + (self.gamma * torch.max(self.model(next_state_batch)) * done_batch)

            # Train on y value as target
            self.opt.zero_grad() 
            out = self.model(state_batch)
            loss = self.mse_loss(out, y)
            loss.backward()
            self.opt.step()

            # log progress bar
            if use_tqdm: 
                progress_bar.set_description(f"Episode: {episode} loss: {loss.item():.3f} Saved: {last_saved_itr}")

            # Save model
            if episode != 0 and episode % self.log_per_itr == 0: 
                torch.save(self.model, self.model_path) 
                last_saved_itr = f"On Episode: {episode}"