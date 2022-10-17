from re import S
from pyparsing import original_text_for
import torch
from random import randint
from tqdm import trange
from ChessEnv import * 
from os.path import exists
from copy import deepcopy

class ReplayMemory ():
    def __init__(self, max_len) -> None:
        self.max_len = max_len 
        self.states = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.actions = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def addToTensor (self, origTensor, addTensor):
        if isinstance(addTensor, bool):
            if addTensor: 
                addTensor = torch.tensor([1]).to(torch.float).to(self.device)
            else:
                addTensor = torch.tensor([0]).to(torch.float).to(self.device)
        else:
            addTensor = torch.tensor(addTensor).unsqueeze(0).to(torch.float).to(self.device)
        if origTensor.size(0) == 0: 
            return addTensor
        else:
            return torch.concat((origTensor, addTensor), dim=0)

    def add (self, action, state, next_state, reward, done):     
        self.states.append(state)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.actions.append(action)

        self.states = self.states[-self.max_len:]
        self.next_states = self.next_states[-self.max_len:]
        self.rewards = self.rewards[-self.max_len:]
        self.dones = self.dones[-self.max_len:]

    def get_batch (self, batch_size):
        actions_return = [] 
        states_return = torch.tensor([])
        next_states_return = torch.tensor([])
        rewards_return = torch.tensor([])
        dones_return = torch.tensor([])

        for _ in range(batch_size):
            random_index = randint(0, len(self.states)-1)
            
            actions_return.append(self.actions[random_index]) 
            states_return = self.addToTensor(states_return, self.states[random_index])
            next_states_return = self.addToTensor(next_states_return, self.next_states[random_index])
            rewards_return = self.addToTensor(rewards_return, self.rewards[random_index])
            dones_return = self.addToTensor(dones_return, self.dones[random_index])
    
        return actions_return, states_return, next_states_return, rewards_return, dones_return

class DQN:
    def __init__(self, 
        model,                          # The model used to train
        replay_mem_max_len,             # The max length of the replay memory 
        epsilon,                        # The probability of choosing a random action
        batch_size,                     # Batch size hyperparameter used for training 
        gamma,                          # Gamma Hyperparameter (used for balancing importance of future rewards) 
        lr,                             # Learning rate hyperparameter used during training
        model_path,                     # Model path for saving the model (could be None to disable saving the model) 
        log_per_epi,                    # Save the model per # episodes (could be None to disable saving the model)
        update_target_model_per_epi,    # Update target weights with the original model weights per # epsidoes 
        max_test_itr                    # Max amount of iterations in an episode when testing the model
    ) -> None:

        self.model = model # The model will act as a value function but will output the value of each output in the output layer of the neural network, thus acting as a q-value function
        self.replay_mem = ReplayMemory(max_len=replay_mem_max_len)
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.gamma = gamma
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.mse_loss = torch.nn.MSELoss()
        self.max_test_itr = max_test_itr

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model saving
        self.model_path = model_path
        self.log_per_epi = log_per_epi

        # Check if model path exists, and if so load the model
        if self.model_path != None and self.log_per_epi != None and exists(model_path):
            print(f"Loading model from path: {model_path}")
            self.model = torch.load(model_path, map_location=torch.device(self.device))

        # Get target function (stable in training)
        self.target_model = deepcopy(self.model)
        self.update_target_model_per_epi = update_target_model_per_epi

        # Put models to device
        self.target_model.to(self.device)
        self.model.to(self.device)

    def test (self, env, max_num_itr, render):
        avg_reward = 0
        
        # ========================== DELETE THIS LATER ===================================    
        state, _ = env.reset()
        # ========================== DELETE THIS LATER ===================================    

        state = [state for i in range(2)]
        itr = 0

        for _ in range(max_num_itr):
            x = torch.tensor(state).to(torch.float).to(self.device)
            move = self.model(x)
            state, reward, done, _ = env.step(torch.argmax(move, dim=0).item())
            avg_reward += reward
            itr += 1
            if done: break
            if render:
                env.render()

        return avg_reward / itr

    def train (self, env, num_episodes, use_database=True, use_tqdm=False, render=False):
        if use_tqdm: progress_bar = trange(num_episodes)
        else: progress_bar = range(num_episodes)
        last_saved_itr = "Not saved" 
        avg_reward = 0

        for episode in progress_bar:
            state, _ = env.reset()

            # ========================== DELETE THIS LATER ===================================    
            state = [state for i in range(2)]
            # ========================== DELETE THIS LATER ===================================    

            done = False

            if use_database: database_rand = randint(0, 2)
            else: database_rand = 1 

            if database_rand == 0: 
                # 1/3 of the chance to draw from master database
                self.replay_mem.add(*env.get_database())
            else:
                # Go to the actual environment
                while not done: 
                    rand_num = randint(0, 100) 
                    if rand_num < self.epsilon * 100: 
                        # Do random action
                        move = env.action_space.sample()
                    else:
                        x = torch.tensor(state).to(torch.float).to(self.device)
                        move = torch.argmax(self.model(x)).item()

                    next_state, reward, done, _ = env.step(move)
                    self.replay_mem.add(move, state, next_state, reward, done) 
                    state = next_state

            # Sample random minibatch of transitions from D, and get expected y value
            action_batch, state_batch, next_state_batch, rewards_batch, done_batch = self.replay_mem.get_batch(self.batch_size)
            y = rewards_batch + (self.gamma * torch.max(self.target_model(next_state_batch), dim=1).values * done_batch)

            # Train on y value as target
            self.opt.zero_grad() 
            out = self.model(state_batch)
            
            # Set the y value that is corresponding with the target
            # If there is anyway to do this without a for loop, let me know. 
            target = out.detach()
            for i in range(self.batch_size):
                target[i][action_batch[i]] = y[i]
            
            loss = self.mse_loss(out, target)
            loss.backward()
            self.opt.step()

            # log progress bar
            if use_tqdm: 
                progress_bar.set_description(f"Episode: {episode} Avg reward: {avg_reward:.2f} Last Model Saved: {last_saved_itr}")

            # Save model
            if self.model_path != None and self.log_per_epi != None and episode != 0 and episode % self.log_per_epi == 0: 
                torch.save(self.model, self.model_path) 
                last_saved_itr = f"On Episode: {episode}"

            # Update target model
            if episode != 0 and episode % self.update_target_model_per_epi == 0:
                self.target_model.load_state_dict(self.model.state_dict())

                # Test model
                avg_reward = self.test(env=env, max_num_itr=self.max_test_itr, render=render) 
