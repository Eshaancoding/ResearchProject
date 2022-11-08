import torch
from random import randint
from tqdm import trange
from Chess.ChessEnv import * 
from os.path import exists
from copy import deepcopy
from RL.ReplayMemory import *
import time
import gym

class DQN:
    def __init__(self, 
        model,                          # The model used to train
        replay_mem_max_len,             # The max length of the replay memory 
        epsilon,                        # The probability of choosing a random action
        batch_size,                     # Batch size hyperparameter used for training 
        gamma,                          # Gamma Hyperparameter (used for balancing importance of future rewards) 
        lr,                             # Learning rate hyperparameter used during training
        model_path,                     # Model path for saving the model (could be None to disable saving the model) 
        should_load_from_path,          # If it should load the model from the specified model_path argument
        save_per_epi,                   # Save the model per # episodes (could be None to disable saving the model)
        update_target_model_per_epi,    # Update target weights with the original model weights per # epsidoes 
        epsilon_decay,              
        epsilon_min,
        test_per_epi
    ) -> None:
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model # The model will act as a value function but will output the value of each output in the output layer of the neural network, thus acting as a q-value function
        self.replay_mem = ReplayMemory(max_len=replay_mem_max_len, device=self.device)
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.gamma = gamma
        self.mse_loss = torch.nn.MSELoss()
        self.lr = lr
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.test_per_epi = test_per_epi

        # Model saving
        self.model_path = model_path
        self.save_per_epi = save_per_epi

        # Check if model path exists, and if so load the model
        if self.model_path != None and should_load_from_path and exists(model_path):
            print(f"Loading model from path: {model_path}")
            self.model = torch.load(model_path, map_location=torch.device(self.device))

        # Get target function (stable in training)
        self.target_model = deepcopy(self.model)
        self.update_target_model_per_epi = update_target_model_per_epi

        # Put models to device
        self.target_model.to(self.device)
        self.model.to(self.device)

    def test (self, env):
        test_env = deepcopy(env)
        avg_reward = 0
        done = False 
        state, extra_state, _ = test_env.reset()
        itr = 0

        while not done:
            x = state.to(torch.float).to(self.device)
            extra_state = extra_state.to(torch.float).to(self.device)
            move = torch.argmax(self.model(x, extra_state), dim=0).item()

            state, extra_state, reward, done, _, _ = test_env.step(move)

            avg_reward += reward
            itr += 1

        return avg_reward / itr

    def train (self, env, num_episodes, use_tqdm=False, should_test=True):
        if use_tqdm: progress_bar = trange(num_episodes)
        else: progress_bar = range(num_episodes)
        last_saved_itr = "Not saved" 
        avg_reward = 0
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for episode in progress_bar:
            # Test env
            state, extra_state, _ = env.reset()
            done = False
            i = 0
            while not done: 
                rand_num = randint(0, 100) 
                if rand_num < self.epsilon * 100: 
                    # Do random action
                    move = env.action_space.sample()
                else:
                    x = state.to(torch.float).to(self.device)
                    extra_state = extra_state.to(torch.float).to(self.device)
                    move = torch.argmax(self.model(x, extra_state)).item()

                if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

                next_state, next_extra_state, reward, done, _, _ = env.step(move)

                self.replay_mem.add(move, state, next_state, reward, done, extra_state, next_extra_state) 
                
                state = next_state
                extra_state = next_extra_state
                i += 1

            # Sample random minibatch of transitions from D, and get expected y value
            losses = torch.tensor([]).to(self.device)
            opt.zero_grad() 
            for state, extra_state, next_state, extra_next_state, action, reward, done in self.replay_mem.get_batch(self.batch_size):
                y = reward + (self.gamma * torch.max(self.target_model(next_state, extra_next_state), dim=0).values * (1 - done))
            
                # Set the y value that is corresponding with the target
                # If there is anyway to do this without a for loop, let me know. 
                
                # train 
                out = self.model(state, extra_state)
                target = out.detach().clone()
                target[action] = y

                loss = self.mse_loss(out, target)
                losses = torch.hstack((losses, loss))

            loss = torch.mean(losses)
            loss.backward()
            opt.step()

            # log progress bar
            if use_tqdm: 
                # Test model
                progress_bar.set_description(f"Episode: {episode} Avg reward: {avg_reward:.2f} loss: {loss.item():.3f} model saved: {last_saved_itr}")

            # Save model
            if self.model_path != None and self.save_per_epi != None and episode != 0 and episode % self.save_per_epi == 0: 
                torch.save(self.model, self.model_path) 
                last_saved_itr = f"epi {episode}"

            # Update target model
            if episode != 0 and episode % self.update_target_model_per_epi == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            # Test
            if episode != 0 and episode % self.test_per_epi == 0 and should_test: 
                avg_reward = self.test(env=env) 

        if self.model_path != None and self.save_per_epi != None and episode != 0 and episode % self.save_per_epi == 0: 
            torch.save(self.model, self.model_path) 