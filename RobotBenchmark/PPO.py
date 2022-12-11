import torch
import evogym.envs
import gym
from random import randint
from copy import deepcopy
from evogym import sample_robot
import numpy as np
import Agent

class Trainer:
    def __init__(self, 
            policy:Agent.Policy,
            value:Agent.Value,
            evogym_name, 
            device,
            min_size:int,
            max_size:int,
            epsilon:float=0.2, 
            gamma:float=0.98,
            exploration_decay_rate:float=0.9995
        ) -> None:

        self.device = device
        self.policy = policy
        self.value = value
        self.old_policy = deepcopy(self.policy)
        self.epsilon = epsilon
        self.gamma = gamma
        self.explore_decay_rate = exploration_decay_rate
        self.env_name = evogym_name
        self.min_size = min_size
        self.max_size = max_size
        self.std = 1

    def test (self, total_timesteps=-1, use_old_policy=False, render=True, return_training_vars=False):
        # find suitable body and reset env
        while True:
            body, connections = sample_robot(
                (randint(self.min_size, self.max_size),randint(self.min_size, self.max_size))
            )

            num_count_below = np.count_nonzero(body[-1] == 3) + np.count_nonzero(body[-1] == 4)
            num_count_one = np.count_nonzero(body[-2] == 3) + np.count_nonzero(body[-2] == 4)
            if num_count_below + num_count_one > 4:
                break
        
        env = gym.make(self.env_name, body=body)
        ob = env.reset()

        # Variables for training
        deltas = torch.tensor([]).to(self.device)

        # set policy
        pol = self.old_policy if use_old_policy else self.policy

        # convert to tensor
        body = torch.tensor(body/10).flatten().to(self.device)
        connections = torch.tensor(connections/10).flatten().to(self.device)

        # main loop
        ind = 0
        while True: 
            x = torch.concat((
                body,
                connections,
                torch.tensor(ob).to(self.device)
            ), dim=0).unsqueeze(0).to(self.device).to(torch.float)
            action = pol(x, env.action_space.shape[0], self.std if return_training_vars else None)

            new_ob, reward, done, _ = env.step(action.cpu().detach().numpy())

            # calculate deltas (for advantage) if we have to
            if return_training_vars:
                new_x = torch.concat((
                    body,
                    connections,
                    torch.tensor(new_ob).to(self.device)
                ), dim=0).unsqueeze(0).to(self.device).to(torch.float)

                delta = reward + self.gamma * self.value(new_x, pol.InputVnnBlock)[0] - self.value(x, pol.InputVnnBlock)[0]
                deltas = torch.hstack((deltas, delta))

            # Render
            if render: 
                env.render()

            # Handle indexes             
            ind += 1
            if ind == total_timesteps: 
                break

            # Break
            if done: 
                break


            # std
            if return_training_vars:
                self.std *= self.explore_decay_rate

            # observation
            ob = new_ob
                
        # convert deltas to advantages
        advantages = torch.tensor([]).to(self.device)
        advantage = 0
        for i in range(deltas.size(0)-1, -1, -1):
            advantage = deltas[i] + self.gamma*advantage 
            advantages = torch.hstack((advantage, advantages))
                
        env.close()
        
        if return_training_vars:
            return advantages
        
    def train (self):
        # Don't forget to negate (-) the surrogate loss objective since we want to maximize this
        return None