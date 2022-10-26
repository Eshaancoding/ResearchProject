import torch
from random import randint

class ReplayMemory ():
    def __init__(self, max_len) -> None:
        self.max_len = max_len 

        self.states = []
        self.next_states = []
        self.rewards = []
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
        self.actions = self.actions[-self.max_len:]
        
    def get_batch (self, batch_size):
        indexes = torch.randperm(len(self.states))[:batch_size].tolist()

        states_return = torch.tensor([])
        next_states_return = torch.tensor([])
        rewards_return = torch.tensor([])
        dones_return = torch.tensor([])
        actions_return = [] 

        for i in range(batch_size):
            if i >= len(indexes):
                random_index = randint(0, len(self.states))
            else: 
                random_index = indexes[i] 
            
            states_return = self.addToTensor(states_return, self.states[random_index])
            next_states_return = self.addToTensor(next_states_return, self.next_states[random_index])
            rewards_return = self.addToTensor(rewards_return, self.rewards[random_index])
            dones_return = self.addToTensor(dones_return, self.dones[random_index])
            actions_return.append(self.actions[random_index]) 
    
        return actions_return, states_return, next_states_return, rewards_return, dones_return