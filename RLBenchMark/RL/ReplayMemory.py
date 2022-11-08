import torch
from random import randint
import numpy as np

class ReplayMemory ():
    def __init__(self, max_len, device) -> None:
        self.max_len = max_len 
        self.arr = []
        self.device = device

    
    def convertToTensor (self, obj):
        if isinstance(obj, np.ndarray): obj = torch.from_numpy(obj).to(torch.float).to(self.device)
        if isinstance(obj, bool): obj = torch.tensor([1 if obj else 0]).to(torch.float).to(self.device)
        if isinstance(obj, float) or isinstance(obj, int): obj = torch.tensor(obj).to(torch.float).to(self.device)
        elif isinstance(obj, torch.Tensor): obj = obj.to(torch.float).to(self.device)

        return obj

    def add (self, action, state, next_state, reward, done, extra_state=None, extra_next_state=None):     
        state = self.convertToTensor(state)
        next_state = self.convertToTensor(next_state)
        reward = self.convertToTensor(reward)
        done = self.convertToTensor(done)
        if extra_state != None: extra_state = self.convertToTensor(extra_state)
        if extra_next_state != None: extra_next_state = self.convertToTensor(extra_next_state)

        self.arr.append((
            state,
            extra_state,
            next_state,
            extra_next_state,
            action,
            reward,
            done,
        ))

        self.arr = self.arr[-self.max_len:]
        
    def get_batch (self, batch_size):
        batch = []
        indexes = torch.randperm(len(self.arr))[:batch_size].tolist() 
        for index in indexes:
            batch.append(self.arr[index])
        return batch