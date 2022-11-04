import torch
from random import randint
import numpy as np

class ReplayMemory ():
    def __init__(self, max_len) -> None:
        self.max_len = max_len 

        self.arr = []
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"

    def addToTensor (self, origTensor, addTensor):
        if isinstance(addTensor, np.ndarray):
            addTensor = torch.from_numpy(addTensor).unsqueeze(0).to(torch.float).to(self.device)
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

    def add (self, action, state, next_state, reward, done, extra_state=None, extra_next_state=None, arrays=False):     
        if arrays:
            for i in range(len(action)):
                self.arr.append((
                    state[i],
                    extra_state[i],
                    next_state[i],
                    extra_next_state[i],
                    action[i],
                    reward[i],
                    done[i],
                )) 
        else:
            if isinstance(extra_state, bool):
                raise KeyError("sdf")
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