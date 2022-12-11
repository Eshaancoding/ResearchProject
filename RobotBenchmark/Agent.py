from copy import deepcopy
import torch
from torch import nn
import sys; sys.path.append("../")
from NN.VNN import *
from NN.VNNv2 import *
from NN.VNNv3 import *


#********** TODO IMPLEMENT STD!! ***************
class Policy (nn.Module):
    def __init__(self, VNN_v, device) -> None:
        super().__init__()
        self.device = device

        if VNN_v == 1: 
            d_model = 64
            weight_model = nn.Sequential(
                nn.Linear(d_model*2+1, 16),
                nn.Tanh(),
                nn.Linear(16, 1),
            ) 
            bias_model = nn.Sequential( 
                nn.Linear(d_model+1, 12),
                nn.Tanh(),
                nn.Linear(12, 1),
            )
            self.InputVnnBlock = VNNBlock(d_model, weight_model, bias_model, device=device)
        elif VNN_v == 2:
            self.InputVnnBlock = VNNBlockTwo(d_model=64, initial_size=10, kernel_size=5, device=device)
        elif VNN_v == 3:
            self.InputVnnBlock = VNNv3(d_model=64, input_kernel_size=200, output_kernel_size=10, hidden_size=64, device=device)

        self.OutputVnnBlock = deepcopy(self.InputVnnBlock) 

        self.mid_nn = nn.Sequential(
            nn.Linear(80, 50),
            nn.Tanh(),
            nn.Linear(50, 30),
            nn.Tanh(),
        )        
        
        self.device = device
        self.to(device)
    
    def forward (self, x, output_size, std=None): 
        x = self.InputVnnBlock(x, 80)
        x = torch.tanh(x)
        x = self.mid_nn(x)
        x = torch.sigmoid(self.OutputVnnBlock(x, output_size)) + 0.6
        if std != None:
            print(x)
            x = torch.normal(x, std)
            print(x)
        return x

class Value (nn.Module):
    def __init__(self, device) -> None:
        super().__init__()

        self.end_nn = nn.Sequential(
            nn.Linear(80, 50),
            nn.Tanh(),
            nn.Linear(50, 30),
            nn.Tanh(),
            nn.Linear(30, 1)
        )

        self.to(device)
    
    def forward (self, x, inputVNNBlock):
        x = inputVNNBlock(x, 80).detach()
        return self.end_nn(x)