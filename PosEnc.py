import torch
from torch import Tensor, nn
import math

class PosEncIndex(nn.Module):
    def __init__(self, d_model: int, device):
        super().__init__()
        self.d_model = d_model
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        length = torch.max(x).item()+1
        
        pe = torch.zeros((length, self.d_model)).to(self.device)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / self.d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe[x]