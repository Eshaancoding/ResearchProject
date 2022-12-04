import torch
from torch import nn
import torch.nn.functional as F
import math
import sys; sys.path.append("../")
from PosEnc import *

class VNNv3 (nn.Module):
    def __init__(self, d_model, x_kernel_size, y_kernel_size, device) -> None:
        super().__init__()

        self.weight_nn = nn.Sequential(
            nn.Linear(d_model*2, 12),
            nn.Tanh(),
            nn.Linear(12, x_kernel_size*y_kernel_size),
        )

        self.bias_nn = nn.Sequential(
            nn.Linear(d_model, 12),
            nn.Tanh(),
            nn.Linear(12, x_kernel_size),
        ) 
        
        self.d_model = d_model
        self.x_kernel_size = x_kernel_size
        self.y_kernel_size = y_kernel_size
        self.pos_enc = PosEncIndex(d_model, device)
        self.device = device
        self.to(device)

    def forward (self, x:torch.Tensor, output_size:int):
        # x shape: (batch_size, input_size), where input_size could be varied
        # output size: could be varied, should output (batch_size, output_size)
        input_size = x.size(1)

        # find how much times we have to generate weight matrxies in both directions
        gen_x = math.ceil(output_size / self.y_kernel_size)
        gen_y = math.ceil(input_size / self.x_kernel_size)

        argument_x = torch.arange(gen_x).to(self.device)
        argument_y = torch.arange(gen_y).to(self.device)
        bias_argument = self.pos_enc(argument_y)

        # expand argument x and y
        argument_x = argument_x.repeat(gen_y) 
        argument_y = argument_y.repeat_interleave(gen_x)
        # Positional encoding and generate argument
        argument_x = self.pos_enc(argument_x)
        argument_y = self.pos_enc(argument_y)
        argument = torch.concat((argument_x, argument_y), dim=1)

        # Go through weight neural network
        weight = self.weight_nn(argument).reshape(gen_y*self.x_kernel_size, gen_x*self.y_kernel_size)

        # Shape weight matrix and matrix multiply the input vector
        weight = weight[:input_size, :output_size]
        out = torch.matmul(x, weight)

        # Now go through bias neural network
        bias = self.bias_nn(bias_argument).flatten()[:input_size].unsqueeze(1)
        bias_out = torch.matmul(x, bias)

        return out + bias_out

if __name__ == "__main__":
    vnn = VNNv3(64, 6, 2, "cpu")
    out = vnn.forward(torch.randn(5, 15), 4)