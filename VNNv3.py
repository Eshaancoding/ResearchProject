import torch
from torch import nn
import torch.nn.functional as F
import math
import sys; sys.path.append("../")
from PosEnc import *

class VNNv3 (nn.Module):
    def __init__(self, d_model, input_kernel_size, output_kernel_size, hidden_size, device) -> None:
        super().__init__()

        self.weight_nn = nn.Sequential(
            nn.Linear((d_model*2)+input_kernel_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_kernel_size*output_kernel_size),
        )

        self.bias_nn = nn.Sequential(
            nn.Linear(d_model+output_kernel_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_kernel_size),
        ) 
        
        self.d_model = d_model
        self.x_kernel_size = input_kernel_size
        self.y_kernel_size = output_kernel_size
        self.pos_enc = PosEncIndex(d_model, device)
        self.device = device
        self.to(device)

    def forward (self, x:torch.Tensor, output_size:int, debug=False):
        # x shape: (batch_size, input_size), where input_size could be varied
        # output size: could be varied, should output (batch_size, output_size)
        batch_size = x.size(0)
        input_size = x.size(1)

        # find how much times we have to generate weight matrxies in both directions
        gen_x = math.ceil(output_size / self.y_kernel_size)
        gen_y = math.ceil(input_size / self.x_kernel_size)

        bias_arg = torch.arange(gen_x).to(self.device)
        argument_y = torch.arange(gen_y).to(self.device)

        # Pad new x for input and split
        new_x = torch.concat((
            x,
            torch.zeros((batch_size, gen_y * self.x_kernel_size - x.size(1))).to(self.device)
        ), dim=1)
        new_x = new_x.view(batch_size, -1, self.x_kernel_size)
        new_x = new_x.repeat(1, gen_x, 1)

        # expand argument x and y
        argument_x = bias_arg.repeat(gen_y) 
        argument_y = argument_y.repeat_interleave(gen_x)

        # Positional encoding and generate argument
        argument_x = self.pos_enc(argument_x).unsqueeze(0).repeat(batch_size, 1, 1)
        argument_y = self.pos_enc(argument_y).unsqueeze(0).repeat(batch_size, 1, 1)
        argument = torch.concat((argument_x, argument_y, new_x), dim=2)

        # Go through weight neural network
        weight = self.weight_nn(argument).reshape(batch_size, gen_y*self.x_kernel_size, gen_x*self.y_kernel_size)
        
        # Shape weight matrix and matrix multiply the input vector
        weight = weight[:, :input_size, :output_size]
        out = torch.bmm(x.unsqueeze(1), weight).squeeze(1)

        # Now go through bias neural network
        out_inp_bias = torch.concat((
            out,
            torch.zeros((batch_size, gen_x * self.y_kernel_size - out.size(1))).to(self.device)
        ), dim=1) 
        out_inp_bias = out_inp_bias.view(batch_size, -1, self.y_kernel_size)
        bias_arg = self.pos_enc(bias_arg).unsqueeze(0).repeat(batch_size, 1, 1)
        bias = self.bias_nn(torch.concat((out_inp_bias, bias_arg), dim=2))
        bias = bias.view(batch_size, -1)[:, :output_size]

        if debug: 
            return out + bias, gen_x, gen_y
        else:  
            return out + bias

if __name__ == "__main__":
    vnn = VNNv3(64, 4, 3, 30, "cpu")
    out = vnn.forward(torch.randn(5, 18), 5)