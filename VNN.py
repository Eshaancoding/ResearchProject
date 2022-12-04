from copy import deepcopy
import torch
from torch import nn
from torch import Tensor
import math
from PosEnc import *


class VNNBlock (nn.Module):
    def __init__(self, d_model, weight_nn, bias_nn, device=None) -> None:
        super().__init__()
        self.weight_nn = weight_nn 
        self.bias_nn = bias_nn 

        self.pos_enc = PosEncIndex(d_model, device=("cpu" if device == None  else device))
        
        if device != None:
            self.device = device
            self.to(device)
        else: 
            self.device = "cpu"

    def weight_propagation (self, x, output_size, extra_out):
        input_size = x.size(1)
        batch_size = x.size(0)
        x = x.to(self.device)

        #* Weight Generation
        # Generate the weight vector
        argument_one = torch.arange(input_size).to(self.device)
        argument_two = torch.arange(output_size).to(self.device)

        # Generate the repeat
        argument_one = argument_one.repeat(batch_size, output_size)
        argument_two = argument_two.repeat(batch_size, input_size)

        x_concat = x.repeat(1, output_size).unsqueeze(2).to(self.device)

        # Positional Encoding + Concat
        argument_one = self.pos_enc(argument_one.detach())
        argument_two = self.pos_enc(argument_two.detach())

        if extra_out != None:  
            argument = torch.concat((argument_one, argument_two, x_concat, extra_out.repeat(1, input_size, 1)), dim=2)
        else:
            argument = torch.concat((argument_one, argument_two, x_concat), dim=2)
        
        weights = self.weight_nn(argument.detach()).view(batch_size, input_size, output_size)
        x = x.view(batch_size, 1, input_size)
        out = torch.bmm(x, weights).squeeze(1)

        #* Bias Generation

        # Create Bias Argument
        argument_one = torch.arange(output_size).to(self.device)
        argument_one = self.pos_enc(argument_one.detach()).squeeze(1)
        argument_one = argument_one.repeat(batch_size, 1, 1)
        argument_two = out.unsqueeze(2)
        if extra_out != None:
            bias_argument = torch.concat((argument_one, argument_two, extra_out), dim=2)
        else:
            bias_argument = torch.concat((argument_one, argument_two), dim=2)

        # Add bias
        bias = self.bias_nn(bias_argument.detach()).squeeze(2)
        out += bias

        return out

    def return_gpu_desc (self):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        return f"Free: {f/1024**2} MB; Allocated: {a/1024**2} MB"

    def forward (self, x, output_size, extra_out=None, chunks="none"):
        # Extra Out size: 
        # first dim is the batch size, second dim is the output space, third dim is the vector added during weight 
        if extra_out != None:
            assert x.size(0) == extra_out.size(0), f"Batch size of x ({x.size(0)}) is the same as the batch size of extra_out ({extra_out.size(0)})"

        if chunks == "all": chunks = output_size
        elif chunks == "none": chunks = 1

        arr = [output_size // chunks for _ in range(chunks)]        
        if output_size % chunks > 0: 
            arr.append(output_size % chunks) 

        out = torch.tensor([])
        for i in range(len(arr)):
            output = self.weight_propagation(x, arr[i], extra_out)
            if out.size(0) == 0: 
                out = output 
            else:
                out = torch.concat((out, output), dim=1)
        return out