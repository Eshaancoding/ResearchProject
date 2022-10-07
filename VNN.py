from copy import deepcopy
import torch
from torch import nn
from torch import Tensor
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:

        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(1, 1, self.d_model)
        print(x.shape)
        pe[:, 0, 0::2] = torch.sin(x * div_term)
        pe[:, 0, 1::2] = torch.cos(x * div_term)
        return pe

class VNNBlock (nn.Module):
    def __init__(self, weight_nn, bias_nn) -> None:
        super().__init__()
        d_model = (weight_nn[0].in_features-1)//2
        self.weight_nn = weight_nn 
        self.bias_nn = bias_nn 
        self.pos_enc = PositionalEncoding(d_model,max_len=5000)

    def weight_propagation (self, x, output_size):
        input_size = x.size(1)
        seq_len = x.size(0)

        #* Weight Generation
        # Generate the weight vector
        argument_one = torch.arange(input_size)
        argument_two = torch.arange(output_size)

        # Generate the repeat
        argument_one = argument_one.repeat(seq_len, output_size)
        argument_two = argument_two.repeat(seq_len, input_size)

        x_concat = x.repeat(1, output_size).unsqueeze(2).to(x.device)

        # Positional Encoding + Concat
        argument_one = self.pos_enc(argument_one.detach())
        argument_two = self.pos_enc(argument_two.detach())

        argument = torch.concat((argument_one, argument_two, x_concat), dim=2)
        
        weights = self.weight_nn(argument.detach()).view(seq_len, input_size, output_size)
        x = x.view(seq_len, 1, input_size)
        out = torch.bmm(x, weights).squeeze(1)

        #* Bias Generation

        # Create Bias Argument
        argument_one = torch.arange(output_size)
        argument_one = self.pos_enc(argument_one.detach()).squeeze(1)
        argument_one = argument_one.repeat(seq_len, 1, 1)
        argument_two = out.unsqueeze(2)
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

    def forward (self, x, output_size, chunks=1):
        if chunks == "all":
            chunks = output_size
        elif chunks == "none":
            chunks = 0

        arr = [output_size // chunks for _ in range(chunks)]        
        if output_size % chunks > 0: 
            arr.append(output_size % chunks) 

        out = torch.tensor([])
        output_size = 5

        for i in range(len(arr)):
            output = self.weight_propagation(x, arr[i])
            if out.size(0) == 0: 
                out = output 
            else:
                out = torch.concat((out, output), dim=1)
        return out