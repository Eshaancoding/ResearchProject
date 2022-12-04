import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
import sys; sys.path.append("../") 
from PosEnc import *

class VNNBlockTwo (nn.Module):
    def __init__(self, d_model, initial_size, kernel_size, pad_size, device) -> None:
        super().__init__()
        self.initial_param = nn.Parameter(torch.randn(1, 1, initial_size, initial_size, requires_grad=True))
        self.initial_param_bias = nn.Parameter(torch.randn(1, 1, initial_size, initial_size, requires_grad=True))

        self.extra_flag_len = 10

        self.weight_nn = nn.Sequential(
            nn.Linear(d_model+self.extra_flag_len, 12),
            nn.Linear(12, kernel_size*kernel_size+1),
        )

        self.bias_nn = nn.Sequential(
            nn.Linear(d_model+self.extra_flag_len, 12),
            nn.Linear(12, (kernel_size*kernel_size+1)),
        )

        self.pad_size = pad_size
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.pos_enc = PosEncIndex(d_model, device=device)
        self.device = device 
        self.to(self.device)

    def expand_matrix (self, matrix, index, use_bias_nn=False, include_activation=True):
        index = self.pos_enc(torch.tensor([index]))
        expansion_input = torch.concat([index, torch.zeros(1,self.extra_flag_len).to(self.device)], dim=1)
        conv_input = torch.concat([index, torch.ones(1,self.extra_flag_len).to(self.device)], dim=1)

        # First expand
        if use_bias_nn: nn_out = self.bias_nn(expansion_input) 
        else: nn_out = self.weight_nn(expansion_input)

        kernel = nn_out[0][:-1].view(1,1,self.kernel_size, self.kernel_size).repeat(1,1,matrix.size(2), matrix.size(2))
        bias = nn_out[0][-1].unsqueeze(0)
        
        m = nn.Upsample(scale_factor=self.kernel_size, mode="nearest")
        matrix = m(matrix)

        matrix = (matrix * kernel) + bias

        # # Then use conv input
        # if use_bias_nn: nn_out = self.bias_nn(conv_input)
        # else: nn_out = self.weight_nn(conv_input)

        # kernel = nn_out[0][:-1].view(1,1,self.kernel_size, self.kernel_size)
        # bias = nn_out[0][-1].unsqueeze(0)

        # matrix = F.conv2d(matrix, kernel, bias)
    
        return matrix

    def return_gpu_desc (self):
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        return f"Free: {f/1024**2} MB; Allocated: {a/1024**2} MB"

    def forward (self, x, output_size, debug=False):
        assert isinstance(output_size, int) and isinstance(x, torch.Tensor)
        
        input_space = x.size(1)
        optimal_out_size = math.ceil(math.sqrt(x.size(1) * output_size))
        optimal_bias_size = math.ceil(math.sqrt(output_size))

        # Afterwards, deconvolute the weight matrix until it exceeds or equals optimal_out_size
        weight_matrix = self.initial_param
        i_upscale = 0
        while weight_matrix.size(2) < optimal_out_size:
            weight_matrix = self.expand_matrix(weight_matrix, i_upscale, use_bias_nn=False)
            i_upscale += 1
        weight_matrix = weight_matrix.repeat(x.size(0), 1, 1, 1) 

        # Deconvolute the bias matrix until it exceeds or equals optimal_bias_size
        i_upscale_bias = 0
        bias_matrix = self.initial_param_bias
        while bias_matrix.size(2) < optimal_bias_size:
            bias_matrix = self.expand_matrix(bias_matrix, i_upscale_bias, use_bias_nn=True)
            i_upscale_bias += 1
        bias_matrix = bias_matrix.repeat(x.size(0), 1, 1, 1)

        # Weight matrix
        i_weight = weight_matrix.flatten(start_dim=1)
        i_weight = i_weight.index_select(1, torch.arange(0,input_space*output_size).to(self.device))
        i_weight = i_weight.view(x.size(0), input_space, output_size)

        # Bias matrix
        i_bias = bias_matrix.flatten(start_dim=1)
        i_bias = i_bias.index_select(1, torch.arange(0, output_size).to(self.device))

        # Propagate through input
        x = x.unsqueeze(1)
        out_i = torch.bmm(x.to(self.device), i_weight).squeeze(1) + i_bias
        if debug: 
            return out_i, i_upscale, i_upscale_bias
        else:
            return out_i