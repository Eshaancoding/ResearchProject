import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
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

class VNNBlockTwo (nn.Module):
    def __init__(self, initial_size, d_model, kernel_size, device=None) -> None:
        super().__init__()
        self.initial_param = nn.Parameter(torch.randn(1, 1, initial_size, initial_size, requires_grad=True))
        self.initial_param_bias = nn.Parameter(torch.randn(1, 1, initial_size, initial_size, requires_grad=True))
        # self.initial_param = torch.ones(1,1,initial_size, initial_size)
        # self.initial_param_bias = torch.ones(1,1,initial_size, initial_size)

        self.weight_nn = nn.Sequential(
            nn.Linear(d_model, 48),
            nn.Tanh(),
            nn.Linear(48, kernel_size*kernel_size+1)
        )

        self.bias_nn = nn.Sequential(
            nn.Linear(d_model, 48),
            nn.Tanh(),
            nn.Linear(48, kernel_size*kernel_size+1)
        )

        self.kernel_size = kernel_size

        if device != None:
            self.device = device 
            self.to(self.device)
            self.pos_enc = PosEncIndex(d_model, device=device)
        else:
            self.pos_enc = PosEncIndex(d_model)

    def weight_kernel (self, weight_matrix, index):
        inp_pos_enc = self.pos_enc(torch.tensor([index]))
        nn_out = self.weight_nn(inp_pos_enc)
        kernel = nn_out[0][:-1].view(1,1,self.kernel_size, self.kernel_size)
        bias = nn_out[0][-1].unsqueeze(0)
        
        # Pad and then convolution
        weight_matrix = F.pad(input=weight_matrix, pad=(2,2,2,2), mode="constant", value=0)
        weight_matrix = F.conv2d(weight_matrix, kernel, bias=bias)        

        return weight_matrix

    def bias_kernel (self, bias_matrix, index):
        inp_pos_enc = self.pos_enc(torch.tensor([index]))
        nn_out = self.bias_nn(inp_pos_enc)
        kernel = nn_out[0][:-1].view(1,1,self.kernel_size, self.kernel_size)
        bias = nn_out[0][-1].unsqueeze(0)
        
        # Pad and then convolution
        bias_matrix = F.pad(input=bias_matrix, pad=(2,2,2,2), mode="constant", value=0)
        bias_matrix = F.conv2d(bias_matrix, kernel, bias=bias)        

        return bias_matrix

    def return_gpu_desc (self):
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        return f"Free: {f/1024**2} MB; Allocated: {a/1024**2} MB"

    def forward (self, x, output_size, debug=False):
        is_single_out = False
        if isinstance(output_size, int) and isinstance(x, torch.Tensor):
            is_single_out = True
            optimal_out_size = math.ceil(math.sqrt(x.size(1) * output_size))
            optimal_bias_size = math.ceil(math.sqrt(output_size))

        else:
            assert len(x) == len(output_size), "The arr length of x must be equal to the array length of output_size"
            optimal_out_size = max([math.ceil(math.sqrt(x[i].size(0) * output_size[i])) for i in range(len(output_size))])
            optimal_bias_size = max([math.ceil(math.sqrt(i)) for i in output_size])

        # Afterwards, deconvolute the weight matrix until it exceeds or equals optimal_out_size
        weight_matrix = self.initial_param
        i_upscale = 0
        while weight_matrix.size(2) < optimal_out_size:
            weight_matrix = self.weight_kernel(weight_matrix, i_upscale)
            i_upscale += 1

        # Deconvolute the bias matrix until it exceeds or equals optimal_bias_size
        i_upscale_bias = 0
        bias_matrix = self.initial_param_bias
        while bias_matrix.size(2) < optimal_bias_size:
            bias_matrix = self.bias_kernel(bias_matrix, i_upscale_bias)
            i_upscale_bias += 1

        if not is_single_out: 
            out = []
            for index, i in enumerate(x):
                # Weight matrix
                i_weight = weight_matrix[0][0].flatten()
                i_weight = i_weight[:i.size(0) * output_size[index]]
                i_weight = i_weight.view(i.size(0), output_size[index])

                # Bias matrix
                i_bias = bias_matrix[0][0].flatten()
                i_bias = i_bias[:output_size[index]]
                
                # Propagate through input
                out_i = torch.matmul(i.to(self.device), i_weight) + i_bias
                
                # Append to out
                out.append(out_i.to(self.device))
            if debug:
                return out, i_upscale, i_upscale_bias
            else: 
                return out
        else:
            # Weight matrix
            i_weight = weight_matrix[0][0].flatten()
            i_weight = i_weight[:x.size(1) * output_size]
            i_weight = i_weight.view(x.size(1), output_size)

            # Bias matrix
            i_bias = bias_matrix[0][0].flatten()
            i_bias = i_bias[:output_size]

            # Repeat weight matrix across all outputs
            i_weight = i_weight.unsqueeze(0).repeat(x.size(0), 1, 1)
            i_bias = i_bias.unsqueeze(0).repeat(x.size(0), 1, 1)

            # Propagate through input
            x = x.unsqueeze(1)
            out_i = torch.bmm(x.to(self.device), i_weight) + i_bias
            if debug: 
                return out_i.squeeze(1), i_upscale, i_upscale_bias
            else:
                return out_i.squeeze(1)