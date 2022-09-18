from argparse import ArgumentError
from copy import deepcopy
import torch
from torch import nn
from torch import Tensor
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.pe[x].squeeze(2)

class VNN (nn.Module):
    def __init__(self, dense_nn, weight_nn, bias_nn) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__()
        self.d_model = (weight_nn[0].in_features-1)//2
        self.first_input = dense_nn[0].in_features

        self.dense_nn = dense_nn
        self.weight_nn = weight_nn 
        self.bias_nn = bias_nn 
        self.pos_enc = PositionalEncoding(self.d_model, dropout=0.1, max_len=5000).to(self.device)

        self.to(self.device)

    def generate_weight_vector (self, x, output_size):
        input_size = x.size(1)
        seq_len = x.size(0)
        x = x.to(self.device)
        #* Weight Generation

        # Generate the weight vector
        argument_one = torch.arange(input_size)
        argument_two = torch.arange(output_size)

        # Generate the repeat
        argument_one = argument_one.repeat(seq_len, output_size)
        argument_two = argument_two.repeat(seq_len, input_size)

        x_concat = deepcopy(x).to(self.device)
        x_concat = x_concat.repeat(1, output_size).unsqueeze(2)

        # Positional Encoding + Concat
        argument_one = self.pos_enc(argument_one).to(self.device)
        argument_two = self.pos_enc(argument_two).to(self.device)
        argument = torch.concat((argument_one, argument_two, x_concat), dim=2)
        weights = self.weight_nn(argument).view(seq_len, input_size, output_size)
        x = x.view(seq_len, 1, input_size)
        out = torch.bmm(x, weights).squeeze(1)

        #* Bias Generation

        # Create Bias Argument
        argument_one = torch.arange(output_size)
        argument_one = self.pos_enc(argument_one).squeeze(1).to(self.device)
        argument_one = argument_one.repeat(seq_len, 1, 1)
        argument_two = out.unsqueeze(2)
        bias_argument = torch.concat((argument_one, argument_two), dim=2)

        # Add bias
        bias = self.bias_nn(bias_argument).squeeze(2)
        out += bias

        return out

    def forward (self, x):
        x = self.generate_weight_vector(x, self.first_input)
        x = self.dense_nn(x)
        return x

d_model = 32
weight_model = nn.Sequential(
    nn.Linear(33, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
) 

bias_model = nn.Sequential(
    nn.Linear(17, 10),
    nn.Tanh(),
    nn.Linear(10, 1)
)

dense_model = nn.Sequential(
    nn.Linear(6, 64),
    nn.Tanh(),
    nn.Linear(64,  10),
)
vnn = VNN(dense_model, weight_model, bias_model)

print(vnn(torch.randn(5,38)).to("cuda" if torch.cuda.is_available() else "cpu").shape)