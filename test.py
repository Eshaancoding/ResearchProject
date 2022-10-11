from VNN import * 
import time

d_model = 16
weight_model = nn.Sequential(
    nn.Linear(43, 16),
    nn.Tanh(),
    nn.Linear(16, 1),
) 

bias_model = nn.Sequential(
    nn.Linear(27, 12),
    nn.Tanh(),
    nn.Linear(12, 1),
)

vnn_block = VNNBlock(weight_model, bias_model, d_model)

input = torch.rand(6, 5)

extra_out = torch.rand(6, 10, 10)

start = time.time()
print(vnn_block(input, extra_out=extra_out, output_size=10, chunks=1).shape)
end = time.time()
print(end-start)