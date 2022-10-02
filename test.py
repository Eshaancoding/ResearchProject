from VNN import * 

d_model = 16
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

vnn_block = VNNBlock(weight_model, bias_model)

input = torch.rand(1, 5)
print(vnn_block(input, 2))