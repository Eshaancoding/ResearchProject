import sys; sys.path.append("..\\")
from VNN import *
from torch import nn
from tqdm import trange
from random import randint

# Model
class AutoEncodingModel (nn.Module):
    def __init__(self) -> None:
        super().__init__()
        d_model = 16
        self.encoding_size = 64
        self.input_size = 256

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
        self.input_block = VNNBlock(weight_model, bias_model)

        output_weight_model = nn.Sequential(
            nn.Linear(d_model*2+1, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        ) 
        output_bias_model = nn.Sequential(
            nn.Linear(d_model+1, 12),
            nn.Tanh(),
            nn.Linear(12, 1),
        )
        self.output_block = VNNBlock(output_weight_model, output_bias_model)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.Sigmoid(),
            nn.Linear(128, self.encoding_size),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_size, 128),
            nn.Sigmoid(),
            nn.Linear(128, self.input_size),
            nn.Sigmoid(),
        )

    def forward (self, x):
        len_seq = x.size(1)
        x = self.input_block(x, self.input_size)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_block(x, len_seq)
        return x

    def encode (self, x):
        x = self.input_block(x, self.input_size)
        x = self.encoder(x)
        return x 

    def decode (self, x, len_size):
        x = self.decoder(x)
        x = self.output_block(x, len_size)
        return x

# Initialize hyperparameters 
device = "cuda" if torch.cuda.is_available() else "cpu"
itr = 10_000
batch_size = 32
low_bound_arr_size = 256
high_bound_arr_size = 512
lr = 0.01

# Declare model and optimizer
model = AutoEncodingModel().to(device) 
criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
progress_bar = trange(itr)
for _ in progress_bar:
    # get batch data
    len_seq = randint(low_bound_arr_size, high_bound_arr_size)
    x = torch.rand(batch_size, len_seq).to(device)
    
    # Train
    opt.zero_grad()
    out = model(x)
    loss = criterion(out, x)
    loss.backward()
    opt.step()

    # set progress bar
    progress_bar.set_description(f"Loss: {loss.item():.3f}")