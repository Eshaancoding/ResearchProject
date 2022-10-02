import sys
from this import s; sys.path.append("..\\")
from VNN import *
from torch import nn
from tqdm import trange
from random import randint
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader, Dataset
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

s

# Dataset
mnist_dataset = MNIST(root="./datasets/", download=True)
mnist_test_dataset = MNIST(root="./datasets/", train=False, download=True)
class MNISTDatasetSize (Dataset):
    def __init__(self, size=512, mag=1, use_test=False) -> None:
        super().__init__()

        self.mag = mag
        self.size = size

        if use_test:
            left = len(mnist_test_dataset) - size
            self.dataset, _ = random_split(mnist_test_dataset, (size, left))
        else:
            left = len(mnist_dataset) - size
            self.dataset, _ = random_split(mnist_dataset, (size, left))

    def __len__ (self):
        return self.size

    def __getitem__(self, index):
        x, y = self.dataset[index]
        
        # Use transformations
        transform = transforms.Compose([
            transforms.PILToTensor(),
        ]) 

        if self.mag == 2:
            transform = transforms.Compose([
                transforms.Resize((35, 35)),
                transforms.ToTensor(),
            ])
        elif self.mag == 3:
            transform = transforms.Compose([
                transforms.Resize((45, 45)),
                transforms.ToTensor(),
            ])

        x = transform(x).to(torch.float).to(device)
        y = torch.tensor(y).to(torch.long).to(device)
        return x, y

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