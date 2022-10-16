import sys
sys.path.append("..\\")
from VNN import *
from torch import nn
from tqdm import trange
from random import randint
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader, Dataset
import torchvision.transforms as transforms
import PIL
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Helper functions
def return_gpu_desc ():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    return f"Free: {f/1024**2} MB; Allocated: {a/1024**2} MB"
# Dataset
mnist_dataset = MNIST(root="./datasets/", download=True)
mnist_test_dataset = MNIST(root="./datasets/", train=False, download=True)

class AddGaussianNoise(object):
    def __init__(self, std=1.):
        self.std = std
        print(self.std)
        
    def __call__(self, tensor):
        return torch.clamp(tensor + (torch.rand_like(tensor) * self.std), 0, 1)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(0, self.std)

class MNISTDatasetSize (Dataset):
    def __init__(self, size=512, mag=1, std_noise=0.25, use_test=False) -> None:
        super().__init__()

        self.mag = mag
        self.size = size
        self.std_noise = std_noise

        if use_test:
            left = len(mnist_test_dataset) - size
            self.dataset, _ = random_split(mnist_test_dataset, (size, left))
        else:
            left = len(mnist_dataset) - size
            self.dataset, _ = random_split(mnist_dataset, (size, left))

    def __len__ (self):
        return self.size

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        
        # Use transformations
        transformY = transforms.Compose([
            transforms.PILToTensor(),
        ]) 
        transformX = transforms.Compose([
            transforms.PILToTensor(),
            AddGaussianNoise(self.std_noise)
        ]) 

        if self.mag == 2:
            transformY = transforms.Compose([
                transforms.Resize((35, 35)),
                transforms.ToTensor(),
            ])
            transformX = transforms.Compose([
                transforms.Resize((35, 35)),
                transforms.ToTensor(),
                AddGaussianNoise(self.std_noise)
            ])
        elif self.mag == 3:
            transformY = transforms.Compose([
                transforms.Resize((45, 45)),
                transforms.ToTensor(),
            ])
            transformX = transforms.Compose([
                transforms.Resize((45, 45)),
                transforms.ToTensor(),
                AddGaussianNoise(self.std_noise)
            ])

        x = transformX(image).to(torch.float).to(device)
        y = transformY(image).to(torch.float).to(device)
        
        # Print image        
        # print(x.squeeze(0).numpy().shape)
        # x_img = PIL.Image.fromarray(np.uint8(x.squeeze(0).numpy()*255))
        # x_img = x_img.resize((500, 500), PIL.Image.NEAREST)
        # x_img.show() 
        # y_img = PIL.Image.fromarray(np.uint8(y.squeeze(0).numpy()*255))
        # y_img = y_img.resize((500, 500), PIL.Image.NEAREST)
        # y_img.show()
        # exit(0)

        return x, y

# Model
class AutoEncodingModel (nn.Module):
    def __init__(self) -> None:
        super().__init__()
        d_model = 16
        self.encoding_size = 64

        #* Output Block
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
        self.output_block = VNNBlock(d_model, output_weight_model, output_bias_model)

        #* Encoder & Decoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=3,              
                stride=3,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.Conv2d(
                in_channels=16,              
                out_channels=8,            
                kernel_size=3,              
                stride=3,                   
                padding=2,                  
            ),                              
            nn.ReLU(),  
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8,              
                out_channels=16,            
                kernel_size=3,              
                stride=3,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.ConvTranspose2d(
                in_channels=16,              
                out_channels=1,            
                kernel_size=3,              
                stride=3,                   
                padding=2,                  
            ),                              
            nn.ReLU(),
        )

    def encode (self, x):
        x = self.encoder(x)
        return x 

    def decode (self, x, orig_size):
        x = self.decoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.output_block(x, orig_size, chunks=6)
        return torch.sigmoid(x)

    def forward (self, x):
        orig_size = x.shape
        x = self.encode(x)
        x = self.decode(x, orig_size[2] * orig_size[3])
        return x.view(orig_size) 

# Initialize hyperparameters 
device = "cuda" if torch.cuda.is_available() else "cpu"
itr = 10_000
batch_size = 16
size_per_itr = 16
lr = 0.01
epochs = 3

# Declare model and optimizer
model = AutoEncodingModel().to(device) 
criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
progress_bar = trange(itr)
for i in progress_bar:
    mag = randint(1,3)
    dataset = MNISTDatasetSize(size_per_itr, mag=mag, std_noise=0.25) 
    dataset = DataLoader(dataset, batch_size=batch_size)
    for x, y in dataset: 
        # Train
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()

        # set progress bar
        progress_bar.set_description(f"Loss: {loss.item():.3f}")