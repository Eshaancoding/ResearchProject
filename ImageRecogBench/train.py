# Import libraries
from random import randint
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import trange
import sys; sys.path.append("..\\")
from VNN import *

# Get Device
device = 'cuda' if torch.cuda.is_available() else "cpu"

# Get Dataset
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

# Get Control Policy
class Policy(nn.Module):
    def __init__(self, use_VNN=False):
        super(Policy, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, 
                kernel_size=5, 
                stride=1,
                padding=2),     
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),                
        )

        self.conv3 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32, 
                out_channels=32, 
                kernel_size=5, 
                stride=1,
                padding=2),     
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),                
        )

        self.use_VNN = use_VNN
        if use_VNN:
            # VNN
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

            self.out = VNNBlock(weight_model, bias_model)
        else:
            # fully connected layer, output 10 classes
            self.out = nn.LazyLinear(10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        if self.use_VNN: output = self.out(x, 10)
        else: output = self.out(x)
        return output

if __name__ == "__main__":
    # Optimizer and training parameters
    use_VNN = True

    batch_size = 16
    validation_size = 64
    num_samples_per_itr = 4
    lr = 0.001
    itr = 5_000
    test_mag = 1

    criterion = nn.CrossEntropyLoss()
    policy = Policy(use_VNN).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    # Training Loop
    progress_bar = trange(itr)
    itr = 0
    mag = 1
    for i in progress_bar:
        # Get data
        if use_VNN: 
            mag = randint(1,3)
        else: mag = test_mag

        dataset = DataLoader(
            MNISTDatasetSize(
                size=batch_size*num_samples_per_itr,
                mag=mag, 
                use_test=False
            ),
            batch_size=batch_size,
            shuffle=True
        )

        # Train on that data
        for x, y in dataset:
            opt.zero_grad()
            out = policy(x) 
            loss = criterion(out, y)
            loss.backward()
            opt.step()

            progress_bar.set_description(f"Loss: {loss.item():.4f} Mag: {mag}")
        i+=1

    # Save Model
    torch.save(policy, "..\models\ImageRecogModel.pt")