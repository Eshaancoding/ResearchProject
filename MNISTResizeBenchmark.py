# Import libraries
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import trange

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
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
            ])
        elif self.mag == 3:
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])

        x = transform(x).to(torch.float)
        y = torch.tensor(y).to(torch.long)
        return x, y

# Get Control Policy
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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

        # fully connected layer, output 10 classes
        self.out = nn.LazyLinear(10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output

# Optimizer and training parameters
batch_size = 16
validation_size = 64
num_samples_per_itr = 4
lr = 0.001
itr = 1_000
test_mag = 2

criterion = nn.CrossEntropyLoss()
policy = CNN()
opt = torch.optim.Adam(policy.parameters(), lr=lr)

# Training Loop
progress_bar = trange(itr)
for i in progress_bar:
    # Get data
    dataset = DataLoader(
        MNISTDatasetSize(
            size=num_samples_per_itr * batch_size,
            mag=test_mag, 
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
        
        progress_bar.set_description(f"Loss: {loss.item():.4f}")

# Testing Loop
test_dataset = MNISTDatasetSize(
    size=validation_size,
    mag=test_mag,
    use_test=True
) 
num_correct = 0
for x, y in test_dataset:
    x = x.unsqueeze(0)
    out = torch.argmax(policy(x), 1)
    if (out == y): 
        num_correct += 1

print(f"Validation Test Accuracy: {(num_correct/validation_size)*100:.1f}%")