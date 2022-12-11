# Import libraries
from random import randint
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import trange, tqdm
import sys; sys.path.append("../")
from NN.VNNv3 import *
from NN.VNNv2 import *
from NN.VNN import *
import time
import os
import json
import matplotlib.pyplot as plt

# Get Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================= Image Recognition Benchmark ===========================================  
mnist_dataset = MNIST(root="./datasets/", download=True)
mnist_test_dataset = MNIST(root="./datasets/", train=False, download=True)
class MNISTDatasetSize (Dataset):
    def __init__(self, size=512, mag=1, use_test=False) -> None:
        super().__init__()

        self.mag = mag
        self.size = size

        if use_test:
            if size == "full":
                self.dataset = mnist_test_dataset
            else:
                left = len(mnist_test_dataset) - size
                self.dataset, _ = random_split(mnist_test_dataset, (size, left))
        else:
            if size == "full":
                self.dataset = mnist_dataset
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

# ======================================= Convolution Neural Network ===========================================  
class ConvolutionNN(nn.Module):
    def __init__(self):
        super(ConvolutionNN, self).__init__()
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

# ======================================= VNN V1 ===========================================  
class VNNModelV1 (nn.Module):
    def __init__(self) -> None:
        super().__init__()
        d_model = 64
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

        self.vnnBlock = VNNBlock(d_model, weight_model, bias_model, device=device)
        self.conv2d = ConvolutionNN()
        self.to(device)

    def forward (self, x): 
        x = self.conv2d(x)
        x = x.view(x.size(0), -1)
        x = self.vnnBlock(x, 10) 
        return x, -1, -1

# ======================================= VNN Model V2 ===========================================  
class VNNModelV2 (nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vnnBlock = VNNBlockTwo(d_model=64, initial_size=10, kernel_size=5, device=device)
        self.conv2d = ConvolutionNN()
        self.to(device)

    def forward (self, x): 
        x = self.conv2d(x)
        x = x.view(x.size(0), -1)
        x, i_upscale, i_upscale_bias = self.vnnBlock(x, 10, debug=True) 
        return x, i_upscale, i_upscale_bias

# ======================================= VNN Model V3 ===========================================  
class VNNModelV3 (nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vnnBlock = VNNv3(d_model=64, input_kernel_size=200, output_kernel_size=10, hidden_size=64, device=device)
        self.conv2d = ConvolutionNN()
        self.to(device)

    def forward (self, x): 
        x = self.conv2d(x)
        x = x.view(x.size(0), -1)
        x, gen_x, gen_y = self.vnnBlock(x, 10, True) 
        return x, gen_y, gen_x

# ======================================= Validation Loop ===========================================  
def validation (mag, model):
    dataset = MNISTDatasetSize(size="full", mag=mag, use_test=True)
    correct = 0
    itr = 0
    p = tqdm(dataset, total=10000)
    for x, y in p:
        itr += 1
        y = y.item()
        output = torch.softmax(model(x.unsqueeze(0).to(device))[0],1)
        argmax = torch.argmax(output).item()
        if argmax == y: correct += 1
        p.set_description(f"Mag: {mag} Acc: {(correct/itr)*100:.1f}%")
    return (correct/itr)*100

# ======================================= Training Loop ===========================================  
def train (model, name):
    losses_one = [] # Loss array for graphing
    losses_two = [] # Loss array for graphing
    losses_three = [] # Loss array for graphing

    batch_size = 16
    num_samples_per_itr = 4
    lr = 0.001
    itr = 3_000

    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Print number of total params
    print(f"============================ Training {name} ============================")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params}")
    
    # Training Loop
    start = time.time()
    progress_bar = trange(itr)
    itr = 0
    mag = 1
    
    for _ in progress_bar:
        # Get data
        mag = randint(1,3)

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
        avg_loss = 0
        for x, y in dataset:
            opt.zero_grad()
            out, i_upscale, i_upscale_bias = model(x) 
            loss = criterion(out, y)
            loss.backward()
            opt.step()

            progress_bar.set_description(f"loss: {loss.item():.4f} w_ups: {i_upscale} b_ups: {i_upscale_bias} mag: {mag}")

            avg_loss += loss.item()
        avg_loss /= len(dataset)
        if avg_loss < 5: # For the sake of actually generating a clean loss data 
            if mag == 1: losses_one.append(avg_loss)
            if mag == 2: losses_two.append(avg_loss)
            if mag == 3: losses_three.append(avg_loss)

    # End time
    elapsed_time = time.time() - start
    
    # Save Model
    saved_name = name.replace(" ", "_").lower()
    dir_path = os.path.join(os.getcwd(), "models")
    file_path = os.path.join(dir_path, f"{saved_name}.pt") 
    if not os.path.isdir(dir_path): os.mkdir(dir_path)
    torch.save(model, file_path)
    print("Saved model")

    # Test the model
    print(f"============================ Validating {name} ============================")
    acc_1 = validation(mag=1, model=model)
    acc_2 = validation(mag=2, model=model)
    acc_3 = validation(mag=3, model=model)
    avg_acc = (acc_1 + acc_2 + acc_3) / 3

    # Return Everything
    return losses_one, losses_two, losses_three, {saved_name: { "Magnitude 1 Accuracy": acc_1, "Magnitude 2 Accuracy": acc_2, "Magnitude 3 Accuracy": acc_3, "Average Accuracy": avg_acc, "Training Time elapsed time (seconds)": elapsed_time, "Total Params:" : total_params}}

# ======================================= Main Loop =========================================== 
if __name__ == "__main__":
    trainers = {
        "VNN Model v3": VNNModelV3()
    }
    
    dir_path = os.path.join(os.getcwd(), "data")
    if not os.path.isdir(dir_path): os.mkdir(dir_path)
    overall_dict = {}
    for name in trainers.keys():
        losses_one, losses_two, losses_three, result = train(trainers[name], name)

        plt.figure()
        plt.plot(losses_one, label = f"{name} Mag 1")
        plt.plot(losses_two, label = f"{name} Mag 2")
        plt.plot(losses_three, label = f"{name} Mag 3")
        overall_dict.update(result)

        # Save loss data and json data
        saved_name = name.replace(" ", "_").lower()
        plt.legend()
        plt.savefig(os.path.join(dir_path, f"loss_plot_{saved_name}.png"))

    # Save overall dict
    with open(os.path.join(dir_path, "data.json"), "w") as f:
        json.dump(overall_dict, f) 
    print("Saved data")