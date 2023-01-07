# Import libraries
from random import randint
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
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
class ImageDataset (Dataset):
    def __init__(self, res, size=512, use_test=False) -> None:
        super().__init__()

        self.res = res
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
        
        transform = transforms.Compose([
            transforms.Resize((self.res, self.res)),
            transforms.ToTensor(),
        ])

        x = transform(x).to(torch.float).to(device)
        y = torch.tensor(y).to(torch.long).to(device)
        return x, y

    @staticmethod
    def getTestImageRandSize (i: int, res:int):
        x, y = mnist_test_dataset[i]

        transform = transforms.Compose([
            transforms.Resize((res, res)),
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
def validation (model, min_res, max_res):
    confusionMatrix = torch.zeros(10, 10)
    correct = 0
    p = trange(10_000)
    for i in p:
        res = randint(min_res, max_res) 
        x, y = ImageDataset.getTestImageRandSize(i, res)

        y = y.item()
        output = torch.softmax(model(x.unsqueeze(0).to(device))[0],1)
        argmax = torch.argmax(output).item()
        if argmax == y: correct += 1
        confusionMatrix[y][argmax] += 1
        p.set_description(f"Res: {res:3d} Acc: {(correct/(i+1))*100:.1f}%")

    return (correct/100), confusionMatrix

# ======================================= Training Loop ===========================================  
def train (model, name):
    losses = [] # Loss array for graphing

    batch_size = 16
    num_samples_per_itr = 4
    min_res = 28
    max_res = 128
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
    
    for _ in progress_bar:
        # Get data
        res = randint(min_res, max_res)

        dataset = DataLoader(
            ImageDataset(
                res=res,
                size=batch_size*num_samples_per_itr,
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

            progress_bar.set_description(f"loss: {loss.item():.4f} w_ups: {i_upscale:2d} b_ups: {i_upscale_bias:2d} res: {res:3d}")

            avg_loss += loss.item()
        avg_loss /= len(dataset)
        losses.append(avg_loss) 

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
    acc, confusionMatrix = validation(model=model, min_res=min_res, max_res=max_res)
    
    # Print Confusion matrix
    print("======== Confusion Matrix: ======== ")
    print(confusionMatrix)
    print("=================================== ")

    # Return Everything
    return losses, {saved_name: { "Accuracy": acc, "Training Time elapsed time (seconds)": elapsed_time, "Total Params:" : total_params, "Confusion Matrix": str(confusionMatrix.tolist())}}

# ======================================= Main Loop =========================================== 
if __name__ == "__main__":
    trainers = {
        "VNN Model v3": VNNModelV3(),
        "VNN Model v2": VNNModelV2()
    }
    
    dir_path = os.path.join(os.getcwd(), "data")
    if not os.path.isdir(dir_path): os.mkdir(dir_path)
    overall_dict = {}
    for name in trainers.keys():
        losses, result = train(trainers[name], name)

        plt.figure()
        plt.plot(losses, label = name)
        overall_dict.update(result)

        # Save loss data and json data
        saved_name = name.replace(" ", "_").lower()
        plt.legend()
        plt.savefig(os.path.join(dir_path, f"loss_plot_{saved_name}.png"))

    # Save overall dict
    with open(os.path.join(dir_path, "data.json"), "w") as f:
        json.dump(overall_dict, f) 
    print("Saved data")