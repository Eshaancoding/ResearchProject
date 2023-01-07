from random import randint
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import trange, tqdm
import sys; sys.path.append("../") 
from VNN import *
from VNNv2 import *
from VNNv3 import *
import time
import os
import json
import matplotlib.pyplot as plt


# Get Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================= Vector Argmax Benchmark ===========================================  
class VectorArgmaxBenchmark (Dataset):
    def __init__(self, width, height, max_not_one) -> None:
        super().__init__()
        self.len = height

        # Generate dataset
        self.x = torch.rand(height, width).to(device)*max_not_one
        self.y = torch.zeros(height, width).to(device)
        exp_out = []
        for i in range(height):
            rand_num = randint(0, width-1)
            self.x[i][rand_num] = 1
            self.y[i][rand_num] = 1
            exp_out.append(rand_num)
        self.exp_out = torch.tensor(exp_out).to(device)

    def __len__ (self):
        return self.len 

    def __getitem__(self, index):
        return self.x[index], self.exp_out[index]
        
    
# ======================================= VNN V1 Model ===========================================  
class VNNV1Model (nn.Module):
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
        self.to(device)

    def forward (self, x): 
        x = x.view(x.size(0), -1)
        length = x.size(1)
        x = self.vnnBlock(x, length)
        return x, -1, -1
        
# ======================================= VNN V2 Model ===========================================  
class VNNV2Model (nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = VNNBlockTwo(d_model=64, initial_size=10, kernel_size=5, device=device)
        self.to(device)

    def forward (self, x): 
        x = x.view(x.size(0), -1)
        length = x.size(1)
        x, i_upscale, i_upscale_bias = self.encoder(x, length, debug=True)
        return x, i_upscale, i_upscale_bias

# ======================================= VNN V2 Model ===========================================  
class VNNV3Model (nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vnnBlock = VNNv3(d_model=64, input_kernel_size=50, output_kernel_size=50, hidden_size=25, device=device)
        self.to(device)

    def forward (self, x): 
        x = x.view(x.size(0), -1)
        length = x.size(1)

        x, gen_x, gen_y = self.vnnBlock(x, length, True)
        return x, gen_y, gen_x

# ======================================= Validation Loop ===========================================  
def validation (mag, model):
    dataset = VectorArgmaxBenchmark()
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
    losses = [] # Loss array for graphing

    max_not_one = 0.6
    batch_size = 16
    num_samples_per_itr = 8
    min_length = 100
    max_length = 300
    lr = 0.01
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
        dataset = DataLoader(
            VectorArgmaxBenchmark(
                width=100,
                height=batch_size*num_samples_per_itr,
                max_not_one=max_not_one
            ),
            batch_size=batch_size,
        )

        # Train on that data
        avg_loss = 0
        for x, y in dataset:
            opt.zero_grad()
            out, i_upscale, i_upscale_bias = model(x) 
            loss = criterion(out, y)
            loss.backward()
            opt.step()

            # calculate accuracy
            acc = (torch.sum(torch.argmax(out, dim=1) == y)/y.size(0)).item()
            progress_bar.set_description(f"loss: {loss.item():.4f} w_ups: {i_upscale} b_ups: {i_upscale_bias} val acc: {(acc*100):.1f}%")

            avg_loss += loss.item()
        avg_loss /= len(dataset)
        if avg_loss < 5: # For the sake of generating a clean loss data 
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
    acc = validation()

    # Return Everything
    return losses_one, {saved_name: { "Accuracy": acc, "Training Time elapsed time (seconds)": elapsed_time, "Total Params:" : total_params}}

# ======================================= Main Loop =========================================== 
if __name__ == "__main__":
    # NOTE: Changed loss, so you might need to change activation function
    trainers = {
        # "VNN Model v2": VNNV2Model(),
        # "VNN Model v3": VNNV3Model(),
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