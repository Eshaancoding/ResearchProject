import torch.nn.functional as F
import sys; sys.path.append("..\\")
from tkinter import W
from torch import nn
from VNN import *
from VNNTwo import *
import torch
from random import randint
from tqdm import trange

def print_var (var, name=""):
    print()
    str = f"----------------{name}----------------"
    print(str)
    print(var)
    print("".join(["-" for _ in range(len(str))]))
    print()

improved = True
device = "cuda"

if not improved: 
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

    model = VNNBlock(d_model, weight_model, bias_model, device=device)
else:
    model = VNNBlockTwo(initial_size=50, kernel_size=10, device=device)
    
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training Params
itr = 1_000 
batch_size = 16
epochs = 5
min_length = 200
max_length = 500

# Training code
for epoch in range(epochs):
    p = trange(itr)
    for _ in p:
        length = randint(min_length, max_length)
        x = torch.rand(batch_size, length)/3
        exp_out = []
        for i in range(batch_size):
            select_index = randint(0, length-1) 
            x[i][select_index] = 1
            exp_out.append(select_index)
        exp_out = torch.tensor(exp_out).to(device)

        # train
        optimizer.zero_grad()
        out = model(x, length)
        loss = criterion(out, exp_out.detach())
        loss.backward()
        optimizer.step()

        p.set_description(f"Epoch: {epoch+1} loss: {loss.item():.4f}")

    # Test
    with torch.no_grad():
        length = 10
        x = torch.rand(1, length)/3
        exp_out = []
        select_index = randint(0, length-1) 
        x[0][select_index] = 1
        exp_out.append(select_index)
        exp_out = torch.tensor(exp_out).to(device)

        # train
        optimizer.zero_grad()
        out = model(x, length)
        loss = criterion(out, exp_out.detach())

        print("\n================================================")
        print(f"X: {x}")
        print(f"Out: {out}")
        print(f"Loss: {loss}")
        print("================================================\n")