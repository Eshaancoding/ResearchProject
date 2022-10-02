import sys; sys.path.append("..\\")
from tkinter import W
from torch import nn
from VNN import *
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

control = False

if control: 
    pass
else:
    #* VNN Experiment
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

    model = VNNBlock(weight_model, bias_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

# Training
itr = 1_000
batch_size = 32
epochs = 10

for epoch in range(epochs):
    p = trange(itr)
    for _ in p:
        length = randint(1,5)
        x = torch.rand(4,length).to(torch.float)
        exp_out = torch.argmax(x, dim=1)

        optimizer.zero_grad()
        out = model(x, length)
        loss = criterion(out, exp_out)
        loss.backward()
        optimizer.step()

        p.set_description(f"Epoch: {epoch+1} loss: {loss.item():.4f}")
    
    # Test
    with torch.no_grad():
        x = torch.rand(1,5).to(torch.float)
        out = model(x, 5)
        exp_out = torch.argmax(x, dim=1)

        loss = criterion(out, exp_out)
        print(f"Test Validation with loss: {loss.item():.4f}")
        print(f"Input/Expected Output:\n{x}")
        print(f"NN Out:\n{torch.softmax(out, dim=1)}")