import sys; sys.path.append("..\\")
from VNNTwo import *
from VNN import *
import torch
from torch import nn
from random import randint
from tqdm import trange

# =========================== Create Dataset ============================== 
min_length = 100
max_length = 500
dataset_size = 64
ones_occurance = 0.3
x_train = [] 
y_train = []
device = "cpu"

for i in range(dataset_size):
    length = randint(min_length, max_length)
    x = torch.zeros(length)
    indices_one = torch.randperm(length)[:math.ceil(ones_occurance*length)]
    x[indices_one] = 1
    x_train.append(x.unsqueeze(0)) 
    y_train.append(torch.tensor([i]))

# ======================== Training Paramaters ========================
use_original = True
itr = 1_000 
batch_size = 16
epochs = 5
mid_layer_size = 40
lr = 0.01

# ======================== Test Model ========================
class VNNBlockTwoModel (nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vnnBlock = VNNBlockTwo(d_model=64, kernel_size=3)
        self.linear_layer = nn.Sequential(
            nn.Linear(mid_layer_size, dataset_size),
        )
        self.tanh = nn.Tanh()

    def forward (self, x): 
        out, i_upscale, i_upscale_bias = self.vnnBlock(x, mid_layer_size, debug=True)
        out = self.tanh(out)
        return self.linear_layer(out), i_upscale, i_upscale_bias

# ======================== Original Model ========================
class VNNBlockModel (nn.Module):
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

        self.model = VNNBlock(d_model, weight_model, bias_model)

    def forward (self, x):
        out = self.model(x, dataset_size)
        return out

def train_epoch (use_original):
    # ======================== Model Setup ========================
    if use_original: 
        model = VNNBlockModel()
    else:
        model = VNNBlockTwoModel()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"============= Total Params: {sum(p.numel() for p in model.parameters())} =============")
    # ======================== Training Code ========================
    p = trange(itr)
    for _ in p:
        index_dataset = randint(0, dataset_size-1) 
        x = x_train[index_dataset]
        y = y_train[index_dataset] 
                
        # train
        optimizer.zero_grad()
        if not use_original:
            out, i_upscale, i_upscale_bias = model(x)
        else:
            out = model(x)
            i_upscale = -1
            i_upscale_bias = -1
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        p.set_description(f"Loss: {loss.item():.4f} weight upscale: {i_upscale} i_upscale_bias: {i_upscale_bias}")

    # ======================== Test Code ========================
    correct = 0
    for i in range(dataset_size):
        x = x_train[index_dataset]
        y = y_train[index_dataset]
        
        if not use_original:
            out, _, _ = model(x)
        else:
            out = model(x)
        out = torch.softmax(out.flatten(), dim=0)
        if torch.argmax(out,dim=0) == y[0].item():
            correct += 1
    percentage_acc = (correct/dataset_size)*100
    print(f"Test Validation Accuracy: {percentage_acc:.2f}%")


print("======================== USING ORIGINAL ========================")
train_epoch(use_original=True)
print("======================== USING NEW ========================")
train_epoch(use_original=False)