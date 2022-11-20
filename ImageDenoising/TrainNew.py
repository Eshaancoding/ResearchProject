# Import libraries
from random import randint
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import trange, tqdm
import sys; sys.path.append("..\\")
from VNNTwo import *
from VNN import *
import time
import os
import json
import matplotlib.pyplot as plt

# Get Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================= MNIST dataset ===========================================  
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
# ======================================= New VNN Model ===========================================  
class NewVNNModel (nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = 30
        self.encoder = VNNBlockTwo(d_model=16, initial_size=10, kernel_size=8, device=device)
        self.decoder = VNNBlockTwo(d_model=16, initial_size=10, kernel_size=8, device=device)
        self.nn_mid = nn.Linear(self.inner, self.inner) 
        self.sigmoid = nn.Sigmoid()
        self.to(device)

    def forward (self, x): 
        x = x.view(x.size(0), -1)
        length = x.size(1)
        x = self.sigmoid(self.encoder(x, self.inner))
        x = self.sigmoid(self.nn_mid(x))
        x, i_upscale, i_upscale_bias = self.decoder(x, length, debug=True) 
        return x, i_upscale, i_upscale_bias


# ======================================= Original VNN Model ===========================================  
class OrigVNNModel (nn.Module):
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

# ======================================= Transformers Model ===========================================  
class TransformersModel (nn.Module):
    def __init__(self) -> None:
        super().__init__()
        d_model = 25
        self.in_nn = nn.Linear(1, d_model)
        self.out_nn = nn.Linear(d_model, 10)
        self.lstm_model = nn.LSTM(input_size=d_model, hidden_size=d_model)
        
        self.to(device)

    def forward (self, x): 
        x = self.conv2d(x)
        x = x.view(-1, x.size(0), 1)
        x = self.in_nn(x)
        x = self.lstm_model(x)[0][-1]
        x = self.out_nn(x)
        return x, -1, -1

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
                width=randint(min_length, max_length),
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

            progress_bar.set_description(f"loss: {loss.item():.4f} w_ups: {i_upscale} b_ups: {i_upscale_bias}")

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
    trainers = {
        "New VNN Model": NewVNNModel(),
        "Original VNN Model": OrigVNNModel(),
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