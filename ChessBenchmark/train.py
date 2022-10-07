import sys; sys.path.append("..\\")
from VNN import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import chess
from tqdm import tqdm, trange
import linecache
from random import randint
from time import sleep

input_database = ".\\chessDB.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
# line cache warmup
linecache.getline(input_database, 0) 

def encode_move (move:chess.Move):
    char_to_num = {
        "a":0,
        "b":1,
        "c":2,
        "d":3,
        "e":4,
        "f":5,
        "g":6,
        "h":7,
    }

    uci_str = move.uci()
    first_num = char_to_num[uci_str[0]]
    second_num = int(uci_str[1])-1
    third_num = char_to_num[uci_str[2]]
    fourth_num = int(uci_str[3])-1
    
    return torch.concat((
        F.one_hot(torch.tensor([first_num]), num_classes=8),
        F.one_hot(torch.tensor([second_num]), num_classes=8),
        F.one_hot(torch.tensor([third_num]), num_classes=8),
        F.one_hot(torch.tensor([fourth_num]), num_classes=8),
    ), dim=1).to(torch.float)

def decode_move (x:torch.tensor):
    num_to_char = {
        0:"a",
        1:"b",
        2:"c",
        3:"d",
        4:"e",
        5:"f",
        6:"g",
        7:"h",
    }
    x = x.view(-1, 8)
    x = torch.argmax(x, dim=1).cpu().tolist()
    first_char = num_to_char[x[0]] 
    second_char = str(x[1]+1) 
    third_char = num_to_char[x[2]] 
    fourth_char = str(x[3]+1) 
    return chess.Move.from_uci(first_char+second_char+third_char+fourth_char)

def encode_board (board):
    x = 0
    y = 0
    return_tensor = torch.zeros(1,8,8,13)
    for char in board.__str__():
        if char == " ": continue
        if char == "r":   return_tensor[0][x][y][0] = 1
        elif char == "n": return_tensor[0][x][y][1] = 1
        elif char == "b": return_tensor[0][x][y][2] = 1
        elif char == "q": return_tensor[0][x][y][3] = 1
        elif char == "k": return_tensor[0][x][y][4] = 1
        elif char == "k": return_tensor[0][x][y][5] = 1
        elif char == "P": return_tensor[0][x][y][6] = 1
        elif char == "R": return_tensor[0][x][y][7] = 1
        elif char == "N": return_tensor[0][x][y][8] = 1
        elif char == "B": return_tensor[0][x][y][9] = 1
        elif char == "Q": return_tensor[0][x][y][10] = 1
        elif char == "K": return_tensor[0][x][y][11] = 1
        if char == "p":   return_tensor[0][x][y][12] = 1

        x += 1
        if char == "\n": 
            y += 1
            x = 0
    return return_tensor

class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_games):
        assert num_games > 0
        self.x = torch.tensor([]) 
        self.y = torch.tensor([])
        self.possible_moves = [] 

        len_lines = 3561469
        for i in range(num_games):
            line = linecache.getline(".\\chessDB.txt", randint(0, len_lines)+6) 
            board = chess.Board()

            line = line.split("###")[1].strip()
            moves = line.split(" ")

            for move in moves:
                move = move.split(".")[1]
                tensor_board = encode_board(board)
                actual_move = board.parse_san(move)
                possible_move = torch.tensor([])

                # Encode possible moves
                for move in board.legal_moves:
                    move_enc = encode_move(move)
                    if possible_move.size(0) == 0: possible_move = move_enc
                    else: possible_move = torch.concat((possible_move, move_enc), dim=0)

                # Append to variables
                self.possible_moves.append(possible_move)
                if self.x.size(0) == 0:
                    self.x = tensor_board
                    self.y = encode_move(actual_move)
                else:
                    self.x = torch.vstack((self.x, tensor_board))
                    self.y = torch.vstack((self.y, encode_move(actual_move)))

                board.push(actual_move)

        self.x.to(device)
        self.y.to(device)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.possible_moves[index].to(device)

class PolicyNeuralNetwork (nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.neuralNet = nn.Sequential(
            nn.Conv2d(13, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.Tanh(),
            nn.LazyLinear(64),
        )

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

        self.model = VNNBlock(weight_model, bias_model)

    def forward (self, x, len_moves):
        x = self.neuralNet(x)
        return self.model(x, len_moves)

dataset = Dataset(1)
x, y, possible_move = dataset[20]
print(x.shape)
print(y.shape)
print(possible_move.shape)

for i in possible_move:
    print("Possible move:", decode_move(i))
print("actual move:", decode_move(y))