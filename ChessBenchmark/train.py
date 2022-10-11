import sys; sys.path.append("..\\")
from VNN import *
import torch
from torch import nn
import torch.nn.functional as F
import chess
from tqdm import trange
import linecache
from random import randint

input_database = ".\\chessDB.txt"
output_model = ".\\model\\ChessEngine.pth"

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
    return_tensor = torch.zeros(1,13,8,8)
    for char in board.__str__():
        if char == " ": continue
        if char == "r":   return_tensor[0][0][x][y] = 1
        elif char == "n": return_tensor[0][1][x][y] = 1
        elif char == "b": return_tensor[0][2][x][y] = 1
        elif char == "q": return_tensor[0][3][x][y] = 1
        elif char == "k": return_tensor[0][4][x][y] = 1
        elif char == "k": return_tensor[0][5][x][y] = 1
        elif char == "P": return_tensor[0][6][x][y] = 1
        elif char == "R": return_tensor[0][7][x][y] = 1
        elif char == "N": return_tensor[0][8][x][y] = 1
        elif char == "B": return_tensor[0][9][x][y] = 1
        elif char == "Q": return_tensor[0][10][x][y] = 1
        elif char == "K": return_tensor[0][11][x][y] = 1
        if char == "p":   return_tensor[0][12][x][y] = 1

        x += 1
        if char == "\n": 
            y += 1
            x = 0
    return return_tensor

class ChessClassificationDatabase(torch.utils.data.Dataset):
    def __init__(self, num_games):
        assert num_games > 0
        self.x = torch.tensor([]) 
        self.y = torch.tensor([])
        self.possible_moves = []

        len_lines = 3561469
        i = 0
        while i < num_games:
            try:
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
                    legal_moves = board.legal_moves
                    for move in legal_moves:
                        move_enc = encode_move(move)
                        if possible_move.size(0) == 0: possible_move = move_enc
                        else: possible_move = torch.concat((possible_move, move_enc), dim=0)

                    # Append to variables
                    self.possible_moves.append(possible_move)
                    y_enc = list(legal_moves).index(actual_move)
                    y_enc = torch.tensor([[y_enc]])
                    if self.x.size(0) == 0:
                        self.x = tensor_board
                        self.y = y_enc 
                    else:
                        self.x = torch.vstack((self.x, tensor_board))
                        self.y = torch.vstack((self.y, y_enc))

                    board.push(actual_move)
                    i += 1
            except:
                continue
        self.x.to(device)
        self.y.to(device)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, index):
        return self.x[index].detach(), self.y[index].detach(), self.possible_moves[index].to(device).detach()

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
            nn.Linear(65, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        ) 

        bias_model = nn.Sequential(
            nn.Linear(49, 12),
            nn.Tanh(),
            nn.Linear(12, 1),
        )

        self.model = VNNBlock(d_model, weight_model, bias_model)

    def forward (self, x, possible_moves):
        x = self.neuralNet(x)
        return self.model(x, possible_moves.size(1), possible_moves)

if __name__ == "__main__":
    # Optimizer and training parameters
    batch_size = 16
    validation_size = 64
    num_games_per_itr = 4
    lr = 0.001
    itr = 10_000

    criterion = nn.CrossEntropyLoss()
    nn = PolicyNeuralNetwork().to(device)
    opt = torch.optim.Adam(nn.parameters(), lr=lr)

    # Training Loop
    progress_bar = trange(itr)
    itr = 0
    mag = 1
    for i in progress_bar:
        # Get data
        dataset = ChessClassificationDatabase(num_games=num_games_per_itr),

        # Train on that data
        opt.zero_grad()
        losses = torch.tensor([])
        for data in dataset:
            x = data[0][0].unsqueeze(0)
            y = data[0][1]
            possible_moves = data[0][2].unsqueeze(0)
            out = nn(x, possible_moves) 
            data_loss = criterion(out, y)
            data_loss = data_loss.unsqueeze(0)
            if losses.size(0) == 0: losses = data_loss
            else: losses = torch.concat((losses, data_loss), 0)
        loss = torch.mean(losses)
        loss.backward()
        opt.step()

        progress_bar.set_description(f"Loss: {loss.item():.4f}")

        if i % 100 == 0 and i != 0:  
            # Save Model
            torch.save(nn, output_model)