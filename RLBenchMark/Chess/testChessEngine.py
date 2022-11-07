import chess
import torch
from torch import nn, Tensor
import math
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

class PosEncIndex(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, x: Tensor) -> Tensor:
        length = torch.max(x).item()+1
        
        pe = torch.zeros((length, self.d_model)).to(device)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / self.d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe[x]

class VNNBlock (nn.Module):
    def __init__(self, d_model, weight_nn, bias_nn) -> None:
        super().__init__()
        self.weight_nn = weight_nn 
        self.bias_nn = bias_nn 
        self.pos_enc = PosEncIndex(d_model)

    def weight_propagation (self, x, output_size, extra_out):
        input_size = x.size(1)
        batch_size = x.size(0)
        
        #* Weight Generation
        # Generate the weight vector
        argument_one = torch.arange(input_size)
        argument_two = torch.arange(output_size)

        # Generate the repeat
        argument_one = argument_one.repeat(batch_size, output_size)
        argument_two = argument_two.repeat(batch_size, input_size)

        x_concat = x.repeat(1, output_size).unsqueeze(2).to(device)

        # Positional Encoding + Concat
        argument_one = self.pos_enc(argument_one.detach())
        argument_two = self.pos_enc(argument_two.detach())

        if extra_out != None:  
            argument = torch.concat((argument_one, argument_two, x_concat, extra_out.repeat(1, input_size, 1)), dim=2)
        else:
            argument = torch.concat((argument_one, argument_two, x_concat), dim=2)
        
        weights = self.weight_nn(argument.detach()).view(batch_size, input_size, output_size)
        x = x.view(batch_size, 1, input_size)
        out = torch.bmm(x, weights).squeeze(1)

        #* Bias Generation

        # Create Bias Argument
        argument_one = torch.arange(output_size)
        argument_one = self.pos_enc(argument_one.detach()).squeeze(1)
        argument_one = argument_one.repeat(batch_size, 1, 1)
        argument_two = out.unsqueeze(2)
        if extra_out != None:
            bias_argument = torch.concat((argument_one, argument_two, extra_out), dim=2)
        else:
            bias_argument = torch.concat((argument_one, argument_two), dim=2)

        # Add bias
        bias = self.bias_nn(bias_argument.detach()).squeeze(2)
        out += bias

        return out

    def return_gpu_desc (self):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        return f"Free: {f/1024**2} MB; Allocated: {a/1024**2} MB"

    def forward (self, x, output_size, extra_out=None, chunks=None):
        # Extra Out size: 
        # first dim is the batch size, second dim is the output space, third dim is the vector added during weight 
        if chunks != None:
            if extra_out != None:
                assert x.size(0) == extra_out.size(0), f"Batch size of x ({x.size(0)}) is the same as the batch size of extra_out ({extra_out.size(0)})"

            arr = [output_size // chunks for _ in range(chunks)]        
            if output_size % chunks > 0: 
                arr.append(output_size % chunks) 

            out = torch.tensor([])
            output_size = 5

            index = 0
            for i in range(len(arr)):
                indx_arr = torch.arange(start=index, end=index+arr[i]).to(device)
                partial_extra_out = torch.index_select(extra_out, dim=1, index=indx_arr)
                output = self.weight_propagation(x, arr[i], partial_extra_out)
                if out.size(0) == 0: 
                    out = output 
                else:
                    out = torch.concat((out, output), dim=1)
                index += arr[i]
            return out
        else:
            return self.weight_propagation(x, output_size, extra_out)
    
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

model = torch.load("..\\models\\ChessEngine.pth", map_location=device)
board = chess.Board()

while True:

    tensor_board = encode_board(board).to(device)
    possible_move = torch.tensor([]).to(device)

    # Encode possible moves
    legal_moves = board.legal_moves
    for move in legal_moves:
        move_enc = encode_move(move)
        if possible_move.size(0) == 0: possible_move = move_enc
        else: possible_move = torch.concat((possible_move, move_enc), dim=0)

    # Append to variables
    max_move = torch.softmax(model(tensor_board, possible_move.unsqueeze(dim=0)).flatten(), dim=0)
    max_move = torch.argmax(max_move).item()
    computer_move = list(legal_moves)[max_move]
    board.push(computer_move)
    print("Computer Move:", computer_move)
    print("-------------------------------------------")
    print(board)

    a = input("Your Move (beshaan): ")
    board.push_san(a)