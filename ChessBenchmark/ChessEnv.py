import torch.nn.functional as F
import torch
import chess
import linecache
from random import randint

class ChessActionSpace (): 
    def __init__ (self, board):
        self.board = board

    def updateBoard (self, board):
        self.board = board

    def sample (self):
        return randint(0, len(self.board.legal_moves)-1)

class ChessEnv ():
    def __init__(self, chess_db_path=None) -> None:
        if chess_db_path != None: 
            self.chess_db_path = chess_db_path
        self.board = chess.Board()
        linecache.getline(chess_db_path, 0) 
        self.action_space = ChessActionSpace(self.board)

    # Encoding move
    def encode_move (self, move:chess.Move):
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

    # Encode board
    def encode_board (self, board):
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

    def reward_function (self, move, board):
        captured_piece = board.piece_at(move.to_square)
        board.push(move)
        self.action_space.updateBoard(board)
        if captured_piece == None: return 0

        # Player captured adversary
        if captured_piece == chess.Piece.from_symbol('p'): return 1 
        if captured_piece == chess.Piece.from_symbol('b'): return 3
        if captured_piece == chess.Piece.from_symbol('n'): return 3
        if captured_piece == chess.Piece.from_symbol('r'): return 5
        if captured_piece == chess.Piece.from_symbol('q'): return 10
         
        # Adversary captured player
        if captured_piece == chess.Piece.from_symbol('P'): return -1 
        if captured_piece == chess.Piece.from_symbol('B'): return -3
        if captured_piece == chess.Piece.from_symbol('N'): return -3
        if captured_piece == chess.Piece.from_symbol('R'): return -5
        if captured_piece == chess.Piece.from_symbol('Q'): return -10

    def reset (self):
        self.board = chess.Board() 
        
        # Encode possible moves (extra state) and get move index (action)
        possible_move = torch.tensor([])
        for leg_move in list(self.board.legal_moves):
            move_enc = self.encode_move(leg_move)
            if possible_move.size(0) == 0: 
                possible_move = move_enc
            else: 
                possible_move = torch.concat((possible_move, move_enc), dim=0)

        return self.encode_board(self.board), possible_move, None

    def step (self, move):
        legal_moves = list(self.board.legal_moves)
        move = legal_moves[move]
        reward = self.reward_function(move, self.board)
        
        outcome = self.board.outcome()
        is_done = False
        if outcome != None:
            result = outcome.result()
            if result == "1-0":   reward = 20
            elif result == "0-1": reward = -20
            is_done = True

        # Encode possible moves (extra state) and get move index (action)
        possible_move = torch.tensor([])
        for leg_move in list(self.board.legal_moves):
            move_enc = self.encode_move(leg_move)
            if possible_move.size(0) == 0: 
                possible_move = move_enc
            else: 
                possible_move = torch.concat((possible_move, move_enc), dim=0)

        return self.encode_board(self.board), possible_move, reward, is_done, None, None

    def get_database (self):
        len_lines = 3561469
        line = linecache.getline(self.chess_db_path, randint(0, len_lines)+6) 
        line = line.split("###")[1].strip()
        moves = line.split(" ")

        actions = []
        states = []
        next_states = []
        rewards = []
        is_dones = []
        possible_moves = []

        board = chess.Board()
        for index, move in enumerate(moves):
            # parse move and get initial state
            move = board.parse_san(move.split(".")[1])
            state = self.encode_board(board)

            # Encode possible moves (extra state) and get move index (action)
            possible_move = torch.tensor([])
            move_index = -1
            for index, leg_move in enumerate(list(board.legal_moves)):
                move_enc = self.encode_move(leg_move)
                if possible_move.size(0) == 0: 
                    possible_move = move_enc
                else: 
                    possible_move = torch.concat((possible_move, move_enc), dim=0)
                if leg_move == move: move_index = index

            # Push move in board to get reward, get next state
            reward = self.reward_function(move, board)
            next_state = self.encode_board(board)

            # Check if it is done
            outcome = board.outcome()
            is_done = False
            if outcome != None and index == len(moves) - 1:
                result = outcome.result()
                if result == "1-0":   reward = 20
                elif result == "0-1": reward = -20
                is_done = True

            # Append 
            is_dones.append(is_done)
            actions.append(move_index)
            states.append(state)
            next_states.append(next_state)
            possible_moves.append(possible_move)
            rewards.append(reward)

        return actions, states, next_states, rewards, is_dones, possible_moves, True

    def render (self):
        pass

# TRAVEL THE WORLD MAKE SURE YOUR KNOWN IN THIS WORLD THAT WE WILL EXPLOREEEEEEE. - Beshaan, Sophomore, going to be 2025 freshman at Stanford University, Accepted into MIT, CalTech, Carnegie Mellon University, Rutgers University, Brookdale Community College, and the Geshanmobile :thumbsup: