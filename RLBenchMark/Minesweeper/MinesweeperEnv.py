from minesweeper.msgame import MSGame
from random import randint

class MinesweeperEnv:
    def get_possible_moves (self):
        info_map = self.game.get_info_map()
        possible_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if info_map[x][y] == 11:
                    possible_moves.append((x, y))
        
        return possible_moves

    def random_move (self):
        index = randint(0, len(self.possible_moves)-1)
        return self.possible_moves[index]

    def reset (self, board_size=10, num_mines=20):
        self.board_size = board_size
        while True:
            self.game = MSGame(board_size, board_size, num_mines)
            self.possible_moves = self.get_possible_moves()

            x, y = self.random_move()
            self.game.play_move("click", x, y)
            self.possible_moves = self.get_possible_moves()
            
            if self.game.game_status == 2: break

        return self.game.get_info_map(), self.possible_moves

    def step (self, index):
        reward = 1
        x, y = self.possible_moves[index]
        self.game.play_move("click", x, y)
        
        is_done = False
        if self.game.game_status == 1: 
            # The AI WON!!!
            is_done = True
            reward = 10
        elif self.game.game_status == 0:
            # The AI LOST!!
            is_done = False
            reward = -10
        
        self.possible_moves = self.get_possible_moves()
        return self.game.get_info_map(), self.possible_moves, reward, is_done, None, None

    def manual_step (self, x, y):
        self.game.play_move("click", x, y)

    def render (self):
        self.game.print_board()

env = MinesweeperEnv()
state, extra_state = env.reset()
index = randint(0, len(extra_state)-1)
next_state, next_extra_state, reward, is_done, _, _ = env.step(index)

print(len(next_extra_state), next_state)