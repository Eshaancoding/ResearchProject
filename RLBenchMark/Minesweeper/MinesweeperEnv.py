from minesweeper.msgame import MSGame

class MinesweeperEnv:
    def get_possible_moves (self):
        print(self.game.get_info_map())
        return None

    def reset (self):
        self.game = MSGame(10, 10, 5)
        self.possible_moves = self.get_possible_moves()

    def step (self, index):
        

    def render (self):
        self.game.print_board()

env = MinesweeperEnv()
env.get_possible_moves()