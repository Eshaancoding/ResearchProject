from random import randint

class TestActionSpace ():
    def sample (self):
        return randint(0, 4)

class CustomEnv ():
    def __init__(self) -> None:
        self.itr = 0
        self.action_space = TestActionSpace()

    def reset (self):
        self.itr = 0
        self.random_integer = randint(0, 4)
        self.x = [0,0,0,0,0]
        self.x[self.random_integer] = 1

        return self.x
        
    def step (self, action):
        self.itr += 1
        reward = 4 - abs(self.random_integer - action)

        self.random_integer = randint(0, 4)
        self.x = [0,0,0,0,0]
        self.x[self.random_integer] = 1

        is_done = False
        if self.itr == 100:
            is_done = True
            self.itr = 0
        
        return self.x, reward, is_done, ""