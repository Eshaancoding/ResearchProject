import sys; sys.path.append("..\\")
from torch import nn
from RL.DQN import *
from VNN import *
from Chess.ChessEnv import *

class ChessEngineNN (nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Model
        self.model = nn.Sequential(
            nn.Linear(832, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
        )        
        
        d_model = 16
        weight_model = nn.Sequential(
            nn.Linear(d_model*2+33, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        ) 

        bias_model = nn.Sequential(
            nn.Linear(d_model+33, 12),
            nn.Tanh(),
            nn.Linear(12, 1),
        )

        self.vnn_model = VNNBlock(
            d_model=d_model,
            weight_nn=weight_model,
            bias_nn=bias_model
        ) 

    def forward (self, x, possible_moves):
        possible_moves = possible_moves.unsqueeze(0)
        x = self.model(x)

        return self.vnn_model(x, possible_moves.size(1), possible_moves)

model = ChessEngineNN()

# Declare env
env = ChessEnv()

# Declare trainer
trainer = DQN(
    model=model,

    replay_mem_max_len=100_000,
    batch_size=32,
    gamma=0.95,
    lr=0.001,

    update_target_model_per_epi=10,
    epsilon=1,
    epsilon_decay=0.995,
    epsilon_min=0.1,
    test_per_epi=10_000,

    model_path=None,
    should_load_from_path=False, 
    save_per_epi=100,
)

# Train
trainer.train(
    env=env,
    num_episodes=1_000_000,
    use_tqdm=True,
    should_test=False
)