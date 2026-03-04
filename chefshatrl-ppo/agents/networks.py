import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        x = self.shared(state)
        return self.actor(x), self.critic(x)