import torch
import torch.nn as nn
import torch.nn.functional as F


class OpponentModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(OpponentModel, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Predict opponent action distribution
        return F.softmax(self.out(x), dim=-1)