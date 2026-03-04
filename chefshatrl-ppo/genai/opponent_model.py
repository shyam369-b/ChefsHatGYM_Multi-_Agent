import torch
import torch.nn as nn
import torch.nn.functional as F


class OpponentModel(nn.Module):
    """
    Opponent Model for Chef's Hat RL

    Purpose:
    - Predict opponent action given current state
    - Used for opponent-aware policy learning (Variant 1)

    Input:
        state (tensor): [batch_size, state_dim]

    Output:
        logits (tensor): [batch_size, action_dim]
    """

    def __init__(self, state_dim, action_dim):
        super(OpponentModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        # Output layer
        self.policy_head = nn.Linear(128, action_dim)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: tensor of shape [batch_size, state_dim]

        Returns:
            logits: raw action scores
        """
        features = self.feature_net(x)
        logits = self.policy_head(features)
        return logits

    def predict(self, state):
        """
        Predict opponent action (for inference)

        Args:
            state: numpy array or tensor

        Returns:
            action (int)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            logits = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1)

        return action.item()

    def predict_proba(self, state):
        """
        Returns probability distribution over actions
        (useful for visualization / analysis)

        Args:
            state

        Returns:
            numpy array of probabilities
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            logits = self.forward(state)
            probs = F.softmax(logits, dim=-1)

        return probs.cpu().numpy()[0]