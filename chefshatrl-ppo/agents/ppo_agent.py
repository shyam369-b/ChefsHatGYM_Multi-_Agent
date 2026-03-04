import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# =========================
# POLICY NETWORK
# =========================
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# PPO AGENT
# =========================
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma

        # ✅ MEMORY (FIX)
        self.log_probs = []
        self.rewards = []
        self.dones = []

    # =========================
    # ACTION SELECTION
    # =========================
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        probs = self.policy(state)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        # ✅ STORE LOG PROB
        self.log_probs.append(log_prob)

        return action.item()   # IMPORTANT: return only action

    # =========================
    # STORE REWARD
    # =========================
    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    # =========================
    # COMPUTE RETURNS
    # =========================
    def compute_returns(self):
        returns = []
        G = 0

        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # normalize (stability improvement)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    # =========================
    # UPDATE POLICY
    # =========================
    def update(self):
        if len(self.rewards) == 0:
            return

        returns = self.compute_returns()

        loss = []
        for log_prob, G in zip(self.log_probs, returns):
            loss.append(-log_prob * G)

        loss = torch.stack(loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ✅ CLEAR MEMORY AFTER UPDATE
        self.log_probs = []
        self.rewards = []
        self.dones = []

    # =========================
    # SAVE / LOAD
    # =========================
    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.policy.to(self.device)