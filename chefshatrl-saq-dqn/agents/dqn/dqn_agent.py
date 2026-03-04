import torch
import torch.optim as optim
import numpy as np
from .q_network import QNetwork

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.action_dim = action_dim

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.q_net(state).argmax().item()

    def update(self, replay_buffer, batch_size=64):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        import torch

        states = torch.FloatTensor(states)  # shape: (batch, state_dim)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)  # shape: (batch,)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.target_net(next_states).max(1)[0]

        target = rewards + self.gamma * next_q * (1 - dones)

        loss = ((q_values - target.detach()) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())