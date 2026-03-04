import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from agents.network import ActorCritic


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.state_values = []

    def clear(self):
        self.__init__()


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        gae_lambda=0.95,
        use_genai=False,
        opponent_model=None,
        device="cpu"
    ):
        self.device = torch.device(device)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda

        self.use_genai = use_genai
        self.opponent_model = opponent_model

        # Policy networks
        self.original_state_dim = state_dim

        if use_genai:
            state_dim = state_dim + action_dim  # augmented state
            
        

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.buffer = RolloutBuffer()

    def augment_state(self, state):
        if not self.use_genai or self.opponent_model is None:
            return state

        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            opp_pred = self.opponent_model(state_tensor)

        # Flatten if needed
        opp_pred = opp_pred.squeeze()

        augmented = torch.cat([state_tensor, opp_pred], dim=-1)

        return augmented.cpu().numpy()

    def select_action(self, state):
        # Apply GenAI augmentation if enabled
        state = self.augment_state(state)

        state = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            action_probs, state_value = self.policy_old(state)

        dist = Categorical(action_probs)
        action = dist.sample()

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(dist.log_prob(action))
        self.buffer.state_values.append(state_value)

        return action.item()

    def store_reward(self, reward, done):
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)

    def compute_gae(self):
        rewards = []
        discounted_reward = 0
        advantages = []
        gae = 0

        values = [v.item() for v in self.buffer.state_values]
        values.append(0)  # for terminal

        for t in reversed(range(len(self.buffer.rewards))):
            delta = (
                self.buffer.rewards[t]
                + self.gamma * values[t + 1] * (1 - self.buffer.dones[t])
                - values[t]
            )

            gae = delta + self.gamma * self.gae_lambda * (1 - self.buffer.dones[t]) * gae
            advantages.insert(0, gae)

            discounted_reward = (
                self.buffer.rewards[t]
                + self.gamma * discounted_reward * (1 - self.buffer.dones[t])
            )
            rewards.insert(0, discounted_reward)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        return rewards, advantages

    def update(self):
        # Convert buffer to tensors
        old_states = torch.stack(self.buffer.states).detach().to(self.device)
        old_actions = torch.stack(self.buffer.actions).detach().to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs).detach().to(self.device)

        rewards, advantages = self.compute_gae()

        for _ in range(self.k_epochs):
            action_probs, state_values = self.policy(old_states)

            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()

            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # PPO loss
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values.squeeze(), rewards)
                - 0.01 * dist_entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy_old.load_state_dict(self.policy.state_dict())