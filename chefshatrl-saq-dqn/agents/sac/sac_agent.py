import torch
import torch.optim as optim
import torch.nn.functional as F


class SACAgent:
    def __init__(self, state_dim, action_dim):
        from .networks import Actor, Critic

        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)

        self.target_critic1 = Critic(state_dim, action_dim)
        self.target_critic2 = Critic(state_dim, action_dim)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=3e-4
        )

        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            probs = self.actor(state)

        if deterministic:
            return torch.argmax(probs, dim=1).item()
        else:
            dist = torch.distributions.Categorical(probs)
            return dist.sample().item()

    def update(self, buffer, batch_size=64):
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # ===== TARGET =====
        with torch.no_grad():
            next_probs = self.actor(next_states)
            next_log_probs = torch.log(next_probs + 1e-8)

            q1_next = self.target_critic1(next_states)
            q2_next = self.target_critic2(next_states)
            q_next = torch.min(q1_next, q2_next)

            next_value = (next_probs * (q_next - self.alpha * next_log_probs)).sum(dim=1)
            target_q = rewards + self.gamma * (1 - dones) * next_value

        # ===== CRITIC =====
        q1 = self.critic1(states).gather(1, actions.unsqueeze(1)).squeeze()
        q2 = self.critic2(states).gather(1, actions.unsqueeze(1)).squeeze()

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ===== ACTOR =====
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)

        q1 = self.critic1(states)
        q2 = self.critic2(states)
        q = torch.min(q1, q2)

        actor_loss = (probs * (self.alpha * log_probs - q)).sum(dim=1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update()

    def soft_update(self):
        for t, s in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            t.data.copy_(t.data * (1 - self.tau) + s.data * self.tau)

        for t, s in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            t.data.copy_(t.data * (1 - self.tau) + s.data * self.tau)