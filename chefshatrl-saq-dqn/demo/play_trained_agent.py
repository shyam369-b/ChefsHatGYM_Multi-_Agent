import numpy as np
import torch
import time

from utils.env_wrapper import make_env
from agents.sac.sac_agent import SACAgent


# =========================
# LOAD TRAINED MODEL
# =========================
env = make_env()

state = np.array(env.reset(), dtype=np.float32)
state_dim = len(state)
action_dim = env.action_space.n

agent = SACAgent(state_dim, action_dim)
agent.actor.load_state_dict(torch.load("models/sac/actor_ep450.pt"))
agent.actor.eval()

print("🎮 DEMO STARTED...\n")


# =========================
# PLAY EPISODES
# =========================
episodes = 5

for ep in range(episodes):
    state = np.array(env.reset(), dtype=np.float32)
    done = False
    total_reward = 0

    print(f"\n===== Episode {ep} =====")

    while not done:
        action = agent.select_action(state, deterministic=True)

        next_state, reward, done, info = env.step(action)

        print(f"Action: {action}, Reward: {reward}")

        state = np.array(next_state, dtype=np.float32)
        total_reward += reward

        time.sleep(0.2)  # slow down for demo

    print(f"Episode Reward: {total_reward}")

print("\n✅ Demo Finished")