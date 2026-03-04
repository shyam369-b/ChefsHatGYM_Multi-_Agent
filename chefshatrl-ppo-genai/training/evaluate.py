import os
import sys
import numpy as np
import torch

# Fix paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "ChefsHatGYM", "src"))

import numpy as np
from env.make_env import make_env
from agents.ppo_agent import PPOAgent

EPISODES = 50

def evaluate():
    env = make_env()

    state_dim = 4
    action_dim = 5

    agent = PPOAgent(state_dim, action_dim)
    agent.load("results/models/ppo_chefshat.pth")

    wins = 0
    total_reward = 0

    for ep in range(EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)

            episode_reward += reward

        total_reward += episode_reward

        if episode_reward > 0:
            wins += 1

    print("===== EVALUATION =====")
    print(f"Win Rate: {wins / EPISODES:.2f}")
    print(f"Average Reward: {total_reward / EPISODES:.2f}")

if __name__ == "__main__":
    evaluate()