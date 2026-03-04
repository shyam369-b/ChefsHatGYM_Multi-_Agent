import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from env.make_env import make_env
from agents.ppo_agent import PPOAgent
from genai.opponent_model import OpponentModel

EPISODES = 200


# =========================
# RANDOM AGENT
# =========================
class RandomAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def select_action(self, state):
        return np.random.randint(self.action_dim)


# =========================
# LOAD AGENTS
# =========================
def load_ppo():
    agent = PPOAgent(1, 5)
    agent.load("results/models/ppo_chefshat.pth")
    return agent


def load_genai():
    model = OpponentModel(1, 5)
    model.load_state_dict(torch.load("results/models/opponent_model.pth"))
    model.eval()
    return model


# =========================
# MATCH FUNCTION
# =========================
def run_episode(agent, env):
    state = env.reset()
    done = False

    total_reward = 0
    steps = 0

    while not done:
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)

        total_reward += reward
        steps += 1

    return total_reward, steps


# =========================
# EXPERIMENT FUNCTION
# =========================
def run_experiment(mode):
    env = make_env()
    agent = load_ppo()

    rewards = []
    lengths = []
    wins = 0

    for _ in tqdm(range(EPISODES)):
        reward, steps = run_episode(agent, env)

        rewards.append(reward)
        lengths.append(steps)

        if reward > 0:
            wins += 1

    return {
        "mode": mode,
        "win_rate": wins / EPISODES,
        "avg_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "avg_length": np.mean(lengths),
    }


# =========================
# MAIN
# =========================
def main():
    print("🚀 Running ALL experiments...")

    results = []

    for mode in ["random", "genai", "selfplay"]:
        res = run_experiment(mode)
        results.append(res)

    df = pd.DataFrame(results)

    print("\n===== FINAL RESULTS =====")
    print(df)

    df.to_csv("results/experiment_results.csv", index=False)


if __name__ == "__main__":
    main()