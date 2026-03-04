import os
import sys
import numpy as np
import torch

# -------------------------------
# FIX IMPORT PATH (VERY IMPORTANT)
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Add ChefsHatGYM path
chefshat_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ChefsHatGYM", "src")
)
if chefshat_path not in sys.path:
    sys.path.append(chefshat_path)
    print(f"📦 Added to PYTHONPATH: {chefshat_path}")

# -------------------------------
# IMPORTS
# -------------------------------
from env.make_env import make_env
from agents.ppo_agent import PPOAgent

rewards_history =[]
def main():
    print("🚀 Starting PPO Training...")
    env = make_env()

    # Get dimensions
    state = env.reset()

    # Handle tuple (gymnasium compatibility)
    if isinstance(state, tuple):
        state = state[0]

    state_dim = len(state) if hasattr(state, "__len__") else 1
    action_dim = env.action_space.n

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        gae_lambda=0.95,
        device="cpu",
    )

    max_episodes = 200
    max_timesteps = 200

    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_reward(reward, done)

            state = next_state
            total_reward += reward

        agent.update()
        episode_rewards.append(total_reward)

        print(f"Episode {episode+1} | Reward: {total_reward}")

    # Save model
    os.makedirs("models", exist_ok=True)
    agent.save("results/models/ppo_chefshat.pth")
    print("✅ Model saved!")
    
    import matplotlib.pyplot as plt

    plt.plot(rewards_history)
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.savefig("results/plots/training_curve.png")
    import pandas as pd

    pd.DataFrame({"reward": [total_reward]}).to_csv(
        "results/training_rewards.csv", index=False
    )


if __name__ == "__main__":
    main()