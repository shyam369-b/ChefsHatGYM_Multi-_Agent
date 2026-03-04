import numpy as np
import matplotlib.pyplot as plt
import os
import json

def moving_average(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot():
    os.makedirs("results/plots", exist_ok=True)

    # Load data
    ppo = np.load("results/logs/ppo.npy")
    random = np.load("results/logs/random.npy")
    genai = np.load("results/logs/genai.npy")

    # Moving averages
    ppo_ma = moving_average(ppo)
    random_ma = moving_average(random)
    genai_ma = moving_average(genai)

    # =========================
    # Plot 1: Raw Rewards
    # =========================
    plt.figure()
    plt.plot(ppo, label="PPO")
    plt.plot(random, label="Random")
    plt.plot(genai, label="PPO + Opponent Model")

    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()

    plt.savefig("results/plots/rewards_raw.png")
    plt.close()

    # =========================
    # Plot 2: Moving Average
    # =========================
    plt.figure()
    plt.plot(ppo_ma, label="PPO")
    plt.plot(random_ma, label="Random")
    plt.plot(genai_ma, label="PPO + Opponent Model")

    plt.title("Moving Average Reward (Window=10)")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.legend()
    plt.grid()

    plt.savefig("results/plots/rewards_smooth.png")
    plt.close()

    print("📊 Plots saved in results/plots/")

if __name__ == "__main__":
    plot()