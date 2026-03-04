import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/experiment_results.csv")
df = pd.read_csv("results/training_rewards.csv")

plt.plot(df["reward"].rolling(10).mean())
plt.title("Learning Curve (Smoothed)")
plt.savefig("results/plots/learning_curve.png")
# Win Rate
plt.figure()
plt.bar(df["mode"], df["win_rate"])
plt.title("Win Rate Comparison")
plt.savefig("results/plots/win_rate.png")

# Reward
plt.figure()
plt.bar(df["mode"], df["avg_reward"])
plt.title("Average Reward")
plt.savefig("results/plots/reward.png")

# Episode Length
plt.figure()
plt.bar(df["mode"], df["avg_length"])
plt.title("Episode Length")
plt.savefig("results/plots/length.png")

print("✅ Plots saved!")