import numpy as np


def moving_average(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')


def compute_metrics(rewards):
    rewards = np.array(rewards)

    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "max_reward": float(np.max(rewards)),
        "min_reward": float(np.min(rewards)),
        "final_10_avg": float(np.mean(rewards[-10:])),
        "moving_avg": moving_average(rewards).tolist(),
        "variance": float(np.var(rewards)),
    }

    return metrics