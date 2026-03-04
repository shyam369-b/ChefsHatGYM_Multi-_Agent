import numpy as np


class MetricsTracker:
    def __init__(self):
        self.rewards = []
        self.wins = []

    def update(self, reward, win):
        self.rewards.append(reward)
        self.wins.append(win)

    def get_win_rate(self):
        return np.mean(self.wins)

    def get_avg_reward(self):
        return np.mean(self.rewards)

    def get_stability(self):
        return np.std(self.rewards)