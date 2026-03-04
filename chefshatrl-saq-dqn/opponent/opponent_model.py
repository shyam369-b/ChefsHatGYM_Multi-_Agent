import numpy as np


class OpponentModel:
    def __init__(self, history_len=5):
        self.history_len = history_len
        self.history = []

    def update(self, action):
        self.history.append(action)
        if len(self.history) > self.history_len:
            self.history.pop(0)

    def get_features(self):
        if len(self.history) == 0:
            return np.zeros(self.history_len)
        padded = self.history + [0] * (self.history_len - len(self.history))
        return np.array(padded, dtype=np.float32)