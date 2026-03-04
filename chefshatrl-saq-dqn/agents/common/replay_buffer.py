import numpy as np

class ReplayBuffer:
    def __init__(self, size=100000, state_dim=10):
        self.size = size
        self.ptr = 0
        self.len = 0

        self.state_mem = np.zeros((size, state_dim), dtype=np.float32)
        self.next_state_mem = np.zeros((size, state_dim), dtype=np.float32)
        self.action_mem = np.zeros(size, dtype=np.int64)
        self.reward_mem = np.zeros(size, dtype=np.float32)
        self.done_mem = np.zeros(size, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.state_mem[self.ptr] = state
        self.next_state_mem[self.ptr] = next_state
        self.action_mem[self.ptr] = action
        self.reward_mem[self.ptr] = reward
        self.done_mem[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.size
        self.len = min(self.len + 1, self.size)

    def sample(self, batch_size=64):
        # ⚠️ Prevent "sample larger than population" error
        actual_batch = min(batch_size, self.len)
        idx = np.random.choice(self.len, actual_batch, replace=False)

        states = self.state_mem[idx]
        next_states = self.next_state_mem[idx]
        actions = self.action_mem[idx]
        rewards = self.reward_mem[idx]
        dones = self.done_mem[idx]

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.len