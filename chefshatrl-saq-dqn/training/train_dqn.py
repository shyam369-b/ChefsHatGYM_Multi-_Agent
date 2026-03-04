from utils.env_wrapper import make_env
from agents.dqn.dqn_agent import DQNAgent
from agents.common.replay_buffer import ReplayBuffer
import numpy as np
import os

env = make_env()

state = env.reset()
state_dim = len(state)
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)
buffer = ReplayBuffer(size=10000, state_dim=state_dim)

os.makedirs("models/dqn", exist_ok=True)
os.makedirs("results", exist_ok=True)

episodes = 500

rewards_log = []
wins_log = []

for ep in range(episodes):
    state = env.reset()
    state = np.array(state, dtype=np.float32)

    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)

        next_state, reward, done, info = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)

        # ✅ ONLY learn from meaningful rewards
        buffer.add(state, action, reward, next_state, done)

        if buffer.size > 64:
            agent.update(buffer)

        state = next_state
        total_reward += reward

    # ✅ Win tracking
    winner = info.get("winner", -1)
    win = 1 if winner == 0 else 0

    rewards_log.append(total_reward)
    wins_log.append(win)

    agent.update_target()

    # ✅ Save model
    if ep % 50 == 0:
        import torch
        torch.save(agent.q_net.state_dict(), f"models/dqn/dqn_ep{ep}.pt")

    print(f"DQN Episode {ep}, Reward: {total_reward}, Win: {win}")

# ✅ Save logs
np.save("results/dqn_rewards.npy", rewards_log)
np.save("results/dqn_wins.npy", wins_log)
print(f"✅ Saved DQN model at episode {ep}")