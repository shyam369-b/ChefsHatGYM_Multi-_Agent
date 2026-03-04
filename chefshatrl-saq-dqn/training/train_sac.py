from utils.env_wrapper import make_env
from agents.sac.sac_agent import SACAgent
from agents.common.replay_buffer import ReplayBuffer

import numpy as np
import torch
import os

env = make_env()

state = env.reset()
state = np.array(state, dtype=np.float32)

state_dim = len(state)
action_dim = env.action_space.n

print(f"State Dim: {state_dim}, Action Dim: {action_dim}")

agent = SACAgent(state_dim, action_dim)

buffer = ReplayBuffer(size=100000, state_dim=state_dim)

os.makedirs("models/sac", exist_ok=True)
os.makedirs("results", exist_ok=True)

rewards_log = []
wins_log = []

episodes = 500
batch_size = 64
update_after = 1000
total_steps = 0

for ep in range(episodes):
    state = np.array(env.reset(), dtype=np.float32)

    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)

        next_state, reward, done, info = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)

        buffer.add(state, action, reward, next_state, done)

        if buffer.size > batch_size and total_steps > update_after:
            agent.update(buffer, batch_size)

        state = next_state
        total_reward += reward
        total_steps += 1

    winner = info.get("winner", -1)
    win = 1 if winner == 0 else 0

    rewards_log.append(total_reward)
    wins_log.append(win)

    # ✅ FIXED SAVE (correct names)
    if ep % 50 == 0:
        torch.save(agent.actor.state_dict(), f"models/sac/actor_ep{ep}.pt")
        torch.save(agent.critic1.state_dict(), f"models/sac/critic1_ep{ep}.pt")
        torch.save(agent.critic2.state_dict(), f"models/sac/critic2_ep{ep}.pt")

    print(f"SAC Episode {ep}, Reward: {total_reward:.2f}, Win: {win}")

np.save("results/sac_rewards.npy", rewards_log)
np.save("results/sac_wins.npy", wins_log)

print("✅ SAC Training Complete")