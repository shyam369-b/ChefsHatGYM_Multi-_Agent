import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.env_wrapper import make_env
from agents.dqn.dqn_agent import DQNAgent
from agents.sac.sac_agent import SACAgent


# =========================
# LOAD MODELS
# =========================
def load_dqn(model_path, state_dim, action_dim):
    agent = DQNAgent(state_dim, action_dim)
    agent.q_net.load_state_dict(torch.load(model_path))
    agent.q_net.eval()
    return agent


def load_sac(actor_path, state_dim, action_dim):
    agent = SACAgent(state_dim, action_dim)
    agent.actor.load_state_dict(torch.load(actor_path))
    agent.actor.eval()
    return agent


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(agent, env, episodes=50, is_sac=False):
    rewards = []
    wins = []

    for ep in range(episodes):
        state = np.array(env.reset(), dtype=np.float32)

        done = False
        total_reward = 0

        while not done:
            if is_sac:
                action = agent.select_action(state, deterministic=True)
            else:
                action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)
            state = np.array(next_state, dtype=np.float32)

            total_reward += reward

        # Win logic
        winner = info.get("winner", -1)
        win = 1 if winner == 0 else 0

        rewards.append(total_reward)
        wins.append(win)

    avg_reward = np.mean(rewards)
    win_rate = np.mean(wins) * 100

    return rewards, avg_reward, win_rate


# =========================
# PLOTTING FUNCTION
# =========================
def plot_results(dqn_rewards, sac_rewards, dqn_wins, sac_wins):
    plt.figure()
    plt.plot(dqn_rewards, label="DQN Rewards")
    plt.plot(sac_rewards, label="SAC Rewards")
    plt.legend()
    plt.title("Reward Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("results/reward_comparison.png")
    plt.show()

    plt.figure()
    plt.plot(dqn_wins, label="DQN Wins")
    plt.plot(sac_wins, label="SAC Wins")
    plt.legend()
    plt.title("Win Rate Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Win")
    plt.savefig("results/winrate_comparison.png")
    plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    env = make_env()

    state = np.array(env.reset(), dtype=np.float32)
    state_dim = len(state)
    action_dim = env.action_space.n

    print(f"Eval State Dim: {state_dim}, Action Dim: {action_dim}")

    # ===== LOAD MODELS =====
    dqn_agent = load_dqn("models/dqn/dqn_ep450.pt", state_dim, action_dim)
    sac_agent = load_sac("models/sac/actor_ep450.pt", state_dim, action_dim)

    # ===== EVALUATE =====
    dqn_rewards, dqn_avg, dqn_win = evaluate(dqn_agent, env, is_sac=False)
    sac_rewards, sac_avg, sac_win = evaluate(sac_agent, env, is_sac=True)

    print("\n===== FINAL RESULTS =====")
    print(f"DQN → Avg Reward: {dqn_avg:.2f}, Win Rate: {dqn_win:.2f}%")
    print(f"SAC → Avg Reward: {sac_avg:.2f}, Win Rate: {sac_win:.2f}%")

    # ===== SAVE ARRAYS =====
    np.save("results/dqn_eval_rewards.npy", dqn_rewards)
    np.save("results/sac_eval_rewards.npy", sac_rewards)
    np.save("results/dqn_eval_wins.npy", dqn_wins := [1 if r > 0 else 0 for r in dqn_rewards])
    np.save("results/sac_eval_wins.npy", sac_wins := [1 if r > 0 else 0 for r in sac_rewards])

    # ===== PLOT =====
    plot_results(dqn_rewards, sac_rewards, dqn_wins, sac_wins)

    print("✅ Evaluation + Graphs Complete")