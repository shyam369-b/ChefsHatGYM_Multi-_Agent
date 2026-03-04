import numpy as np
import os
import json
import torch

from env.make_env import make_env
from agents.ppo_agent import PPOAgent
from genai.opponent_model import OpponentModel
from experiments.metrics import compute_metrics


# =========================
# CONFIG
# =========================
EPISODES = 100
ACTION_DIM = 5


# =========================
# RUN SINGLE AGENT
# =========================
def run(agent_type="ppo", use_opponent_model=False):
    env = make_env()

    state = env.reset()
    state_dim = len(state)

    rewards = []

    device = torch.device("cpu")

    # =========================
    # AGENT SETUP
    # =========================
    if agent_type == "ppo":
        agent = PPOAgent(state_dim, ACTION_DIM)

        # Load trained model if exists
        if os.path.exists("results/models/ppo.pth"):
            agent.load("results/models/ppo.pth")
            print("✅ PPO model loaded!")

    else:
        agent = None

    # =========================
    # OPPONENT MODEL SETUP
    # =========================
    opponent_model = None
    if use_opponent_model:
        opponent_model = OpponentModel(state_dim, ACTION_DIM).to(device)

        if os.path.exists("results/models/opponent_model.pth"):
            opponent_model.load_state_dict(
                torch.load("results/models/opponent_model.pth", map_location=device)
            )
            opponent_model.eval()
            print("🧠 Opponent model loaded!")

    # =========================
    # EPISODE LOOP
    # =========================
    for ep in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:

            # -------- RANDOM --------
            if agent_type == "random":
                action = np.random.randint(0, ACTION_DIM)

            # -------- PPO --------
            elif agent_type == "ppo" and not use_opponent_model:
                action = agent.select_action(state)

            # -------- PPO + OPPONENT MODEL --------
            elif agent_type == "ppo" and use_opponent_model:
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

                with torch.no_grad():
                    probs = torch.softmax(opponent_model(state_tensor), dim=-1).cpu().numpy()

                action= agent.select_action(state)

                # Opponent-aware adjustment (SAFE – no state change)
                action = int((action + np.argmax(probs)) % ACTION_DIM)

            else:
                raise ValueError("Invalid agent type")

            # Step environment
            next_state, reward, done, _ = env.step(action)

            if agent_type == "ppo":
                agent.store_reward(reward, done)

            state = next_state
            total_reward += reward

        # Update PPO after each episode
        if agent_type == "ppo":
            agent.update()

        rewards.append(total_reward)

        # Moving average (IMPORTANT for evaluation)
        if len(rewards) >= 10:
            moving_avg = np.mean(rewards[-10:])
        else:
            moving_avg = np.mean(rewards)

        print(
            f"[{agent_type.upper()}] Episode {ep} | Reward: {total_reward:.2f} | MA(10): {moving_avg:.2f}"
        )

    return rewards


# =========================
# MAIN EXPERIMENT RUNNER
# =========================
def main():
    os.makedirs("results/logs", exist_ok=True)

    print("\n🚀 Running PPO baseline...")
    ppo_rewards = run("ppo")

    print("\n🎲 Running Random baseline...")
    random_rewards = run("random")

    print("\n🧠 Running PPO + Opponent Model...")
    genai_rewards = run("ppo", use_opponent_model=True)

    # =========================
    # SAVE RAW LOGS
    # =========================
    np.save("results/logs/ppo.npy", ppo_rewards)
    np.save("results/logs/random.npy", random_rewards)
    np.save("results/logs/genai.npy", genai_rewards)

    # =========================
    # METRICS
    # =========================
    metrics = {
        "ppo": compute_metrics(ppo_rewards),
        "random": compute_metrics(random_rewards),
        "genai": compute_metrics(genai_rewards),
    }

    # Save metrics.json (VERY IMPORTANT)
    with open("results/logs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n📊 FINAL METRICS:")
    print(json.dumps(metrics, indent=4))

    print("\n✅ Experiments completed successfully!")


if __name__ == "__main__":
    main()