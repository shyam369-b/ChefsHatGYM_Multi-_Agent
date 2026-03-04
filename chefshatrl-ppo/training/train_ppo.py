import os
import torch
from env.make_env import make_env
from agents.ppo_agent import PPOAgent


def main():
    os.makedirs("results/models", exist_ok=True)

    print("🚀 Creating Chef’s Hat environment...")
    env = make_env()

    state = env.reset()
    state_dim = len(state)
    action_dim = 5

    agent = PPOAgent(state_dim, action_dim)

    episodes = 200

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # ✅ Only action (NO unpacking)
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action)

            # ✅ Store reward internally
            agent.store_reward(reward, done)

            state = next_state
            total_reward += reward

        # ✅ Update after episode (NO arguments)
        agent.update()

        print(f"Episode {ep} | Reward: {total_reward}")

    # ✅ Save model
    agent.save("results/models/ppo.pth")

    print("✅ PPO training completed & model saved!")


if __name__ == "__main__":
    main()