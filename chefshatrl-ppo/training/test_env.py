import sys
import os

# Fix paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "ChefsHatGYM", "src"))

from env.make_env import make_env


def main():
    print("🚀 Creating environment...")
    env = make_env()

    print("✅ Environment created:", env)

    obs = env.reset()
    print("🎯 Initial observation:", obs)

    done = False
    step = 0

    while not done and step < 5:
        action = env.action_space.sample()
        print(f"➡️ Step {step} | Action:", action)

        obs, reward, done, info = env.step(action)

        print("   Obs:", obs)
        print("   Reward:", reward)
        print("   Done:", done)

        step += 1

    print("🏁 Test completed successfully!")


if __name__ == "__main__":
    main()