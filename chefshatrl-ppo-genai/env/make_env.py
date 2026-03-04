import sys
import os

# ✅ Get project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ✅ Add Chef's Hat src path
CHEF_PATH = os.path.join(BASE_DIR, "ChefsHatGYM", "src")
sys.path.insert(0, CHEF_PATH)

print("📦 Added to PYTHONPATH:", CHEF_PATH)

# ✅ Now import AFTER path fix
import gym
from ChefsHatGym.env.ChefsHatEnv import ChefsHatEnv

def make_env():
    print("🚀 Creating Chef’s Hat environment...")
    env = ChefsHatEnv()
    return env