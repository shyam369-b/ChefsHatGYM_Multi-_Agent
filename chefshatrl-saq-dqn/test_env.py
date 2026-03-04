import sys
sys.path.append("ChefsHatGYM")

from ChefsHatGym.env.ChefsHatEnv import ChefsHatEnv

env = ChefsHatEnv()
obs = env.reset()

print("Observation:", obs)

action = 0
obs, reward, done, info = env.step(action)

print("Step works:", obs, reward, done)