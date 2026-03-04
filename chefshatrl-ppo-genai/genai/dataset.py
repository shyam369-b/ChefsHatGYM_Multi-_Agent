import numpy as np
from tqdm import tqdm
from env.make_env import make_env


def preprocess_state(state):
    return np.array(state, dtype=np.float32).flatten()


def collect_dataset(num_episodes=100):
    from env.make_env import make_env

    env = make_env()

    states = []
    actions = []

    for _ in range(num_episodes):
        result = env.reset()

        if isinstance(result, tuple):
            state = result[0]
        else:
            state = result

        done = False

        while not done:
            action = env.action_space.sample()

            result = env.step(action)

            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            states.append(state)
            actions.append(action)

            state = next_state

    return states, actions