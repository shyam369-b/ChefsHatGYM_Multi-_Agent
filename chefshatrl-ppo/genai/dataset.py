import numpy as np
from env.make_env import make_env
from tqdm import tqdm


def collect_dataset(num_episodes=200):
    """
    Collect dataset of (state, opponent_action)

    Returns:
        states: np.array [N, state_dim]
        actions: np.array [N]
    """

    print("🚀 Collecting dataset...")

    env = make_env()

    states = []
    actions = []

    for _ in tqdm(range(num_episodes)):
        state = env.reset()

        done = False
        while not done:
            # Random policy (baseline opponent behaviour)
            action = np.random.randint(0, 5)

            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)

            state = next_state

    return np.array(states), np.array(actions)