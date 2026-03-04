import numpy as np

class EnvWrapper:
    def __init__(self):
        from ChefsHatGym.env.ChefsHatEnv import ChefsHatEnv

        # ✅ DO NOT pass unsupported args
        self.env = ChefsHatEnv()

        self.action_space = self.env.action_space
        self._printed_debug = False

    def _extract_state(self, state):
        """
        Convert raw env state into meaningful vector
        """

        # 🔥 HANDLE tuple (new gym API)
        if isinstance(state, tuple):
            state = state[0]

        # CASE 1: ndarray
        if isinstance(state, np.ndarray):
            state = state.flatten().astype(np.float32)

        # CASE 2: dict (IMPORTANT)
        elif isinstance(state, dict):
            features = []

            for key, value in state.items():
                if isinstance(value, list):
                    features.extend(value)
                elif isinstance(value, np.ndarray):
                    features.extend(value.flatten().tolist())
                elif isinstance(value, (int, float)):
                    features.append(value)

            state = np.array(features, dtype=np.float32)

        # CASE 3: list
        elif isinstance(state, list):
            state = np.array(state, dtype=np.float32)

        # FALLBACK
        else:
            state = np.array([state], dtype=np.float32)

        # 🔥 DEBUG (PRINT ONCE)
        if not self._printed_debug:
            print("\n===== STATE DEBUG =====")
            print(state)
            print("Shape:", state.shape)
            print("=======================\n")
            self._printed_debug = True

        return state

    def reset(self):
        state = self.env.reset()

        # gymnasium compatibility
        if isinstance(state, tuple):
            state = state[0]

        return self._extract_state(state)

    def step(self, action):
        result = self.env.step(action)

        # 🔥 handle both APIs
        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_state, reward, done, info = result

        return self._extract_state(next_state), reward, done, info


def make_env():
    return EnvWrapper()