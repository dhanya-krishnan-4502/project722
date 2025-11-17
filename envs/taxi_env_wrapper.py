# import gymnasium as gym
# import numpy as np
# from dataclasses import dataclass

# ACTIONS = {
#     "south": 0, "north": 1, "east": 2, "west": 3,
#     "pickup": 4, "dropoff": 5,
# }

# @dataclass
# class StepResult:
#     state: int
#     reward: float
#     terminated: bool
#     truncated: bool
#     info: dict


# class TaxiWrapper:
#     def __init__(self, fail_prob: float = 0.05, seed: int = 0):
#         self.env = gym.make("Taxi-v3")
#         self.fail_prob = fail_prob
#         self.state, _ = self.env.reset(seed=seed)
#         self.decode = lambda i: tuple(np.unravel_index(i, (5, 5, 5, 4)))

#     def reset(self):
#         self.state, _ = self.env.reset(seed=self.seed)
#         taxi_row, taxi_col, pass_loc, dest_idx = self.decode(self.state)

#         # Gym sometimes initializes with passenger already in taxi (index=4)
#         # Force it to a valid location (0–3)
#         if pass_loc == 4:
#             pass_loc = self.rng.choice([0, 1, 2, 3])
#             self.state = self.encode(taxi_row, taxi_col, pass_loc, dest_idx)
#             return self.state

#     def get_state_predicates(self):
#         taxi_row, taxi_col, pass_loc, dest_idx = self.decode(self.state)
#         return {
#             "taxi_at": (int(taxi_row), int(taxi_col)),
#             "passenger_loc": int(pass_loc),
#             "destination": int(dest_idx),
#         }

#     def step(self, action):
#         """Step the Taxi-v3 environment and standardize info flags."""
#         ACTION_MAP = {
#             "south": 0,
#             "north": 1,
#             "east": 2,
#             "west": 3,
#             "pickup": 4,
#             "dropoff": 5,
#         }

#         if isinstance(action, str):
#             action = ACTION_MAP[action]

#         obs, reward, terminated, truncated, info = self.env.step(action)
#         self.state = obs  # keep env state synced

#         # Initialize info flags safely
#         info = dict(info)  # make a copy to modify
#         info.setdefault("success", False)
#         info.setdefault("failed", False)
#         info.setdefault("done", False)

#         # --- Success: passenger dropped off ---
#         if reward == 20:
#             info["success"] = True
#             info["done"] = True
#         # --- Illegal action (pickup/dropoff in wrong place) ---
#         elif reward == -10:
#             info["failed"] = True
#         # --- Episode ended (timeout, wrong drop, etc.) ---
#         elif terminated or truncated:
#             info["done"] = True
#         self.state = obs

#         return StepResult(obs, reward, terminated, truncated, info)
import gymnasium as gym
import numpy as np
import random
from dataclasses import dataclass


ACTIONS = {
    "south": 0, "north": 1, "east": 2, "west": 3,
    "pickup": 4, "dropoff": 5
}


@dataclass
class StepResult:
    state: int
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class TaxiWrapper:
    def __init__(self, fail_prob: float = 0, seed: int = 0):
        self.env = gym.make("Taxi-v3")
        self.seed = seed
        self.rng = random.Random(seed)
        self.fail_prob = float(fail_prob)
        self.state, _ = self.env.reset(seed=seed)

        # ✅ Use Gym's built-in encode/decode (returns a proper tuple)
        try:
            self.decode = self.env.unwrapped.decode
            self.encode = self.env.unwrapped.encode
        except AttributeError:
            # Fallback if using older gym version without unwrapped.decode
            self.decode = lambda i: tuple(np.unravel_index(i, (5, 5, 5, 4)))
            self.encode = lambda r, c, p, d: np.ravel_multi_index((r, c, p, d), (5, 5, 5, 4))

    # -------------------------------------------------------------------------
    def reset(self):
        """Reset environment and ensure passenger starts outside taxi."""
        self.state, _ = self.env.reset(seed=self.seed)
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(self.state)

        # 🚫 Fix: Gym sometimes starts with passenger already in taxi (index 4)
        if pass_loc == 4:
            pass_loc = self.rng.choice([0, 1, 2, 3])
            self.state = self.encode(taxi_row, taxi_col, pass_loc, dest_idx)

        return self.state


    # -------------------------------------------------------------------------
    def get_state_predicates(self):
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(self.state)
        return {
            "taxi_at": (int(taxi_row), int(taxi_col)),
            "passenger_loc": int(pass_loc),
            "destination": int(dest_idx),
        }

    # -------------------------------------------------------------------------
    def _inject_failure(self):
        """Randomly fail movement actions."""
        return self.rng.random() < self.fail_prob

    # -------------------------------------------------------------------------
    def step(self, action_name: str) -> StepResult:
        """Take one step in Taxi-v3, handling random failures and success flags."""
        assert action_name in ACTIONS, f"Unknown action: {action_name}"
        intended = ACTIONS[action_name]
        info = {"intended": action_name, "failed": False, "success": False, "done": False}

        # Optional: Inject stochastic failure for movement actions
        if intended in (0, 1, 2, 3) and self._inject_failure():
            info["failed"] = True
            # Simulate no-op by not calling env.step
            return StepResult(self.state, -1, False, False, info)

        obs, reward, terminated, truncated, env_info = self.env.step(intended)
        self.state = obs  # <-- ✅ Keep wrapper state synced

        info.update(env_info)

        # Success/failure signals
        if reward == 20:
            info["success"] = True
            info["done"] = True
        elif reward == -10:
            info["failed"] = True

        return StepResult(obs, reward, terminated, truncated, info)
