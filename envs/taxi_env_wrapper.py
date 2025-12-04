import gymnasium as gym
import numpy as np
import random
from dataclasses import dataclass

ACTIONS = {
    "south": 0, "north": 1, "east": 2, "west": 3,
    "pickup": 4, "dropoff": 5
}

GLOBAL_WALLS = {
    (0,0): {'north', 'west'},
    (0,1): {'east', 'north'},
    (0,2): {'north', 'west'},
    (0,3): {'north'},
    (0,4): {'east', 'north'},
    (1,0): {'west'},
    (1,1): {'east'},
    (1,2): {'west'},
    (1,3): set(),
    (1,4): {'east'},
    (2,0): {'west'},
    (2,1): set(),
    (2,2): set(),
    (2,3): set(),
    (2,4): {'east'},
    (3,0): {'east', 'west'},
    (3,1): {'west'},
    (3,2): {'east'},
    (3,3): {'west'},
    (3,4): {'east'},
    (4,0): {'east', 'south', 'west'},
    (4,1): {'south', 'west'},
    (4,2): {'east', 'south'},
    (4,3): {'south', 'west'},
    (4,4): {'east', 'south'},
}

DELTAS = {
    "north": (-1, 0),
    "south": (1, 0),
    "east": (0, 1),
    "west": (0, -1),
}

REV = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}

# Landmark positions in Taxi-v3 (fixed)
LANDMARKS = {
    0: (0, 0),
    1: (0, 4),
    2: (4, 0),
    3: (4, 3),
}

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
    def __init__(self, fail_prob: float = 0.0, seed: int = 0):
        self.env = gym.make("Taxi-v3")
        self.seed = seed
        self.rng = random.Random(seed)
        self.fail_prob = float(fail_prob)
        self.state, _ = self.env.reset(seed=seed)

        try:
            self.decode = self.env.unwrapped.decode
            self.encode = self.env.unwrapped.encode
        except AttributeError:
            self.decode = lambda i: tuple(np.unravel_index(i, (5, 5, 5, 4)))
            self.encode = lambda r, c, p, d: np.ravel_multi_index((r, c, p, d), (5, 5, 5, 4))

        self._fix_initial_passenger()

    def _fix_initial_passenger(self):
        r, c, p, d = self.decode(self.state)
        if p == 4:
            p = self.rng.choice([0, 1, 2, 3])
            self.state = self.encode(r, c, p, d)

    def reset(self):
        self.state, _ = self.env.reset(seed=self.seed)
        self._fix_initial_passenger()
        return self.state

    def get_state_predicates(self):
        r, c, p, d = self.decode(self.state)
        return {
            "taxi_at": (int(r), int(c)),
            "passenger_loc": int(p),
            "destination": int(d),
        }

    def _inject_failure(self) -> bool:
        return self.rng.random() < self.fail_prob
    
    def _blocked(self, r, c, direction):
        """Check if movement is blocked by walls or bounds."""
        # Origin walls
        if direction in GLOBAL_WALLS.get((r, c), set()):
            return True

        dr, dc = DELTAS[direction]
        nr, nc = r + dr, c + dc

        # Bounds check
        if not (0 <= nr < 5 and 0 <= nc < 5):
            return True

        # Destination reverse walls
        # if REV[direction] in GLOBAL_WALLS.get((nr, nc), set()):
        #     return True

        return False

    def step(self, action_name: str) -> StepResult:
        assert action_name in ACTIONS
        intended = ACTIONS[action_name]

        info = {
            "intended": action_name,
            "failed": False,
            "success": False,
            "done": False,
        }

        if intended in (0, 1, 2, 3) and self._inject_failure():
            info["failed"] = True
            return StepResult(self.state, -1, False, False, info)

        obs, reward, terminated, truncated, env_info = self.env.step(intended)
        self.state = obs

        info.update(env_info)

        if reward == 20:
            info["success"] = True
            info["done"] = True

        elif reward == -10:
            info["failed"] = True

        return StepResult(obs, reward, terminated, truncated, info)
    def simulate_action(self, action_name: str) -> dict:
       
        assert action_name in ACTIONS

        r, c, p, d = self.decode(self.state)

        # ------------------------------------------------------
        # MOVEMENT ACTIONS
        # ------------------------------------------------------
        if action_name in ("north", "south", "east", "west"):

            # Correct wall model
            if self._blocked(r, c, action_name):
                return {"failed": True}

            dr, dc = DELTAS[action_name]
            nr, nc = r + dr, c + dc

            # Bounds already enforced inside _blocked, 
            # but we keep this for safety against unexpected states
            if not (0 <= nr < 5 and 0 <= nc < 5):
                return {"failed": True}

            return {"failed": False, "next_state": (nr, nc, p, d)}

        # ------------------------------------------------------
        # PICKUP ACTION
        # ------------------------------------------------------
        if action_name == "pickup":

            # Passenger already in taxi → illegal
            if p == 4:
                return {"failed": True}

            # Taxi must be exactly at passenger's landmark
            if (r, c) != LANDMARKS[p]:
                return {"failed": True}

            # Successful pickup → p becomes 4 (in taxi)
            return {"failed": False, "next_state": (r, c, 4, d)}

        # ------------------------------------------------------
        # DROPOFF ACTION
        # ------------------------------------------------------
        if action_name == "dropoff":

            # Passenger must be in taxi
            if p != 4:
                return {"failed": True}

            # Must be at the destination landmark
            if (r, c) != LANDMARKS[d]:
                return {"failed": True}

            # Successful dropoff (Taxi-v3 handles reward in real step)
            return {"failed": False, "next_state": (r, c, p, d)}

        # ------------------------------------------------------
        # UNKNOWN ACTION (should not happen)
        # ------------------------------------------------------
        return {"failed": True}

# import gymnasium as gym
# import numpy as np
# import random
# from dataclasses import dataclass

# # Walls must match HTN + PDDL planners exactly
# GLOBAL_WALLS = {
#     (0, 0): {"west", "north"},
#     (0, 1): {"north"},
#     (0, 2): {"north"},
#     (0, 3): {"north"},
#     (0, 4): {"east", "north"},
#     (1, 0): {"west"},
#     (1, 4): {"east"},
#     (2, 0): {"west"},
#     (2, 4): {"east"},
#     (3, 0): {"west", "south"},
#     (3, 1): {"south"},
#     (3, 2): {"south"},
#     (3, 3): {"south"},
#     (3, 4): {"east", "south"},
# }

# DELTAS = {
#     "north": (-1, 0),
#     "south": (1, 0),
#     "east": (0, 1),
#     "west": (0, -1),
# }

# REV = {
#     "north": "south",
#     "south": "north",
#     "east": "west",
#     "west": "east",
# }

# # Landmark positions in Taxi-v3 (fixed)
# LANDMARKS = {
#     0: (0, 0),
#     1: (0, 4),
#     2: (4, 0),
#     3: (4, 3),
# }

# ACTIONS = {
#     "south": 0,
#     "north": 1,
#     "east": 2,
#     "west": 3,
#     "pickup": 4,
#     "dropoff": 5,
# }


# @dataclass
# class StepResult:
#     state: int
#     reward: float
#     terminated: bool
#     truncated: bool
#     info: dict


# class TaxiWrapper:
#     """
#     Thin wrapper around Gymnasium's Taxi-v3 that:
#     - Adds stochastic movement failure via fail_prob.
#     - Exposes a deterministic simulate_action() for lookahead.
#     - Keeps an explicit encoded state synchronized with env.unwrapped.s.
#     """

#     def __init__(self, fail_prob: float = 0.0, seed: int = 0):
#         self.env = gym.make("Taxi-v3")
#         self.seed = seed
#         self.rng = random.Random(seed)
#         self.fail_prob = float(fail_prob)

#         obs, _ = self.env.reset(seed=seed)
#         # state is the encoded integer; keep it in sync with env.unwrapped.s
#         self.state = obs
#         self.env.unwrapped.s = obs

#         try:
#             self.decode = self.env.unwrapped.decode
#             self.encode = self.env.unwrapped.encode
#         except AttributeError:
#             # Fallback if Gym API changes
#             self.decode = lambda i: tuple(np.unravel_index(i, (5, 5, 5, 4)))
#             self.encode = lambda r, c, p, d: np.ravel_multi_index(
#                 (r, c, p, d), (5, 5, 5, 4)
#             )

#         self._fix_initial_passenger()

#     # ----------------------------------------------------------------------
#     # Internal state helpers
#     # ----------------------------------------------------------------------

#     def _set_state(self, encoded_state: int) -> None:
#         """Set both wrapper state and Gym's internal Taxi-v3 state."""
#         self.state = encoded_state
#         # Gym's Taxi-v3 uses 's' as the encoded state
#         if hasattr(self.env, "unwrapped"):
#             self.env.unwrapped.s = encoded_state

#     def set_state(self, r: int, c: int, p: int, d: int) -> None:
#         """
#         Convenience method used in tests and debugging.
#         Sets both the wrapper state and the underlying env state.
#         """
#         encoded = self.encode(r, c, p, d)
#         self._set_state(encoded)

#     def _fix_initial_passenger(self):
#         """
#         Ensure the passenger starts at one of the 4 landmarks, not inside the taxi.
#         If Gym ever initializes with p == 4, re-sample p in {0,1,2,3}.
#         """
#         r, c, p, d = self.decode(self.state)
#         if p == 4:
#             p = self.rng.choice([0, 1, 2, 3])
#             self.set_state(r, c, p, d)

#     # ----------------------------------------------------------------------
#     # Public API
#     # ----------------------------------------------------------------------

#     def reset(self) -> int:
#         obs, _ = self.env.reset(seed=self.seed)
#         self._set_state(obs)
#         self._fix_initial_passenger()
#         return self.state

#     def get_state_predicates(self):
#         r, c, p, d = self.decode(self.state)
#         return {
#             "taxi_at": (int(r), int(c)),
#             "passenger_loc": int(p),
#             "destination": int(d),
#         }

#     # ----------------------------------------------------------------------
#     # Failure model and physics helpers
#     # ----------------------------------------------------------------------

#     def _inject_failure(self) -> bool:
#         """Return True when a stochastic movement failure occurs."""
#         return self.rng.random() < self.fail_prob

#     def _blocked(self, r: int, c: int, direction: str) -> bool:
#         """
#         Check if movement is blocked by walls or grid bounds.
#         Exactly matches the HTN & PDDL domain's wall model.
#         """
#         # Origin walls
#         if direction in GLOBAL_WALLS.get((r, c), set()):
#             return True

#         dr, dc = DELTAS[direction]
#         nr, nc = r + dr, c + dc

#         # Bounds
#         if not (0 <= nr < 5 and 0 <= nc < 5):
#             return True

#         # Destination reverse walls
#         if REV[direction] in GLOBAL_WALLS.get((nr, nc), set()):
#             return True

#         return False

#     # ----------------------------------------------------------------------
#     # Real environment step (with stochastic movement failure)
#     # ----------------------------------------------------------------------

#     def step(self, action_name: str) -> StepResult:
#         assert action_name in ACTIONS
#         intended = ACTIONS[action_name]

#         info = {
#             "intended": action_name,
#             "failed": False,
#             "success": False,
#             "done": False,
#         }

#         # Stochastic failure for movement only (not in simulate_action)
#         if intended in (0, 1, 2, 3) and self._inject_failure():
#             info["failed"] = True
#             # state does not change, reward -1, episode continues
#             return StepResult(self.state, -1.0, False, False, info)

#         # Let the real Taxi-v3 environment handle the transition
#         obs, reward, terminated, truncated, env_info = self.env.step(intended)
#         self._set_state(obs)

#         # Merge env_info from Taxi-v3
#         info.update(env_info)

#         # Success and failure semantics
#         if reward == 20:
#             info["success"] = True
#             info["done"] = True
#         elif reward == -10:
#             info["failed"] = True

#         return StepResult(obs, float(reward), bool(terminated), bool(truncated), info)

#     # ----------------------------------------------------------------------
#     # Deterministic simulation for lookahead
#     # ----------------------------------------------------------------------

#     def simulate_action(self, action_name: str) -> dict:
#         """
#         Deterministic feasibility check for lookahead:
#         - No RNG or fail_prob here.
#         - Correct wall checks.
#         - Correct pickup/dropoff preconditions.
#         Returns:
#             {"failed": True} on structural failure, or
#             {"failed": False, "next_state": (r, c, p, d)} on success.
#         """
#         assert action_name in ACTIONS

#         r, c, p, d = self.decode(self.state)

#         # ----- MOVEMENT -----
#         if action_name in ("north", "south", "east", "west"):
#             if self._blocked(r, c, action_name):
#                 return {"failed": True}

#             dr, dc = DELTAS[action_name]
#             nr, nc = r + dr, c + dc
#             return {"failed": False, "next_state": (nr, nc, p, d)}

#         # ----- PICKUP -----
#         if action_name == "pickup":
#             # passenger already in taxi → illegal
#             if p == 4:
#                 return {"failed": True}
#             # taxi must be exactly at passenger landmark
#             if (r, c) != LANDMARKS[p]:
#                 return {"failed": True}
#             # Structural success: passenger moves into taxi
#             return {"failed": False, "next_state": (r, c, 4, d)}

#         # ----- DROPOFF -----
#         if action_name == "dropoff":
#             # passenger must be in taxi
#             if p != 4:
#                 return {"failed": True}
#             # taxi must be at destination landmark
#             if (r, c) != LANDMARKS[d]:
#                 return {"failed": True}
#             # Structural success: for sim we keep p=4; env will award success
#             return {"failed": False, "next_state": (r, c, p, d)}

#         # Unknown or unsupported action
#         return {"failed": True}


# __all__ = [
#     "TaxiWrapper",
#     "StepResult",
#     "GLOBAL_WALLS",
#     "DELTAS",
#     "REV",
#     "LANDMARKS",
#     "ACTIONS",
# ]
