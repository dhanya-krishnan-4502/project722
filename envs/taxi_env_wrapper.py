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

        if direction in GLOBAL_WALLS.get((r, c), set()):
            return True

        dr, dc = DELTAS[direction]
        nr, nc = r + dr, c + dc

        if not (0 <= nr < 5 and 0 <= nc < 5):
            return True


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

        if action_name in ("north", "south", "east", "west"):

            if self._blocked(r, c, action_name):
                return {"failed": True}

            dr, dc = DELTAS[action_name]
            nr, nc = r + dr, c + dc

            if not (0 <= nr < 5 and 0 <= nc < 5):
                return {"failed": True}

            return {"failed": False, "next_state": (nr, nc, p, d)}

        if action_name == "pickup":

            if p == 4:
                return {"failed": True}

            if (r, c) != LANDMARKS[p]:
                return {"failed": True}

            return {"failed": False, "next_state": (r, c, 4, d)}

        if action_name == "dropoff":

            if p != 4:
                return {"failed": True}

            if (r, c) != LANDMARKS[d]:
                return {"failed": True}

            return {"failed": False, "next_state": (r, c, p, d)}

        return {"failed": True}

