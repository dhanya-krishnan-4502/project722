import os
import time
import re
from pyperplan.planner import search_plan
from pyperplan.search.breadth_first_search import breadth_first_search
from pyperplan.heuristics.blind import BlindHeuristic



from pyperplan.heuristics.blind import BlindHeuristic
from pddl.generate_problem import generate_problem
import tempfile
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


REV = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
DELTAS = {'north': (-1, 0), 'south': (1, 0), 'east': (0, 1), 'west': (0, -1)}
class PDDLPlanner:

    def __init__(self, domain_file="pddl/taxi_domain.pddl", heuristic="blind"):
        self.domain_file = domain_file
        self.heuristic = "blind"
        self._heuristic_obj = BlindHeuristic

        # Build connectivity ONCE
        self._connectivity_string = self._build_connectivity()

    
    def _build_connectivity(self):
        """
        Build adjacency consistent with Taxi-v3 walls.
        Only add (connected A B) if movement in that direction is allowed.
        """
        lines = []

        for (r, c), blocked in GLOBAL_WALLS.items():
            for direction, (dr, dc) in DELTAS.items():

                # Skip blocked directions
                if direction in blocked:
                    continue

                nr, nc = r + dr, c + dc
                if not (0 <= nr < 5 and 0 <= nc < 5):
                    continue

                # Ensure reverse movement isn't blocked on the neighbor
                if REV[direction] in GLOBAL_WALLS.get((nr, nc), set()):
                    continue

                # Add connectivity
                lines.append(f"(connected loc-{r}-{c} loc-{nr}-{nc})")
                # Optionally add reverse; but domain is directional, so don't add automatically

        return "\n      ".join(lines)

    def _create_problem_file(self, preds):
        """
        Use generate_problem() to create a valid PDDL problem file.
        Returns the temporary file path.
        """
        pddl_text = generate_problem(preds, self._connectivity_string)
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pddl")
        tf.write(pddl_text.encode("utf-8"))
        tf.close()
        return tf.name

    def _normalize_step(self, step):
        """
        Robustly extract the operator name + arguments as lowercase strings.
        Handles both pyperplan Operator objects and string fallback.
        """
        try:
            # Pyperplan Operator
            name = step.name.lower()
            params = []
            for arg in step.args:
                if hasattr(arg, "name"):
                    params.append(arg.name.lower())
                else:
                    params.append(str(arg).lower())
            return name, params

        except Exception:
            # fallback for string-based operators
            s = str(step).strip().lower()
            s = s.strip("()")
            parts = s.split()
            return parts[0], parts[1:]


    # ---------------------------------------------------------------------
    def _dir_from_locs(self, loc_from, loc_to):
        try:
            r1, c1 = map(int, loc_from.split("-")[1:])
            r2, c2 = map(int, loc_to.split("-")[1:])
        except:
            return None

        # Row decreases → north
        if r2 == r1 - 1:
            return "north"

        # Row increases → south
        if r2 == r1 + 1:
            return "south"

        # Column decreases → west
        if c2 == c1 - 1:
            return "west"

        # Column increases → east
        if c2 == c1 + 1:
            return "east"

        return None

    # ---------------------------------------------------------------------
    def plan(self, preds):
        """Produce a list of primitive taxi actions using Pyperplan."""
        from inspect import signature
        bfs_sig = signature(breadth_first_search)
        num_params = len(bfs_sig.parameters)

        # 1. Generate a fresh problem file for this state
        problem_file = self._create_problem_file(preds)

        # 2. Run pyperplan search
        try:
            if num_params == 1:
                # Older pyperplan: breadth_first_search(task)
                plan = search_plan(
                    self.domain_file,
                    problem_file,
                    breadth_first_search,
                    None,
                )
            else:
                # Newer pyperplan: breadth_first_search(task, heuristic)
                plan = search_plan(
                    self.domain_file,
                    problem_file,
                    breadth_first_search,
                    self._heuristic_obj,
                )
        except Exception as e:
            print("[PDDL ERROR]:", e)
            return []

        if not plan:
            return []

        # 3. Convert plan operators -> Taxi-v3 primitive actions
        env_actions = []

        for step in plan:
            s = str(step).lower()

            # ---- MOVE OPERATORS ----
            if "(move" in s:
                # Extract all loc-* tokens from the string
                # The FIRST TWO are the head of the operator:
                #   (move taxi1 loc-3-4 loc-2-4)
                locs = re.findall(r"loc-\d-\d", s)
                if len(locs) >= 2:
                    loc_from = locs[0]
                    loc_to = locs[1]
                    d = self._dir_from_locs(loc_from, loc_to)
                    if d is not None:
                        env_actions.append(d)
                    else:
                        print("[WARN] could not infer direction from:", loc_from, loc_to)

            # ---- PICKUP / DROPOFF ----
            elif "(pickup" in s:
                env_actions.append("pickup")

            elif "(dropoff" in s:
                env_actions.append("dropoff")

        # Debug prints for sanity
        print("\n[PDDL RAW PLAN]")
        for step in plan:
            print("  ", step)

        print("\n[PDDL ENV ACTIONS]")
        print(env_actions)

        return env_actions
