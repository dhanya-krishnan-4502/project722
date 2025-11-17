import os
import time
import re
from pyperplan.planner import search_plan
from pyperplan.search.breadth_first_search import breadth_first_search
from pyperplan.heuristics.blind import BlindHeuristic



from pyperplan.heuristics.blind import BlindHeuristic
from pddl.generate_problem import generate_problem
import tempfile

class PDDLPlanner:

    def __init__(self, domain_file="pddl/taxi_domain.pddl", heuristic="blind"):
        self.domain_file = domain_file
        self.heuristic = "blind"
        self._heuristic_obj = BlindHeuristic

        # Build connectivity ONCE
        self._connectivity_string = self._build_connectivity()

    def _build_connectivity(self):
        """Return full 5×5 Manhattan adjacency PDDL text."""
        lines = []
        for r in range(5):
            for c in range(5):
                if r + 1 < 5:
                    lines.append(f"(connected loc-{r}-{c} loc-{r+1}-{c})")
                    lines.append(f"(connected loc-{r+1}-{c} loc-{r}-{c})")
                if c + 1 < 5:
                    lines.append(f"(connected loc-{r}-{c} loc-{r}-{c+1})")
                    lines.append(f"(connected loc-{r}-{c+1} loc-{r}-{c})")
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
        """Extract operator name + canonical ordered args."""
        # step is a pyperplan Operator
        try:
            name = step.name.lower()
            params = [p.lower() for p in step.args]  # correct order
            return name, params
        except:
            # Fallback for string version
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

        # 1. Generate problem
        problem_file = self._create_problem_file(preds)

        # 2. Run search
        try:
            if num_params == 1:
                plan = search_plan(self.domain_file, problem_file, breadth_first_search, None)
            else:
                plan = search_plan(self.domain_file, problem_file, breadth_first_search, self._heuristic_obj)
        except Exception as e:
            print("[PDDL ERROR]:", e)
            return []

        if not plan:
            return []

        # 3. Convert operators → env actions
        env_actions = []
        for step in plan:
            name, args = self._normalize_step(step)
            s = str(step).lower()

            if "move" in s:
                locs = [a for a in args if a.startswith("loc-")]
                if len(locs) >= 2:
                    d = self._dir_from_locs(locs[-2], locs[-1])
                    if d:
                        env_actions.append(d)

            elif name == "pickup":
                env_actions.append("pickup")

            elif name == "dropoff":
                env_actions.append("dropoff")

        print("\n[PDDL RAW PLAN]")
        for step in plan:
            print("  ", step)

        print("\n[PDDL ENV ACTIONS]")
        print(env_actions)

        return env_actions
