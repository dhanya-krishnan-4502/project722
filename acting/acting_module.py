from typing import List, Tuple
import time
from envs.taxi_env_wrapper import StepResult

class ActingModule:
    def __init__(self, env, planner, strategy: str = "lazy"):
        assert strategy in ("lazy", "lookahead")
        self.env = env
        self.planner = planner
        self.strategy = strategy
        self.plan: List[str] = []
        self.last_pos = None

        self.total_planning_time = 0.0
        self.replan_count = 0

    def _replan(self):
        start = time.time()
        preds = self.env.get_state_predicates()
        self.plan = self.planner.plan(preds)
        end = time.time()

        self.total_planning_time += (end - start)
        self.replan_count += 1

    def run_episode(self, max_steps: int = 200) -> Tuple[bool, int, int, float, float]:
        if self.strategy == "lookahead":
            success, steps, replans = self._run_lookahead(max_steps)
        else:
            success, steps, replans = self._run_lazy(max_steps)

        avg_time = (
            self.total_planning_time / self.replan_count
            if self.replan_count > 0 else 0.0
        )

        return success, steps, replans, self.total_planning_time, avg_time

    # LAZY strategy
    def _run_lazy(self, max_steps: int = 200):
        steps, replans = 0, 0

        self._replan()
        replans += 1

        while steps < max_steps:

            if not self.plan:
                self._replan()
                replans += 1
                if not self.plan:
                    break

            act = self.plan.pop(0)
            result: StepResult = self.env.step(act)
            steps += 1

            taxi_row, taxi_col, _, _ = self.env.decode(self.env.state)
            curr_pos = (taxi_row, taxi_col)

            if result.info.get("success", False) or result.info.get("done", False):
                return True, steps, replans

            if result.info.get("failed", False):
                self._replan()
                replans += 1
            elif act in ("north", "south", "east", "west") and self.last_pos == curr_pos:
                self._replan()
                replans += 1

            self.last_pos = curr_pos

        return False, steps, replans

    # LOOKAHEAD strategy
    def _run_lookahead(self, max_steps: int = 200):
        steps, replans = 0, 0

        self._replan()
        replans += 1

        while steps < max_steps:

            if not self.plan:
                self._replan()
                replans += 1
                if not self.plan:
                    break

            act = self.plan[0] 

            sim = self.env.simulate_action(act)
            if sim.get("failed", False):
                self._replan()
                replans += 1
                continue

            act = self.plan.pop(0)
            result: StepResult = self.env.step(act)
            steps += 1

            if result.info.get("success", False) or result.info.get("done", False):
                return True, steps, replans

            if result.info.get("failed", False):
                self._replan()
                replans += 1

        return False, steps, replans
