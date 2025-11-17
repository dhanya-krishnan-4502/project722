from typing import List, Tuple
from envs.taxi_env_wrapper import StepResult

class ActingModule:
    def __init__(self, env, planner, strategy: str = "lazy"):
        assert strategy in ("lazy", "lookahead")
        self.env = env
        self.planner = planner
        self.strategy = strategy
        self.plan: List[str] = []
        self.last_pos = None

    def _replan(self):

        preds = self.env.get_state_predicates()
        # print(f"[DEBUG] Replanning with state: {preds}")   # <— ADD THIS

        self.plan = self.planner.plan(preds)

    def run_episode(self, max_steps: int = 200) -> Tuple[bool, int, int]:
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
            if result.info.get("done", False):
                return result.info.get("success", False), steps, replans


            # ✅ stop on success or episode end
            if result.info.get("success", False) or result.info.get("done", False):
                return True, steps, replans

            # ✅ replan on failures or actual stuck movement
            if result.info.get("failed", False):
                self._replan()
                replans += 1
            elif act in ("north", "south", "east", "west") and self.last_pos == curr_pos:
                self._replan()
                replans += 1

            self.last_pos = curr_pos

        return False, steps, replans
