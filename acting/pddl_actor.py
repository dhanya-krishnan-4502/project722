# acting/pddl_actor.py
from typing import Tuple
from envs.taxi_env_wrapper import TaxiWrapper, StepResult
from planners.pddl_planner import PDDLPlanner


class PDDLActor:
    """
    Minimal actor for Sprint 2 pilot:
    - Calls PDDL planner once from the initial state.
    - Executes the returned env-level actions sequentially.
    - Reports initial state, plan, and final state.
    - No online replanning; that is left as future work.
    """

    def __init__(self, env: TaxiWrapper, planner: PDDLPlanner):
        self.env = env
        self.planner = planner

    def run_episode(self, max_steps: int = 200) -> Tuple[bool, int, int]:
        """
        Returns (success, steps, replans)
        For this pilot: replans will always be 1 (single call to PDDL planner).
        """
        steps = 0
        replans = 0

        # 1) Get current state as predicates for logging (even if PDDL uses static problem)
        preds = self.env.get_state_predicates()

        # 2) Call PDDL planner ONCE
        plan = self.planner.plan(preds)
        replans += 1

        print(f"[DEBUG] Initial predicates: {preds}")
        print(f"[DEBUG] PDDL (env-level) plan: {plan}")

        # 3) Execute the plan step-by-step
        success = False
        for a in plan:
            if steps >= max_steps:
                break

            result: StepResult = self.env.step(a)
            steps += 1

            taxi_row, taxi_col, pass_loc, dest_idx = self.env.decode(self.env.state)
            print(
                f"[STEP {steps:02d}] action={a:7s} "
                f"→ taxi=({taxi_row},{taxi_col}) pass_loc={pass_loc} dest={dest_idx} "
                f"reward={result.reward}"
            )

            # For now, we trust the environment's success flag
            if result.info.get("success", False):
                success = True
                break

        return success, steps, replans
