import os, time
from typing import List

class PDDLPlanner:
    name = "PDDL(PyPerplan)"

    def __init__(self, domain_path="pddl/domain_taxi.pddl", out_dir="pddl/problems"):
        self.domain_path = domain_path
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

    def _write_problem(self, preds) -> str:
        # Sprint 1: stub a problem file and return its path
        ts = int(time.time())
        prob_path = os.path.join(self.out_dir, f"problem_{ts}.pddl")
        with open(prob_path, "w") as f:
            f.write("(define (problem taxi-prob) (:domain taxi)\n")
            f.write("  ; TODO: encode initial state and goal\n")
            f.write(")\n")
        return prob_path

    def plan(self, preds) -> List[str]:
        # Sprint 1: return a naive plan that mirrors HTN’s Manhattan moves
        # so the acting module can run end-to-end. Replace with real pyperplan call in Sprint 2.
        taxi_r, taxi_c = preds["taxi_at"]
        # derive passenger and destination rows/cols
        landmark = {0:(0,0),1:(0,4),2:(4,0),3:(4,3)}
        plan = []
        if preds["passenger_loc"] != 4:
            pr, pc = landmark[preds["passenger_loc"]]
            # manhattan moves
            if pr < taxi_r: plan += ['north']*(taxi_r-pr)
            if pr > taxi_r: plan += ['south']*(pr-taxi_r)
            if pc < taxi_c: plan += ['west']*(taxi_c-pc)
            if pc > taxi_c: plan += ['east']*(pc-taxi_c)
            plan += ['pickup']
            taxi_r, taxi_c = pr, pc
        dr, dc = landmark[preds["destination"]]
        if dr < taxi_r: plan += ['north']*(taxi_r-dr)
        if dr > taxi_r: plan += ['south']*(dr-taxi_r)
        if dc < taxi_c: plan += ['west']*(taxi_c-dc)
        if dc > taxi_c: plan += ['east']*(dc-taxi_c)
        plan += ['dropoff']
        return plan
