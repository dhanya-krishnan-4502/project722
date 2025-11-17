# temp_main.py

from planners.htn_gtpyhop import HTNPlanner
from planners.pddl_planner2 import PDDLPlanner
from acting.acting_module import ActingModule
from envs.taxi_env_wrapper import TaxiWrapper

print("=== HTN ===")
env_htn = TaxiWrapper(seed=42, fail_prob=0.05)
htn = HTNPlanner()
actor_htn = ActingModule(env_htn, planner=htn, strategy="lazy")

success_htn, steps_htn, replans_htn = actor_htn.run_episode()
print("Success:", success_htn)
print("Steps:", steps_htn)
print("Replans:", replans_htn)

print("\n=== PDDL ===")
env_pddl = TaxiWrapper(seed=42, fail_prob=0.05)
pddl = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
actor_pddl = ActingModule(env_pddl, planner=pddl, strategy="lazy")

success_pddl, steps_pddl, replans_pddl = actor_pddl.run_episode()
print("Success:", success_pddl)
print("Steps:", steps_pddl)
print("Replans:", replans_pddl)
