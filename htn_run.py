from planners.htn_gtpyhop import HTNPlanner
from envs.taxi_env_wrapper import TaxiWrapper
from acting.acting_module import ActingModule

# --- Initialize environment and planner ---
env = TaxiWrapper(fail_prob=0.0, seed=42)
planner = HTNPlanner()
actor = ActingModule(env, planner, strategy="lazy")

# --- Initial State ---
preds = env.get_state_predicates()
decoded = list(env.decode(env.state))
print(" INITIAL STATE (HTN) ")
print(f"taxi location ={decoded[0:2]}, passenger loation={decoded[2]}, destination={decoded[3]}")
print(f"Predicates: {preds}\n")

# --- Plan Generation ---
plan = planner.plan(preds)
print(f"Planner: {planner.name}")
print(f"Plan ({len(plan)} steps): {plan}\n")

# --- Execute Plan ---
for action in plan:
    result = env.step(action)
print("FINAL STATE (HTN) ")
decoded_final = list(env.decode(env.state))
print(f"taxi location={decoded_final[0:2]}, passenger location={decoded_final[2]}, destination={decoded_final[3]}")
print(f"Success: {result.info.get('success', False)} Reward: {result.reward}")

