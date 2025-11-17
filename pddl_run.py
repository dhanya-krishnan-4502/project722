from pyperplan.planner import search_plan
from pyperplan.search.breadth_first_search import breadth_first_search
from envs.taxi_env_wrapper import TaxiWrapper

# --- Initialize Taxi-v3 wrapper ---
env = TaxiWrapper(seed=42)
decoded = tuple(env.decode(env.state))
print("=== INITIAL STATE (PDDL) ===")
print(f"Decoded: taxi_at={decoded[0:2]}, passenger_loc={decoded[2]}, destination={decoded[3]}\n")

# --- PDDL files ---
domain_file = "pddl/taxi_domain.pddl"
problem_file = "pddl/taxi_problem.pddl"

# --- Run planner (Pyperplan 2.1) ---
plan = search_plan(domain_file, problem_file, breadth_first_search)

print("Planner: PDDL (Pyperplan 2.1 – BFS)")
print(f"Plan length: {len(plan)}\n")

for i, step in enumerate(plan, 1):
    print(f"{i}. {step}")

# --- Optional: simulate plan in Taxi-v3 ---
for step in plan:
    act = str(step).lower()
    if "pickup" in act:
        env.step("pickup")
    elif "dropoff" in act:
        env.step("dropoff")
    elif "move" in act:
        env.step("north")  # placeholder

decoded_final = tuple(env.decode(env.state))
print("\n=== FINAL STATE (PDDL) ===")
print(f"Decoded: taxi_at={decoded_final[0:2]}, passenger_loc={decoded_final[2]}, destination={decoded_final[3]}")
