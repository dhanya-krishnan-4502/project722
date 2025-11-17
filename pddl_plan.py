# pddl_plan.py
import re
from inspect import signature
from pyperplan.planner import search_plan
from pyperplan.search.breadth_first_search import breadth_first_search
from pyperplan.heuristics.blind import BlindHeuristic
from envs.taxi_env_wrapper import TaxiWrapper


# === Helper functions ===
def normalize_step(step):
    """Return (action, args) for both Operator and string-based steps."""
    if hasattr(step, "name"):
        action = str(step.name).lower()
        args = [str(a) for a in getattr(step, "args", [])]
        return action, args

    s = str(step).strip()
    m = re.search(r"\(([^)]+)\)", s)
    inner = m.group(1) if m else s.strip("()")
    tokens = inner.split()
    action = tokens[0].lower()
    args = tokens[1:]
    return action, args


def dir_from_locs(loc_from, loc_to):
    """Infer direction between two location symbols like loc-3-4 → loc-2-4."""
    try:
        r1, c1 = map(int, loc_from.split("-")[1:])
        r2, c2 = map(int, loc_to.split("-")[1:])
    except Exception:
        return None

    if r2 == r1 - 1:
        return "south"
    if r2 == r1 + 1:
        return "north"
    if c2 == c1 - 1:
        return "west"
    if c2 == c1 + 1:
        return "east"
    return None


# === Run PDDL planning ===
domain = "pddl/taxi_domain.pddl"
problem = "pddl/taxi_problem.pddl"
env = TaxiWrapper(seed=42)

print("\nINITIAL STATE (PDDL)")
decoded = tuple(env.decode(env.state))
print(f" taxi location={decoded[0:2]}, passenger location={decoded[2]}, destination={decoded[3]}\n")

# --- Handle pyperplan version differences ---
sig = signature(breadth_first_search)
num_params = len(sig.parameters)

if num_params == 1:
    plan = search_plan(domain, problem, breadth_first_search, None)
else:
    plan = search_plan(domain, problem, breadth_first_search, BlindHeuristic)

print("Plan length:", len(plan))
print("ACTIONS:")
for i, step in enumerate(plan, 1):
    if isinstance(step, tuple):
        print(f"{i}. {' '.join(step)}")
    else:
        try:
            name = getattr(step, "name", str(step))
            params = [getattr(p, 'name', str(p)) for p in getattr(step, "parameters", getattr(step, "args", []))]
            print(f"{i}. {name} {' '.join(params)}")
        except Exception:
            print(f"{i}. {step}")

# --- Parse PDDL actions into environment actions ---
# --- Parse PDDL actions into environment actions ---
env_actions = []
for step in plan:
    action, args = normalize_step(step)
    s = str(step).lower()

    # Handle move actions robustly
    if "move" in s:
        # extract all location-like tokens (e.g., loc-3-4)
        locs = re.findall(r"loc-\d-\d", s)
        if len(locs) >= 2:
            d = dir_from_locs(locs[-2], locs[-1])
            if d:
                env_actions.append(d)

    elif "pickup" in s:
        env_actions.append("pickup")

    elif "dropoff" in s:
        env_actions.append("dropoff")

print("\nactions:", env_actions)

# --- Execute the plan in the Taxi environment ---
for a in env_actions:
    env.step(a)

# --- Show final state ---
decoded = tuple(env.decode(env.state))
print("\nFINAL STATE (PDDL)")
print(f"taxi location={decoded[0:2]}, passenger location={decoded[2]}, destination={decoded[3]}")
