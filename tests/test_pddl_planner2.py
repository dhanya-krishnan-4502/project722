# tests/test_pddl_planner.py
import os
import pytest
from planners.pddl_planner2 import PDDLPlanner
from envs.taxi_env_wrapper import TaxiWrapper


# ----------------------------------------------------------------------
def test_problem_file_generation(tmp_path):
    """Test whether the planner correctly generates a PDDL problem file."""
    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")

    preds = {
        "taxi_at": (3, 4),
        "passenger_loc": 1,
        "destination": 2,
    }

    # NEW: planner creates its own temp file
    problem_file = planner._create_problem_file(preds)

    # Validate file exists
    assert os.path.exists(problem_file)
    assert problem_file.endswith(".pddl")

    with open(problem_file, "r") as f:
        text = f.read()

    # Basic sanity checks
    assert "(define" in text
    assert "(at taxi1 loc-3-4)" in text
    assert "(at-passenger" in text or "(in-taxi" in text


# ----------------------------------------------------------------------
def test_pddl_planner_returns_nonempty_plan():
    """Test that the PDDLPlanner produces at least one primitive action."""
    env = TaxiWrapper(seed=42, fail_prob=0)
    preds = env.get_state_predicates()

    planner = PDDLPlanner(
        domain_file="pddl/taxi_domain.pddl",
        heuristic="hadd"
    )

    plan = planner.plan(preds)

    assert isinstance(plan, list)
    assert len(plan) > 0, "PDDL planner returned an empty plan."


# ----------------------------------------------------------------------
def test_pddl_actions_are_primitive_strings():
    """Test that PDDL actions are converted into env-compatible primitives."""
    env = TaxiWrapper(seed=42, fail_prob=0)
    preds = env.get_state_predicates()

    planner = PDDLPlanner(
        domain_file="pddl/taxi_domain.pddl",
        heuristic="hadd"
    )

    actions = planner.plan(preds)

    allowed = {"north", "south", "east", "west", "pickup", "dropoff"}
    for a in actions:
        assert a in allowed, f"Illegal primitive action returned: {a}"


# ----------------------------------------------------------------------
def test_plan_executes_in_environment():
    """Ensure the PDDL action list can be executed in TaxiWrapper without crashing."""
    env = TaxiWrapper(seed=42, fail_prob=0)
    preds = env.get_state_predicates()

    planner = PDDLPlanner("pddl/taxi_domain.pddl", heuristic="hadd")
    plan = planner.plan(preds)

    # Execute plan
    for a in plan:
        result = env.step(a)
        assert isinstance(result.state, int)
        assert isinstance(result.info, dict)

    # After full plan, ensure Taxi moved somewhere meaningful
    final_r, final_c, _, _ = env.decode(env.state)
    assert 0 <= final_r < 5 and 0 <= final_c < 5


# ----------------------------------------------------------------------
def test_planner_works_when_passenger_in_taxi():
    """Test planning after pickup-like state."""
    env = TaxiWrapper(seed=42)
    preds = env.get_state_predicates()
    
    # force passenger into taxi
    preds["passenger_loc"] = 4

    planner = PDDLPlanner("pddl/taxi_domain.pddl")
    plan = planner.plan(preds)

    assert len(plan) > 0
    assert plan.count("pickup") == 0, "Planner incorrectly added pickup."


# ----------------------------------------------------------------------
def test_planner_works_after_movement():
    """Make one step in environment, then ensure planner replans correctly."""
    env = TaxiWrapper(seed=42)
    
    # take one movement
    env.step("south")
    preds = env.get_state_predicates()

    planner = PDDLPlanner("pddl/taxi_domain.pddl")
    plan = planner.plan(preds)

    assert len(plan) > 0
    assert any(a in ["north", "south", "east", "west", "pickup", "dropoff"]
               for a in plan)
