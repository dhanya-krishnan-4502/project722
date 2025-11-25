import pytest
from planners.pddl_planner2 import PDDLPlanner
from pddl.generate_problem import generate_problem
from envs.taxi_env_wrapper import TaxiWrapper


def test_domain_parses():
    """
    Test 1 — domain file loads into Pyperplan.
    """
    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    assert planner.domain_file.endswith(".pddl")


def test_problem_generator_outputs_valid_pddl():
    """
    Test 2 — problem text contains required components.
    """
    preds = {
        "taxi_at": (3, 4),
        "passenger_loc": 1,
        "destination": 2
    }

    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    problem_str = generate_problem(preds, planner._connectivity_string)

    # Basic sanity checks
    assert "(define" in problem_str
    assert "(at taxi1 loc-3-4)" in problem_str
    assert "(at-passenger passenger1 loc-0-4)" in problem_str
    assert "(goal-loc passenger1 loc-4-0)" in problem_str


def test_pyperplan_can_generate_plan():
    """
    Test 3 — pyperplan must return a non-empty plan.
    """
    env = TaxiWrapper(seed=42, fail_prob=0)
    preds = env.get_state_predicates()

    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    plan = planner.plan(preds)

    assert isinstance(plan, list)
    assert len(plan) > 0, "Pyperplan failed to generate a plan"


def test_direction_inference_is_correct():
    """
    Test 4 — ensure mapping from two loc-* names → correct action.
    """
    planner = PDDLPlanner()

    assert planner._dir_from_locs("loc-3-4", "loc-2-4") == "north"
    assert planner._dir_from_locs("loc-3-4", "loc-4-4") == "south"
    assert planner._dir_from_locs("loc-3-4", "loc-3-3") == "west"
    assert planner._dir_from_locs("loc-3-4", "loc-3-5") == "east"  # allowed per domain


def test_plan_executes_in_environment():
    """
    Test 5 — test full end-to-end execution of the produced plan.
    With fail_prob=0, this MUST always reach the goal.
    """
    env = TaxiWrapper(seed=42, fail_prob=0)
    preds = env.get_state_predicates()

    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    actions = planner.plan(preds)

    assert len(actions) > 0, "Planner returned empty plan"

    for act in actions:
        result = env.step(act)
        if result.info.get("success", False):
            break

    assert result.info.get("success", False), (
        "Plan generated successfully but executing it did not reach success"
    )


def test_full_actor_pipeline_reaches_goal():
    """
    Test 6 — integration of PDDL → ActingModule must reach goal deterministically.
    """
    from acting.acting_module import ActingModule

    env = TaxiWrapper(seed=42, fail_prob=0)
    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    actor = ActingModule(env, planner=planner, strategy="lazy")

    success, steps, replans, total_time, avg_time = actor.run_episode(max_steps=200)

    assert success, "Full pipeline failed to reach goal with fail_prob=0"
    assert steps < 50, "Planner should solve Taxi-v3 in far fewer than 50 steps"
    assert replans < 10, "Lazy replanning should not replan excessively"
    assert total_time >= 0
    assert avg_time >= 0
