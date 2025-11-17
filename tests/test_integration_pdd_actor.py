import pytest
from planners.pddl_planner2 import PDDLPlanner
from acting.acting_module import ActingModule
from envs.taxi_env_wrapper import TaxiWrapper


def test_acting_calls_pddlplanner_and_returns_tuple():
    """Basic smoke test: ActingModule must call planner.plan() and return tuple."""
    env = TaxiWrapper(seed=42, fail_prob=0)   # deterministic
    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    actor = ActingModule(env, planner=planner, strategy="lazy")

    result = actor.run_episode(max_steps=10)

    assert isinstance(result, tuple), "Actor must return (success, steps, replans)"
    assert len(result) == 3


def test_acting_receives_plan_from_pddlplanner():
    """Check that ActingModule receives a list of actions from PDDLPlanner."""
    env = TaxiWrapper(seed=42, fail_prob=0)
    preds = env.get_state_predicates()

    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")

    plan = planner.plan(preds)

    assert isinstance(plan, list), "PDDL planner should return a list of actions"


def test_actor_executes_at_least_one_action():
    """ActingModule must execute at least 1 action unless planner returns empty."""
    env = TaxiWrapper(seed=42, fail_prob=0)
    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    actor = ActingModule(env, planner=planner)

    success, steps, replans = actor.run_episode(max_steps=20)

    assert steps >= 1, "Actor should execute at least 1 step (unless planner empty)"


def test_actor_triggers_replanning_on_failure():
    """If fail_prob is high, actor must replan at least once."""
    env = TaxiWrapper(seed=42, fail_prob=0.9)    # FORCE many movement failures
    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    actor = ActingModule(env, planner=planner)

    success, steps, replans = actor.run_episode(max_steps=20)

    assert replans >= 2, "Actor must invoke replanning when actions fail"


def test_plan_integration_pipeline_does_not_crash():
    """End-to-end pipeline test: PDDLPlanner + ActingModule must run without error."""
    env = TaxiWrapper(seed=42, fail_prob=0.05)
    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    actor = ActingModule(env, planner=planner)

    success, steps, replans = actor.run_episode(max_steps=200)

    assert isinstance(success, bool)
    assert isinstance(steps, int)
    assert isinstance(replans, int)
