import pytest
from planners.pddl_planner2 import PDDLPlanner
from acting.acting_module import ActingModule
from envs.taxi_env_wrapper import TaxiWrapper


def test_acting_calls_pddlplanner_and_returns_tuple():
    """Basic smoke test: ActingModule must return 5 values."""
    env = TaxiWrapper(seed=42, fail_prob=0)
    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    actor = ActingModule(env, planner=planner, strategy="lazy")

    result = actor.run_episode(max_steps=10)

    assert isinstance(result, tuple), "Actor must return a tuple"
    assert len(result) == 5, "Actor must return (success, steps, replans, total_planning_time, avg_planning_time)"


def test_acting_receives_plan_from_pddlplanner():
    """Check that ActingModule receives a list of actions from PDDLPlanner."""
    env = TaxiWrapper(seed=42, fail_prob=0)
    preds = env.get_state_predicates()

    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")

    plan = planner.plan(preds)

    assert isinstance(plan, list), "PDDL planner should return a list of actions"


def test_actor_executes_at_least_one_action():
    """Actor must execute at least 1 step unless planner is empty."""
    env = TaxiWrapper(seed=42, fail_prob=0)
    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    actor = ActingModule(env, planner=planner)

    success, steps, replans, total_time, avg_time = actor.run_episode(max_steps=20)

    assert steps >= 1, "Actor should execute at least 1 step"
    assert total_time >= 0
    assert avg_time >= 0


def test_actor_triggers_replanning_on_failure():
    """If fail_prob is high, actor must replan at least once."""
    env = TaxiWrapper(seed=42, fail_prob=0.9)
    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    actor = ActingModule(env, planner=planner)

    success, steps, replans, total_time, avg_time = actor.run_episode(max_steps=20)

    assert replans >= 2, "Actor must replan when actions repeatedly fail"
    assert total_time >= 0


def test_plan_integration_pipeline_does_not_crash():
    """End-to-end pipeline test: PDDLPlanner + ActingModule must run."""
    env = TaxiWrapper(seed=42, fail_prob=0.05)
    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    actor = ActingModule(env, planner=planner)

    success, steps, replans, total_time, avg_time = actor.run_episode(max_steps=200)

    assert isinstance(success, bool)
    assert isinstance(steps, int)
    assert isinstance(replans, int)
    assert isinstance(total_time, float)
    assert isinstance(avg_time, float)
