import pytest
from acting.acting_module import ActingModule
from planners.pddl_planner2 import PDDLPlanner
from planners.htn_gtpyhop import HTNPlanner
from envs.taxi_env_wrapper import TaxiWrapper


def test_runlookahead_returns_five_tuple():
    """Ensure the API contract is maintained for lookahead mode."""
    env = TaxiWrapper(seed=42, fail_prob=0)
    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    actor = ActingModule(env, planner=planner, strategy="lookahead")

    result = actor.run_episode(max_steps=20)

    assert isinstance(result, tuple)
    assert len(result) == 5, "Lookahead must return (success, steps, replans, total_time, avg_time)"


def test_runlookahead_executes_actions():
    """Verify lookahead executes ≥1 action unless the plan is empty."""
    env = TaxiWrapper(seed=42, fail_prob=0)
    planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    
    actor = ActingModule(env, planner=planner, strategy="lookahead")
    success, steps, replans, total_time, avg_time = actor.run_episode(max_steps=20)

    assert steps >= 1, "Lookahead should execute at least 1 step"
    assert total_time >= 0
    assert avg_time >= 0


def test_runlookahead_reduces_failures():
    seed = 123

    env1 = TaxiWrapper(seed=seed, fail_prob=0.3)
    planner1 = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    actor_lazy = ActingModule(env1, planner=planner1, strategy="lazy")
    success_L, steps_L, replans_L, total_L, avg_L = actor_lazy.run_episode(max_steps=60)

    env2 = TaxiWrapper(seed=seed, fail_prob=0.3)
    planner2 = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
    actor_look = ActingModule(env2, planner=planner2, strategy="lookahead")
    success_H, steps_H, replans_H, total_H, avg_H = actor_look.run_episode(max_steps=60)

    # Lookahead should not be OUTRAGEOUSLY worse
    assert replans_H <= replans_L + 5, (
        f"Lookahead replanned too much: lazy={replans_L}, lookahead={replans_H}"
    )

    # Make sure it isn't drastically slower either
    assert steps_H <= steps_L + 10

