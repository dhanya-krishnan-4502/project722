import pytest

from envs.taxi_env_wrapper import (
    TaxiWrapper,
    GLOBAL_WALLS,
    DELTAS,
    LANDMARKS,
    ACTIONS,
)


def decode(wrapper):
    return wrapper.decode(wrapper.state)


def set_state(wrapper, r, c, p, d):
    wrapper.state = wrapper.encode(r, c, p, d)



def test_simulate_action_ignores_fail_prob():
    env1 = TaxiWrapper(fail_prob=0.0, seed=42)
    env2 = TaxiWrapper(fail_prob=1.0, seed=42)

    r, c, p, d = decode(env1)
    set_state(env2, r, c, p, d)

    res1 = env1.simulate_action("north")
    res2 = env2.simulate_action("north")

    assert res1 == res2



def test_simulate_action_respects_walls_and_bounds():
    env = TaxiWrapper(fail_prob=0.0, seed=0)

    for (r, c), blocked_dirs in GLOBAL_WALLS.items():
        set_state(env, r, c, 0, 1)

        for direction in ["north", "south", "east", "west"]:
            result = env.simulate_action(direction)
            if direction in blocked_dirs:
                assert result["failed"] is True
            else:
                dr, dc = DELTAS[direction]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < 5 and 0 <= nc < 5):
                    assert result["failed"] is True
                else:
                    assert result["failed"] is False
                    assert "next_state" in result
                    nr2, nc2, p2, d2 = result["next_state"]
                    assert (nr2, nc2) == (nr, nc)


def test_simulate_action_pickup_preconditions():
    env = TaxiWrapper(fail_prob=0.0, seed=0)

    for p in range(4):
        pr, pc = LANDMARKS[p]
        wrong_r, wrong_c = (pr, pc + 1) if pc + 1 < 5 else (pr, pc - 1)
        set_state(env, wrong_r, wrong_c, p, 0)
        res = env.simulate_action("pickup")
        assert res["failed"] is True

        set_state(env, pr, pc, p, 0)
        res = env.simulate_action("pickup")
        assert res["failed"] is False
        nr, nc, p2, d2 = res["next_state"]
        assert (nr, nc) == (pr, pc)
        assert p2 == 4   



def test_simulate_action_dropoff_preconditions():
    env = TaxiWrapper(fail_prob=0.0, seed=0)

    for d in range(4):
        dr, dc = LANDMARKS[d]

        set_state(env, dr, dc, d, d)
        res = env.simulate_action("dropoff")
        assert res["failed"] is True

        set_state(env, 0, 0, 4, d)
        if (0, 0) != (dr, dc):
            res = env.simulate_action("dropoff")
            assert res["failed"] is True

        set_state(env, dr, dc, 4, d)
        res = env.simulate_action("dropoff")
        assert res["failed"] is False
        nr, nc, p2, d2 = res["next_state"]
        assert (nr, nc) == (dr, dc)
        assert p2 == 4  


def test_step_uses_failure_but_simulate_does_not():
    env = TaxiWrapper(fail_prob=1.0, seed=0)

    sim = env.simulate_action("south")
    assert "failed" in sim  

    result = env.step("south")
    assert result.info["failed"] is True


def test_step_moves_taxi_when_not_failed():
    env = TaxiWrapper(fail_prob=0.0, seed=0)
    r, c, p, d = decode(env)

    for direction in ["north", "south", "east", "west"]:
        dr, dc = DELTAS[direction]
        nr, nc = r + dr, c + dc
        blocked = (r, c) in GLOBAL_WALLS and direction in GLOBAL_WALLS[(r, c)]
        if not blocked and 0 <= nr < 5 and 0 <= nc < 5:
            before = (r, c)
            result = env.step(direction)
            r2, c2, _, _ = decode(env)
            assert (r2, c2) != before
            break
    else:
        pytest.fail("No available non-blocked move found from initial Taxi-v3 state.")


def test_get_state_predicates_matches_decode():
    env = TaxiWrapper(fail_prob=0.0, seed=123)
    r, c, p, d = decode(env)
    preds = env.get_state_predicates()
    assert preds["taxi_at"] == (r, c)
    assert preds["passenger_loc"] == p
    assert preds["destination"] == d
