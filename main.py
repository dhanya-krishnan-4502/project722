# import random
# from planners.htn_gtpyhop import HTNPlanner
# from envs.taxi_env_wrapper import TaxiWrapper
# from acting.acting_module import ActingModule

# def run_one(planner, strategy="lazy", fail_prob=0.0, seed=42, max_steps=200):
#     """
#     Runs a single Taxi-v3 episode using the given planner and acting strategy.
#     Returns a summary dict with success flag, steps, and replans.
#     """
#     random.seed(seed)
#     env = TaxiWrapper(fail_prob=fail_prob, seed=seed)
#     actor = ActingModule(env, planner, strategy=strategy)
#     ok, steps, replans = actor.run_episode(max_steps=max_steps)


#     print(f"\n--- Running {planner.name} ({strategy}) ---")
#     success, steps, replans = actor.run_episode(max_steps=max_steps)

#     result = {
#         "planner": planner.name,
#         "strategy": strategy,
#         "fail_prob": fail_prob,
#         "seed": seed,
#         "success": success,
#         "steps": steps,
#         "replans": replans,
#     }
#     print(f"✅ Result: {result}\n")
#     return result


# def main():
#     """
#     Entry point: Runs both lazy and lookahead strategies on the HTN planner.
#     """
#     planner = HTNPlanner()

#     results = []
#     for strategy in ["lazy", "lookahead"]:
#         result = run_one(planner, strategy=strategy, fail_prob=0.1, seed=42)
#         results.append(result)

#     print("=== SUMMARY ===")
#     for r in results:
#         print(r)


# if __name__ == "__main__":
#     main()
# from planners.htn_gtpyhop import HTNPlanner

# planner = HTNPlanner()
# preds = {"taxi_at": (3, 4), "passenger_loc": 1, "destination": 2}  # typical start
# plan = planner.plan(preds)
# print("PLAN:", plan)

# test_wrapper.py
# from envs.taxi_env_wrapper import TaxiWrapper

# def test_taxi_wrapper_step_and_decode():
#     env = TaxiWrapper(seed=0, fail_prob=0.0)

#     # Reset and get initial decoded state
#     state = env.reset()
#     decoded = env.decode(state)
#     print("Initial decoded:", decoded)
#     assert len(decoded) == 4, "Decoded state must have 4 elements"

#     # Test a valid movement action
#     result = env.step("south")
#     assert hasattr(result, "info"), "StepResult must include info dict"
#     assert "success" in result.info, "StepResult.info must have success flag"
#     assert isinstance(result.reward, (float, int)), "Reward must be numeric"

#     # Force success by simulating dropoff
#     result.info["success"] = True
#     assert result.info.get("success", False) is True

# if __name__ == "__main__":
#     test_taxi_wrapper_step_and_decode()
#     print("✅ TaxiWrapper test passed")

# test_acting_module.py
# from acting.acting_module import ActingModule
# from envs.taxi_env_wrapper import TaxiWrapper

# class DummyPlanner:
#     """Mock planner for testing the actor without full HTN."""
#     def __init__(self):
#         self.calls = 0
#     def plan(self, preds):
#         self.calls += 1
#         # always produce a simple 3-step plan
#         return ["south", "pickup", "dropoff"]

# def test_acting_module_run():
#     env = TaxiWrapper(fail_prob=0.0, seed=0)
#     planner = DummyPlanner()
#     actor = ActingModule(env, planner, strategy="lazy")

#     success, steps, replans = actor.run_episode(max_steps=10)
#     print("Actor result:", success, steps, replans)

#     assert isinstance(success, bool)
#     assert steps > 0
#     assert replans >= 1
#     assert planner.calls >= 1, "Planner should be called at least once"

# if __name__ == "__main__":
#     test_acting_module_run()
#     print("✅ ActingModule test passed")


# test_system_integration.py
# from planners.htn_gtpyhop import HTNPlanner
# from envs.taxi_env_wrapper import TaxiWrapper
# from acting.acting_module import ActingModule

# def test_full_integration():
#     planner = HTNPlanner()
#     env = TaxiWrapper(seed=42, fail_prob=0.0)
#     actor = ActingModule(env, planner, strategy="lazy")

#     success, steps, replans = actor.run_episode(max_steps=200)
#     print(f"Integration → success={success}, steps={steps}, replans={replans}")

#     assert isinstance(success, bool)
#     assert steps > 0
#     assert replans >= 1

# if __name__ == "__main__":
#     test_full_integration()
#     print("✅ Full integration test passed")

from planners.htn_gtpyhop import HTNPlanner
from envs.taxi_env_wrapper import TaxiWrapper

def sanity_check_env_alignment():
    print("\n=== SANITY CHECK: Taxi env ↔ wrapper ↔ planner ===")

    env = TaxiWrapper(fail_prob=0.0, seed=42)      # no random failures
    planner = HTNPlanner()

    # 1) Decode/encode round-trip + landmarks
    decode = env.env.unwrapped.decode
    encode = env.env.unwrapped.encode
    locs = getattr(env.env.unwrapped, "locs", [(0,0),(0,4),(4,0),(4,3)])

    s = env.state
    tr, tc, p, d = decode(s)
    s2 = encode(tr, tc, p, d)

    print(f"[STATE] raw={s} decoded=(r={tr}, c={tc}, pass={p}, dest={d})")
    print(f"[ROUNDTRIP] encode(decode(s)) == s ? {s2 == s}")
    print(f"[LANDMARKS] gym = {locs}  (expected [(0,0),(0,4),(4,0),(4,3)])")

    # 2) Action set & mapping
    print(f"[ACTIONS] env.action_space.n = {env.env.action_space.n} (expect 6)")
    ACTIONS = {"south":0,"north":1,"east":2,"west":3,"pickup":4,"dropoff":5}
    print(f"[ACTIONS] wrapper mapping = {ACTIONS}")

    # 3) Ensure passenger not “in taxi” to start (p=4); if it is, fix and report
    if p == 4:
        print("[WARN] Passenger was 'in taxi' (4) on reset; forcing a valid loc (0..3).")
        # force a valid start
        p = 0
        env.state = encode(tr, tc, p, d)
        tr, tc, p, d = decode(env.state)
    print(f"[DECODE CHECK] after fix: (r={tr}, c={tc}, pass={p}, dest={d})")

    # 4) One-step move check (does the env actually move when it should?)
    #    Save/restore env internal state (TaxiEnv exposes .s)
    saved = env.env.unwrapped.s
    moved_ok = {}
    for name, code, (dr, dc) in [
        ("north", 1, (-1, 0)),
        ("south", 0, ( 1, 0)),
        ("east",  2, ( 0, 1)),
        ("west",  3, ( 0,-1)),
    ]:
        env.env.unwrapped.s = saved  # restore
        r0, c0, *_ = decode(env.env.unwrapped.s)
        obs, reward, term, trunc, _ = env.env.step(code)
        r1, c1, *_ = decode(obs)
        moved_ok[name] = ((r1 == r0 + dr) and (c1 == c0 + dc))
    env.env.unwrapped.s = saved  # final restore
    print(f"[MOVE TEST] delta match? {moved_ok} (False can be wall/boundary)")

    # 5) Compare planner walls (if available)
    try:
        from planners.htn_gtpyhop import GLOBAL_WALLS as PLANNER_WALLS
        # quick local recompute of what the planner thinks is valid from current cell
        REV = {'north':'south','south':'north','east':'west','west':'east'}
        # DELTAS = {'north':(-1,0),'south':(1,0),'east':(0,1),'west':(0,-1)}
        DELTAS = {
    "south": (1, 0),
    "north": (-1, 0),
    "east": (0, 1),
    "west": (0, -1),
}

        def planner_valid_moves(cell):
            r,c = cell
            val = []
            for d,(dr,dc) in DELTAS.items():
                nr,nc = r+dr, c+dc
                if not (0 <= nr < 5 and 0 <= nc < 5): 
                    continue
                if d in PLANNER_WALLS.get((r,c), set()): 
                    continue
                if REV[d] in PLANNER_WALLS.get((nr,nc), set()): 
                    continue
                val.append(d)
            return set(val)

        planner_ok = planner_valid_moves((tr,tc))
        # Empirical valid moves from env at same cell (moves that actually changed (r,c))
        env_valid = set([d for d, ok in moved_ok.items() if ok])
        print(f"[WALLS] planner-valid from {(tr,tc)} = {planner_ok}")
        print(f"[WALLS] env-valid     from {(tr,tc)} = {env_valid}")
        print(f"[WALLS] match? {planner_ok == env_valid}")
    except Exception as e:
        print(f"[WALLS] skipped (no GLOBAL_WALLS import): {e}")

    # 6) Quick planner path (no acting) to check it returns a reasonable plan
    preds = {"taxi_at": (int(tr), int(tc)), "passenger_loc": int(p), "destination": int(d)}
    plan = planner.plan(preds)
    # print(f"[PLANNER] plan length={len(plan)}  plan={plan[:12]}{'...' if len(plan)>12 else ''}")
    # print("=== SANITY CHECK DONE ===\n")

from planners.htn_gtpyhop import HTNPlanner
from acting.acting_module import ActingModule
from envs.taxi_env_wrapper import TaxiWrapper

def test_full_integration():

    env = TaxiWrapper(fail_prob=0.0, seed=42)
    env.reset()  # ✅ ensure valid passenger start here
    planner = HTNPlanner()
    actor = ActingModule(env, planner, strategy="lazy")
    # print("[INIT STATE]", env.decode(env.state))

    ok, steps, replans = actor.run_episode(max_steps=100)

    taxi_row, taxi_col, pass_loc, dest_idx = env.decode(env.state)
    # print(f"\n✅ Integration summary: success={ok}, steps={steps}, replans={replans}")
    # print(f"Final decoded state: taxi=({taxi_row},{taxi_col}), passenger_loc={pass_loc}, dest={dest_idx}")

    # Assertions
    assert steps > 0, "Taxi never moved"
    assert replans < steps, "Replanning every step → likely decode mismatch"
    assert ok, "Planner+Actor failed to reach goal"
from planners.htn_gtpyhop import HTNPlanner

if __name__ == "__main__":
    # sanity_check_env_alignment()
    test_full_integration()
    # planner = HTNPlanner()
    # preds = {"taxi_at": (3,4), "passenger_loc": 1, "destination": 2}
    # plan = planner.plan(preds)
    # print("PLAN:", plan)


    # env = TaxiWrapper(fail_prob=0.0, seed=42)
    # env.state = env.encode(3,4,1,2)              # match the test preds
    # r0,c0,_,_ = env.decode(env.state)

    # a0  = plan[0]                                 # e.g., 'north'
    # res = env.step(a0)
    # r1,c1,_,_ = env.decode(env.state)
    # print("step0:", a0, "pos:", (r0,c0), "→", (r1,c1), "reward:", res.reward)
    # assert (r1,c1) != (r0,c0), "Env didn’t move on first action"



