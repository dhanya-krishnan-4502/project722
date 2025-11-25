

from planners.htn_gtpyhop import HTNPlanner
from envs.taxi_env_wrapper import TaxiWrapper

def sanity_check_env_alignment():
    print("\n=== SANITY CHECK: Taxi env ↔ wrapper ↔ planner ===")

    env = TaxiWrapper(fail_prob=0.0, seed=42)      
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

    # 3) Ensure passenger not “in taxi” to start (p=4)
    if p == 4:
        print("[WARN] Passenger was 'in taxi' (4) on reset; forcing a valid loc (0..3).")
        # force a valid start
        p = 0
        env.state = encode(tr, tc, p, d)
        tr, tc, p, d = decode(env.state)
    print(f"[DECODE CHECK] after fix: (r={tr}, c={tc}, pass={p}, dest={d})")

    # 4) One-step move check (does the env actually move when it should?)
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

    # 5) Compare planner walls (if available)
    try:
        from planners.htn_gtpyhop import GLOBAL_WALLS as PLANNER_WALLS
        REV = {'north':'south','south':'north','east':'west','west':'east'}
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



