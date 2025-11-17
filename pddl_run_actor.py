# pddl_run_actor.py
from envs.taxi_env_wrapper import TaxiWrapper
from planners.pddl_planner import PDDLPlanner
from acting.pddl_actor import PDDLActor


def main():
    # 1) Make a deterministic Taxi-v3 env
    env = TaxiWrapper(seed=42, fail_prob=0.0)

    # 2) Show initial decoded Gym state
    init = tuple(env.decode(env.state))
    print("\n🔵 INITIAL STATE (Taxi-v3 / Gym)")
    print(f"  taxi_at = ({init[0]}, {init[1]})")
    print(f"  passenger_loc = {init[2]}")
    print(f"  destination   = {init[3]}")

    # 3) PDDL planner (using your existing domain & problem)
    planner = PDDLPlanner(
        domain="pddl/taxi_domain.pddl",
        problem="pddl/taxi_problem.pddl",
    )

    # 4) Minimal PDDL-based actor (no dynamic replanning)
    actor = PDDLActor(env, planner)

    # 5) Run one episode
    success, steps, replans = actor.run_episode(max_steps=200)

    # 6) Final state
    final = tuple(env.decode(env.state))
    print("\n🔵 FINAL STATE (Taxi-v3 / Gym)")
    print(f"  taxi_at = ({final[0]}, {final[1]})")
    print(f"  passenger_loc = {final[2]}")
    print(f"  destination   = {final[3]}")

    print(f"\nResult: success={success}, steps={steps}, replans={replans}")


if __name__ == "__main__":
    main()
