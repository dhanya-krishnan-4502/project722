import os
import json
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from planners.htn_gtpyhop import HTNPlanner
from planners.pddl_planner2 import PDDLPlanner
from acting.acting_module import ActingModule
from envs.taxi_env_wrapper import TaxiWrapper


# CONFIGURATION
PLANNERS = {
    "HTN": HTNPlanner,
    "PDDL": lambda: PDDLPlanner(domain_file="pddl/taxi_domain.pddl")
}

STRATEGIES = ["lazy", "lookahead"]
FAIL_PROBS = [0.0, 0.05, 0.10]
SEEDS = list(range(30))    
GRID_SIZES = [5]        

OUTPUT_DIR = "experiment_results"
CSV_FILE = os.path.join(OUTPUT_DIR, "results.csv")


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/json_logs", exist_ok=True)



CSV_HEADER = [
    "planner","strategy","fail_prob","seed","grid_size",
    "success","steps","replans",
    "planning_time_total",
    "planning_time_avg",
    "wallclock_time"
]

def run_single_experiment(planner_name, strategy, fail_prob, seed, grid_size):

    # --- Initialize environment ---
    env = TaxiWrapper(seed=seed, fail_prob=fail_prob)

    # --- Initialize planner ---
    if planner_name == "HTN":
        planner = HTNPlanner()
    else:
        planner = PDDLPlanner(domain_file="pddl/taxi_domain.pddl")

    # --- Initialize acting module ---
    actor = ActingModule(env, planner, strategy=strategy)

    # --- Run episode and measure planning time ---
    t0 = time.time()
    success, steps, replans, plan_time_total, plan_time_avg = actor.run_episode()
    t1 = time.time()

    wallclock_time = round((t1 - t0), 4)

    record = {
        "planner": planner_name,
        "strategy": strategy,
        "fail_prob": fail_prob,
        "seed": seed,
        "grid_size": grid_size,
        "success": success,
        "steps": steps,
        "replans": replans,
        "planning_time_total": plan_time_total,
        "planning_time_avg": plan_time_avg,
        "wallclock_time": wallclock_time,
    }

    json_path = f"{OUTPUT_DIR}/json_logs/{planner_name}_{strategy}_p{fail_prob}_seed{seed}.json"
    with open(json_path, "w") as f:
        json.dump(record, f, indent=4)

    return record


def run_all_experiments():

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for planner_name, planner_ctor in PLANNERS.items():
            for strategy in STRATEGIES:
                for fail_prob in FAIL_PROBS:
                    for seed in SEEDS:
                        for grid_size in GRID_SIZES:

                            print(f"[RUN] {planner_name:<4} | {strategy:<9} | p={fail_prob:<4} | seed={seed}")

                            result = run_single_experiment(
                                planner_name, strategy, fail_prob, seed, grid_size
                            )

                            writer.writerow([
                            result["planner"],
                            result["strategy"],
                            result["fail_prob"],
                            result["seed"],
                            result["grid_size"],
                            result["success"],
                            result["steps"],
                            result["replans"],
                            result["planning_time_total"],
                            result["planning_time_avg"],
                            result["wallclock_time"],
                        ])



def generate_plots():

    import pandas as pd

    df = pd.read_csv(CSV_FILE)

    sns.set(style="whitegrid", font_scale=1.2)

    # 1. Success rate vs fail_prob
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df, x="fail_prob", y="success",
        hue="planner", style="strategy", markers=True
    )
    plt.title("Success Rate vs Failure Probability")
    plt.savefig(f"{OUTPUT_DIR}/success_rate.png", dpi=200)
    plt.close()

    # 2. Replans vs fail_prob
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df, x="fail_prob", y="replans",
        hue="planner", style="strategy", markers=True
    )
    plt.title("Replans vs Failure Probability")
    plt.savefig(f"{OUTPUT_DIR}/replans.png", dpi=200)
    plt.close()

    # 3. Steps vs fail_prob
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df, x="fail_prob", y="steps",
        hue="planner", style="strategy", markers=True
    )
    plt.title("Steps vs Failure Probability")
    plt.savefig(f"{OUTPUT_DIR}/steps.png", dpi=200)
    plt.close()

    # 4. Total planning time vs fail_prob
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df, x="fail_prob", y="planning_time_total",
        hue="planner", style="strategy", markers=True
    )
    plt.title("Total Planning Time vs Failure Probability")
    plt.savefig(f"{OUTPUT_DIR}/planning_time_total.png", dpi=200)
    plt.close()

    # 5. Average planning time per replan vs fail_prob
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df, x="fail_prob", y="planning_time_avg",
        hue="planner", style="strategy", markers=True
    )
    plt.title("Average Planning Time per Replan vs Failure Probability")
    plt.savefig(f"{OUTPUT_DIR}/planning_time_avg.png", dpi=200)
    plt.close()

    # 6. Wall-clock time of run vs fail_prob
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df, x="fail_prob", y="wallclock_time",
        hue="planner", style="strategy", markers=True
    )
    plt.title("Total Wallclock Episode Time vs Failure Probability")
    plt.savefig(f"{OUTPUT_DIR}/wallclock_time.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    run_all_experiments()
    generate_plots()
