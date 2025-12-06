Taxi-v3 Planning & Acting Framework
Comparing HTN vs. PDDL under Lazy Lookahead and Full Lookahead

This project evaluates how two planning paradigms—Hierarchical Task Networks (HTN) and PDDL classical planning—interact with two acting strategies in the Taxi-v3 environment:

RunLazyLookahead: reactive replanning

RunLookahead: proactive, simulation-based replanning

The goal is to understand the tradeoffs between reactivity, efficiency, robustness, and computational cost under stochastic action failures.

The full pipeline includes symbolic environment modeling, deterministic simulation, stochastic execution, unified acting logic, and automated experimental evaluation over 360 controlled runs.

Planners

HTN (GTPyhop):
Uses domain-structured methods (navigate, get_passenger, deliver_passenger) for fast, efficient planning.

PDDL (Pyperplan):
Uses a grounded PDDL domain and state-space search, providing full completeness and robustness.

Acting Strategies

RunLazyLookahead — replan only after execution failures

RunLookahead — simulate next action; replan proactively if simulation predicts failure

TaxiWrapper

A corrected, symbolic wrapper around Taxi-v3 that provides:

deterministic simulation for lookahead

stochastic failures only during execution

correct wall constraints & adjacency

proper pickup/dropoff rules

consistent mappings to/from Gym state

Project Structure
project/
│
├── planners/             # HTN + PDDL implementations
├── acting/               # Acting module (Lazy/Lookahead)
├── envs/                 # Taxi-v3 symbolic wrapper
├── pddl/                 # Domain + template problem
├── experiment_results/   # CSV + plots + per-run logs
├── tests/                # Unit tests (wrapper correctness)
├── run_experiments.py    # Main experiment driver
└── README.md

Running the Experiments
1. Install requirements
pip install -r requirements.txt

2. Run all experiment configurations
python run_experiments.py


This executes every combination of:

Planner: {HTN, PDDL}

Strategy: {lazy, lookahead}

Failure probability: {0, 0.05, 0.10}

30 random seeds

A total of 360 episodes are run, logged, and plotted.

3. View Results

All summary plots appear in:

experiment_results/
    success_rate.png
    steps.png
    replans.png
    planning_time_total.png
    planning_time_avg.png
    wallclock_time.png

Key Findings

HTN produces the shortest plans and is ~10× faster per planning call.

PDDL achieves 100% success, even under failure probability 0.10.

RunLookahead reduces wasted movement but increases replanning due to proactive failure detection.

Replans, steps, and total planning time all scale linearly with noise.

Both planners show similar replanning curves once the environment wrapper is corrected.


Issues and pull requests are welcome—whether for extending planners, adding new acting algorithms, or introducing new environments.
