LANDMARKS = {
    0: "loc-0-0",
    1: "loc-0-4",
    2: "loc-4-0",
    3: "loc-4-3",
}

def generate_problem(preds, connectivity_string):


    taxi_r, taxi_c = preds["taxi_at"]
    p_loc = preds["passenger_loc"]
    dest = preds["destination"]

    taxi_loc = f"loc-{taxi_r}-{taxi_c}"

    if p_loc == 4:
        passenger_init = "(in-taxi passenger1)"
    else:
        passenger_init = f"(at-passenger passenger1 {LANDMARKS[p_loc]})"

    goal_loc_decl = f"(goal-loc passenger1 {LANDMARKS[dest]})"

    all_locations = " ".join(f"loc-{r}-{c}" for r in range(5) for c in range(5))

    pddl = f"""
(define (problem taxi-instance)
  (:domain taxi)

  (:objects
     taxi1 - taxi
     passenger1 - passenger
     {all_locations} - location
  )

  (:init
     (at taxi1 {taxi_loc})
     {passenger_init}
     {goal_loc_decl}
     {connectivity_string}
  )

  (:goal (served passenger1))
)
"""
    return pddl.strip()
