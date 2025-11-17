import re
from pddl.generate_problem import generate_problem

# -----------------------------------------------------------
# Helper: minimal connectivity so tests don't depend on full graph
# -----------------------------------------------------------

TEST_CONNECTIVITY = """
(connected loc-0-0 loc-0-1)
(connected loc-0-1 loc-0-0)
"""

# -----------------------------------------------------------
# 1. Return type sanity check
# -----------------------------------------------------------

def test_problem_returns_string():
    preds = {
        "taxi_at": (3, 4),
        "passenger_loc": 1,
        "destination": 2,
    }

    pddl = generate_problem(preds, TEST_CONNECTIVITY)

    assert isinstance(pddl, str)
    assert "(define" in pddl
    assert "(:domain taxi)" in pddl


# -----------------------------------------------------------
# 2. Taxi must appear at correct coordinates
# -----------------------------------------------------------

def test_problem_has_correct_taxi_location():
    preds = {
        "taxi_at": (3, 4),
        "passenger_loc": 0,
        "destination": 2,
    }

    pddl = generate_problem(preds, TEST_CONNECTIVITY)

    assert "(at taxi1 loc-3-4)" in pddl


# -----------------------------------------------------------
# 3. Passenger outside taxi → must appear at correct landmark
# -----------------------------------------------------------

def test_passenger_on_ground_correct_landmark():
    preds = {
        "taxi_at": (2, 3),
        "passenger_loc": 1,   # landmark index 1 → loc-0-4
        "destination": 0,
    }

    pddl = generate_problem(preds, TEST_CONNECTIVITY)

    assert "(at-passenger passenger1 loc-0-4)" in pddl


# -----------------------------------------------------------
# 4. Passenger inside taxi (index 4)
# -----------------------------------------------------------

def test_passenger_in_taxi():
    preds = {
        "taxi_at": (1, 1),
        "passenger_loc": 4,   # in the taxi
        "destination": 3,
    }

    pddl = generate_problem(preds, TEST_CONNECTIVITY)

    assert "(in-taxi passenger1)" in pddl
    assert "at-passenger" not in pddl    # should not appear anywhere


# -----------------------------------------------------------
# 5. All 25 location objects must appear
# -----------------------------------------------------------

def test_includes_all_location_objects():
    preds = {
        "taxi_at": (0, 0),
        "passenger_loc": 0,
        "destination": 2,
    }

    pddl = generate_problem(preds, TEST_CONNECTIVITY)

    for r in range(5):
        for c in range(5):
            assert f"loc-{r}-{c}" in pddl


# -----------------------------------------------------------
# 6. Connectivity string must be included
# -----------------------------------------------------------

def test_includes_connectivity():
    preds = {
        "taxi_at": (0, 0),
        "passenger_loc": 0,
        "destination": 2,
    }

    pddl = generate_problem(preds, TEST_CONNECTIVITY)

    assert "(connected loc-0-0 loc-0-1)" in pddl
    assert "(connected loc-0-1 loc-0-0)" in pddl


# -----------------------------------------------------------
# 7. Destination landmark must appear somewhere
# -----------------------------------------------------------

def test_destination_landmark_present():
    preds = {
        "taxi_at": (1, 1),
        "passenger_loc": 2,
        "destination": 3,
    }

    pddl = generate_problem(preds, TEST_CONNECTIVITY)

    # Landmarks: {0:(0,0), 1:(0,4), 2:(4,0), 3:(4,3)}
    assert "loc-4-3" in pddl
