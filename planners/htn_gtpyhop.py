import gtpyhop
from collections import deque

# --- Domain initialization ---
domain = gtpyhop.Domain("taxi_htn")
gtpyhop.current_domain = domain
gtpyhop.current_domain.verbose = 0

# --- Static wall map (Taxi-v3 standard) ---
GLOBAL_WALLS = {
    (0,0): {'north', 'west'},
    (0,1): {'east', 'north'},
    (0,2): {'north', 'west'},
    (0,3): {'north'},
    (0,4): {'east', 'north'},
    (1,0): {'west'},
    (1,1): {'east'},
    (1,2): {'west'},
    (1,3): set(),
    (1,4): {'east'},
    (2,0): {'west'},
    (2,1): set(),
    (2,2): set(),
    (2,3): set(),
    (2,4): {'east'},
    (3,0): {'east', 'west'},
    (3,1): {'west'},
    (3,2): {'east'},
    (3,3): {'west'},
    (3,4): {'east'},
    (4,0): {'east', 'south', 'west'},
    (4,1): {'south', 'west'},
    (4,2): {'east', 'south'},
    (4,3): {'south', 'west'},
    (4,4): {'east', 'south'},
}


REV = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
DELTAS = {'north': (-1, 0), 'south': (1, 0), 'east': (0, 1), 'west': (0, -1)}
# for (r, c), dirs in list(GLOBAL_WALLS.items()):
#     for d in list(dirs):
#         dr, dc = DELTAS[d]
#         nr, nc = r + dr, c + dc
#         if 0 <= nr < 5 and 0 <= nc < 5:
#             opposite = REV[d]
#             GLOBAL_WALLS.setdefault((nr, nc), set()).add(opposite)
# print("[DEBUG WALLS SUMMARY]")
# for (r, c), dirs in sorted(GLOBAL_WALLS.items()):
#     print(f"  ({r},{c}): {dirs}")

# --- State factory ---
def make_initial_state(preds):
    s = gtpyhop.State("s0")
    s.taxi = preds["taxi_at"]
    s.passenger_loc = preds["passenger_loc"]
    s.destination = preds["destination"]
    s.landmarks = {0: (0, 0), 1: (0, 4), 2: (4, 0), 3: (4, 3)}
    return s

# --- Operators ---
def move(state, direction):
    r, c = state.taxi
    dr, dc = DELTAS[direction]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < 5 and 0 <= nc < 5): return False
    if direction in GLOBAL_WALLS.get((r, c), set()): return False
    if REV[direction] in GLOBAL_WALLS.get((nr, nc), set()): return False
    state.taxi = (nr, nc)
    return state

def pickup(state):
    if state.passenger_loc == 4: return False
    if state.taxi == state.landmarks[state.passenger_loc]:
        state.passenger_loc = 4
        return state
    return False

def dropoff(state):
    if state.passenger_loc != 4: return False
    if state.taxi == state.landmarks[state.destination]:
        state.passenger_loc = -1
        return state
    return False

gtpyhop.declare_operators(move, pickup, dropoff)

# --- BFS navigation helper ---
def valid_moves(cell):
    r, c = cell
    for d, (dr, dc) in DELTAS.items():
        nr, nc = r + dr, c + dc
        if not (0 <= nr < 5 and 0 <= nc < 5): continue
        if d in GLOBAL_WALLS.get((r, c), set()): continue
        if REV[d] in GLOBAL_WALLS.get((nr, nc), set()): continue
        yield d, (nr, nc)

def navigate(state, goal_pos):

    start, goal = state.taxi, tuple(goal_pos)
    # print(f"[DEBUG NAV] start={start} goal={goal}")
    # print(f"[DEBUG NAV] len(GLOBAL_WALLS)={len(GLOBAL_WALLS)} sample={(3,4)}→{GLOBAL_WALLS.get((3,4))}")

    if start == goal: return []
    q = deque([(start, [])])
    visited = {start}
    while q:
        pos, path = q.popleft()
        for d, nxt in valid_moves(pos):
            if nxt in visited: continue
            visited.add(nxt)
            new_path = path + [('move', d)]
            if nxt == goal: return new_path
            q.append((nxt, new_path))
    return []

# def get_passenger(state):
#     if state.passenger_loc == 4:
#         return []
#     path = navigate(state, state.landmarks[state.passenger_loc])
#     if not path:
#         return False
#     return path + [('pickup',)]
def get_passenger(state):
    if state.passenger_loc == 4:
        return []
    goal_pos = state.landmarks[state.passenger_loc]
    # print(f"[DEBUG] get_passenger: start={state.taxi}, goal={goal_pos}")
    path = navigate(state, goal_pos)
    # print(f"[DEBUG] navigate returned: {path}")
    if not path:
        return False
    return path + [('pickup',)]


def deliver_passenger(state):
    if state.passenger_loc != 4:
        return False
    path = navigate(state, state.landmarks[state.destination])
    if not path:
        return False
    return path + [('dropoff',)]

def transport_passenger(state):
    return [('get_passenger',), ('deliver_passenger',)]

def root_task(state):
    return [('transport_passenger',)]


gtpyhop.declare_actions(move, pickup, dropoff)
gtpyhop.declare_task_methods('get_passenger', get_passenger)
gtpyhop.declare_task_methods('deliver_passenger', deliver_passenger)
gtpyhop.declare_task_methods('transport_passenger', transport_passenger)
gtpyhop.declare_task_methods('root_task', root_task)

# --- Planner interface ---
class HTNPlanner:
    name = "HTN(GTPyhop)"

    def plan(self, preds):
        """Convert predicates → state → plan."""
        import sys, io
        # _stdout = sys.stdout
        # sys.stdout = io.StringIO()  # temporarily silence FP> logs

        s0 = make_initial_state(preds)
        tasks = [("root_task",)]
        plan = gtpyhop.find_plan(s0, tasks)  # <-- FIXED HERE

        # sys.stdout = _stdout  # restore output

        if not plan or plan is False:
            return []

        actions = []
        for step in plan:
            if isinstance(step, tuple):
                if step[0] == "move":
                    actions.append(step[1])
                else:
                    actions.append(step[0])
        return actions

