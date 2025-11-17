# walls.py
import gymnasium as gym
import numpy as np

REV = {"north": "south", "south": "north", "east": "west", "west": "east"}

def extract_walls_from_env(env):
    """
    Gymnasium 0.29 Taxi-v3 wall extractor.
    desc shape = (7, 11)
    - Grid rows live at desc rows 1..5  -> cell_r = 1 + r
    - Grid cols live at desc cols 1,3,5,7,9 -> cell_c = 2*c + 1
    Only vertical walls are encoded with '|'.
    North/south walls are borders only (r==0 and r==4).
    """
    desc = np.asarray(env.unwrapped.desc, dtype="S1")  # shape (7, 11)
    nrows, ncols = 5, 5
    walls = {(r, c): set() for r in range(nrows) for c in range(ncols)}

    for r in range(nrows):
        for c in range(ncols):
            cell_r = 1 + r            # <-- FIXED: no 2*
            cell_c = 2 * c + 1

            # borders
            if r == 0:
                walls[(r, c)].add("north")
            if r == nrows - 1:
                walls[(r, c)].add("south")

            # vertical internal walls (west/east)
            # west barrier sits at (cell_r, cell_c - 1)
            if cell_c - 1 >= 0 and desc[cell_r, cell_c - 1] == b"|":
                walls[(r, c)].add("west")
            # east barrier sits at (cell_r, cell_c + 1)
            if cell_c + 1 < desc.shape[1] and desc[cell_r, cell_c + 1] == b"|":
                walls[(r, c)].add("east")

    # make vertical walls bidirectional
    for (r, c), dirs in list(walls.items()):
        for d in list(dirs):
            dr = {"north": -1, "south": 1, "east": 0, "west": 0}[d]
            dc = {"north": 0, "south": 0, "east": 1, "west": -1}[d]
            nr, nc = r + dr, c + dc
            if 0 <= nr < nrows and 0 <= nc < ncols:
                if d in ("east", "west"):  # borders shouldn't be mirrored
                    walls[(nr, nc)].add(REV[d])

    return walls

if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    env.reset(seed=0)

    print("Gymnasium version:", gym.__version__)
    desc = np.asarray(env.unwrapped.desc, dtype="S1")
    print("desc shape:", desc.shape, "dtype:", desc.dtype)
    print("Raw desc:")
    for row in desc:
        print(row.tobytes().decode())

    walls = extract_walls_from_env(env)
    print("\n[DEBUG WALLS SUMMARY]")
    for (r, c) in sorted(walls):
        print(f"  ({r},{c}): {sorted(walls[(r,c)])}")

    # quick sanity: east<->west symmetry on internal walls
    ok = True
    for (r, c), dirs in walls.items():
        for d in dirs:
            if d not in ("east", "west"):
                continue
            dr = {"east": 0, "west": 0}[d]
            dc = {"east": 1, "west": -1}[d]
            nr, nc = r + dr, c + dc
            if 0 <= nr < 5 and 0 <= nc < 5:
                if REV[d] not in walls[(nr, nc)]:
                    ok = False
                    print(f"Asymmetry: {(r,c)} {d} vs {(nr,nc)} {REV[d]}")
    print("Bidirectional vertical walls:", ok)
