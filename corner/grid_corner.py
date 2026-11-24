"""The 301 x 301 map grid (identical to the original run's results/config.json) and the corner selection.
Indices (i, j) are the map's, so corner results drop straight into map_cache.npz."""
import numpy as np, os
CFG = dict(ns=301, na=301, smax=0.95, smin=1e-4, sgrid="log", amin=2.8, amax=60.0, ppw=7.0, nobs=49, tol=1e-3, floor=1e-8)
A_MAX, S_MIN = 10.0, 0.3            # corner: a < 10 lambda and a/Rc > 0.3


def map_grid():
    s_vals = np.concatenate([[0.0], np.geomspace(CFG["smin"], CFG["smax"], CFG["ns"] - 1)])
    a_vals = np.geomspace(CFG["amin"], CFG["amax"], CFG["na"])
    return a_vals, s_vals


def corner_tasks(stride=1):
    a_vals, s_vals = map_grid()
    return [(i, j, float(a), float(s)) for i, a in enumerate(a_vals) for j, s in enumerate(s_vals)
            if a < A_MAX and s > S_MIN and i % stride == 0 and j % stride == 0]


def cost(t):                         # relative cost model (number of sources at ppw 7)
    a, s = t[2], t[3]
    return np.pi * a**2 * (7 / np.sqrt(1 - s**2))**2


def fname(out, i, j):
    return os.path.join(out, f"pt_{i:03d}_{j:03d}.json")
