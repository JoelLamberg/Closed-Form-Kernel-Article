"""The 301 x 301 map grid (identical to the original run's results/config.json) and the cell list for this run:
the 2,723 map cells OUTSIDE the corner run (a < 10, a/Rc > 0.3) where some observation point lies behind the tangent
plane of some source point, so that the map's two-sided CBIM value differs from the one-sided (sigma = +1) one.
The list is in ../reference/cells.json; indices (i, j) are the map's, so results drop straight into map_cache.npz."""
import json, os
import numpy as np
CFG = dict(ns=301, na=301, smax=0.95, smin=1e-4, sgrid="log", amin=2.8, amax=60.0, ppw=7.0, nobs=49, tol=1e-3, floor=1e-8)
_HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = json.load(open(os.path.join(_HERE, "..", "reference", "cells.json")))["cells"]


def map_grid():
    s_vals = np.concatenate([[0.0], np.geomspace(CFG["smin"], CFG["smax"], CFG["ns"] - 1)])
    a_vals = np.geomspace(CFG["amin"], CFG["amax"], CFG["na"])
    return a_vals, s_vals


def cell_tasks(stride=1):
    a_vals, s_vals = map_grid()
    out = []
    for c in CELLS[::stride]:
        i, j = c["i"], c["j"]
        assert abs(a_vals[i] - c["a"]) < 1e-9 and abs(s_vals[j] - c["s"]) < 1e-12, "grid mismatch"
        out.append((i, j, float(a_vals[i]), float(s_vals[j])))
    return out


def cost(t):                         # relative cost model (number of sources at ppw 7)
    a, s = t[2], t[3]
    return np.pi * a**2 * (7 / np.sqrt(1 - s**2))**2


def fname(out, i, j):
    return os.path.join(out, f"pt_{i:03d}_{j:03d}.json")
