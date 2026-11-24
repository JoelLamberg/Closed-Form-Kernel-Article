"""Cells: a subsample of the 301 x 301 map (indices i, j as in the map, so results line up with it).
--set coarse: every 10th index in a/lambda and a/Rc (31 x 31 = 961 cells, includes the plane column j = 0)
--set fine:   every 5th index (61 x 61 = 3,721 cells, contains all coarse cells; run coarse first)"""
import os
import numpy as np
CFG = dict(ns=301, na=301, smax=0.95, smin=1e-4, amin=2.8, amax=60.0, nobs=49, tol=1e-3, floor=1e-8)


def map_grid():
    s_vals = np.concatenate([[0.0], np.geomspace(CFG["smin"], CFG["smax"], CFG["ns"] - 1)])
    a_vals = np.geomspace(CFG["amin"], CFG["amax"], CFG["na"])
    return a_vals, s_vals


def cell_tasks(which="coarse"):
    st = {"coarse": 10, "fine": 5}[which]
    a_vals, s_vals = map_grid()
    return [(i, j, float(a_vals[i]), float(s_vals[j])) for i in range(0, 301, st) for j in range(0, 301, st)]


def cost(t):
    a, s = t[2], t[3]
    return np.pi * a**2 * (7 / np.sqrt(1 - s**2))**2


def fname(out, i, j):
    return os.path.join(out, f"pt_{i:03d}_{j:03d}.json")
