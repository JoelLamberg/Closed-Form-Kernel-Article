"""After the map run: coverage and headline numbers; writes ../map_las_summary.csv.
usage: python summarize_map.py"""
import csv, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
K = 2 * np.pi
rows = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "..", "results_map", "pt_*.json"))]
print(f"cells done {len(rows)}")
if not rows:
    sys.exit(0)
keys = ["i", "j", "a", "s", "device", "ppw_used", "seconds", "err_sc_exact", "err_dbl_sc", "err_sclpwa_sc", "err_sclas_sc",
        "err_dbl_sc_int", "err_sclpwa_sc_int", "err_sclas_sc_int", "err_dbl_exact", "err_sclpwa_exact", "err_sclas_exact",
        "unc_dbl", "unc_sclpwa", "unc_sclas", "surfH_err_lpwa", "surfH_err_las"]
with open(os.path.join(HERE, "..", "map_las_summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
vis = [r for r in rows if max(r["err_dbl_sc"], r["err_sclas_sc"]) > 10 * r["err_sc_exact"]]
print(f"cells where one-sided or LAS2c is above 10x the reference accuracy: {len(vis)}")
las_wins = [r for r in vis if r["err_sclas_sc"] < r["err_dbl_sc"]]
print(f"  SC+LAS2c more accurate than one-sided: {len(las_wins)}; one-sided more accurate: {len(vis) - len(las_wins)}")
print(f"  one-sided beats SC+LPWA: {sum(r['err_dbl_sc'] < r['err_sclpwa_sc'] for r in vis)} of {len(vis)}")
for lo, hi in ((0, 1e-9), (1e-9, 0.01), (0.01, 0.1), (0.1, 0.3), (0.3, 1)):
    m = [r for r in vis if lo <= r["s"] < hi]
    if m:
        q = [r["err_sclas_sc"] / r["err_dbl_sc"] for r in m]
        print(f"  a/Rc in [{lo},{hi}): {len(m)} cells, LAS2c/one-sided median {np.median(q):.2g} range {min(q):.2g}..{max(q):.2g}, "
              f"LAS2c wins {sum(x < 1 for x in q)}")
x = [K * r["a"] / r["s"] / (K * r["a"] / 5.5)**2 for r in vis if r["s"] > 0]
y = [r["err_sclas_sc"] / r["err_dbl_sc"] for r in vis if r["s"] > 0]
if x:
    o = np.argsort(x); x, y = np.array(x)[o], np.array(y)[o]
    print("  LAS2c/one-sided vs kRc/(kw0)^2 (binned medians):")
    for lo, hi in ((0, 0.3), (0.3, 1), (1, 3), (3, 10), (10, 1e9)):
        m = (x >= lo) & (x < hi)
        if m.any():
            print(f"    kRc/(kw0)^2 in [{lo},{hi}): n={m.sum()}, median ratio {np.median(y[m]):.2g}")
print(f"  max Richardson uncertainty: one-sided {max(r.get('unc_dbl', 0) for r in rows):.1e}, LAS2c {max(r.get('unc_sclas', 0) for r in rows):.1e}")
print(f"  total compute {sum(r['seconds'] for r in rows)/3600:.1f} h (sum over cells)")
