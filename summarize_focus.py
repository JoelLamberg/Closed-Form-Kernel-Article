"""After the focus run: headline numbers; writes ../focus_las_summary.csv.  usage: python summarize_focus.py"""
import csv, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "..", "results_focus", "pt_*.json"))]
ok = [r for r in rows if r.get("status") == "ok"]
print(f"points done {len(rows)}, computed {len(ok)}")
if not ok:
    sys.exit(0)
keys = ["na", "a", "s", "concave", "ppw_used", "seconds", "err_sc_exact", "err_dbl_sc", "err_lpwa_sc", "err_las_sc", "err_two_sc",
        "err_dbl_sc_int", "err_lpwa_sc_int", "err_las_sc_int", "err_dbl_exact", "err_lpwa_exact", "err_las_exact",
        "unc_dbl", "unc_lpwa", "unc_las", "surfH_err_lpwa", "surfH_err_las"]
with open(os.path.join(HERE, "..", "focus_las_summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(ok)
vis = [r for r in ok if max(r["err_dbl_sc"], r["err_las_sc"]) > 10 * r["err_sc_exact"]]
q = [r["err_las_sc"] / r["err_dbl_sc"] for r in vis]
print(f"points above 10x the reference accuracy: {len(vis)}; SC+LAS2c more accurate than one-sided at {sum(x < 1 for x in q)}")
if q:
    print(f"  LAS2c/one-sided: median {np.median(q):.2g}, range {min(q):.2g}..{max(q):.2g}")
    for na in sorted({r['na'] for r in vis}):
        m = [r["err_las_sc"] / r["err_dbl_sc"] for r in vis if r["na"] == na]
        print(f"  NA {na}: n={len(m)} median LAS2c/one-sided {np.median(m):.2g}, LAS2c wins {sum(x < 1 for x in m)}")
print(f"  total compute {sum(r['seconds'] for r in rows)/3600:.1f} h (sum over points)")
