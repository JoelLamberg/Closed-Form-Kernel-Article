"""Run before the focus sweep (CPU, a few minutes). On one point (NA 0.4, a = 10, plane, and NA 0.5, a = 10, convex 0.3)
with a short sampling ladder (5, 7), the new run_point_focus_las must reproduce the verified focus_core.run_point_focus
for SC exact, one-sided, SC+LPWA and two-sided (same code path, extra method only). Prints the LAS2c numbers.
usage: python validate_focus.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from focus_core import run_point_focus, Z_RIM
from focus_las_core import run_point_focus_las
CHOICE = json.load(open(os.path.join(HERE, "..", "reference", "focus_choice.json")))
worst = 0.0
for na, a, s, cc in [(0.4, 10.0, 0.0, False), (0.5, 10.0, 0.3, False)]:
    st, F = CHOICE[f"{na}_{a}_{s}_{int(cc)}"]
    kw = dict(zf=Z_RIM + F, ladder=(5, 7), nobs=49, half_width=3.0, tol=1e-4, floor=1e-10)
    r0 = run_point_focus(a, s, na, cc, **kw); r1 = run_point_focus_las(a, s, na, cc, **kw)
    for k in ("err_sc_exact", "err_dbl_sc", "err_lpwa_sc", "err_two_sc"):
        worst = max(worst, abs(r1[k] - r0[k]) / abs(r0[k]))
    print(f"NA {na} a {a} s {s}: dbl {r1['err_dbl_sc']:.3e} lpwa {r1['err_lpwa_sc']:.3e} las {r1['err_las_sc']:.3e} | "
          f"surface H: LPWA {r1['surfH_err_lpwa']:.1e} LAS2c {r1['surfH_err_las']:.1e} (ppw 7, not converged)", flush=True)
ok = worst < 1e-10
print(f"VALIDATE FOCUS worst rel diff vs verified code {worst:.1e} -> {'OK' if ok else 'FAIL (stop and report)'}"); sys.exit(0 if ok else 1)
