"""Run first (CPU, about a minute). Writes nothing into the results folders.
 (A) LAS2c on exact plane waves (plane and spheres, oblique): surface H must be exact to < 1e-4.
 (B) Four corner-run cells: run_point_las must reproduce the verified one-sided, SC+LPWA and SC-vs-exact errors.
 (C) The two cheapest coarse cells: one-sided and SC+LPWA must reproduce the 301 x 301 map (merged one-sided map).
usage: python validate_map.py"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from las_core import las_H
from las_map_core import run_point_las
from grid_map import CFG, cell_tasks, cost
K = 2 * np.pi
R = lambda f: json.load(open(os.path.join(HERE, "..", "reference", f)))
run = lambda a, s: run_point_las(a, s, nobs=CFG["nobs"], tol=CFG["tol"], floor=CFG["floor"])
d = lambda x, y: abs(x - y) / abs(y)
worst = 0.0
print("(A) LAS2c on plane waves")
rng = np.random.default_rng(0)
for Rc, th in [(None, 30), (20.0, 0), (20.0, 30), (5.0, 30), (100.0, 45)]:
    sd = np.array([np.sin(np.radians(th)), 0, np.cos(np.radians(th))]); pol = np.array([np.cos(np.radians(th)), 0, -np.sin(np.radians(th))])
    if Rc is None:
        P = rng.uniform(-3, 3, (200, 3)); P[:, 2] = 0; N = np.tile([0, 0, 1.0], (200, 1)); c = None
    else:
        al, ph = rng.uniform(0, 0.5, 200), rng.uniform(0, 2 * np.pi, 200)
        N = np.stack([np.sin(al) * np.cos(ph), np.sin(al) * np.sin(ph), np.cos(al)], 1); c = np.zeros(3); P = Rc * N
    f = lambda Q: np.exp(1j * K * (Q @ sd))[:, None] * pol[None, :]
    E = f(P); Hx = np.cross(sd, E); e = np.linalg.norm(las_H(P, N, E, f, c, Rc) - Hx) / np.linalg.norm(Hx)
    worst_pw = e; print(f"  Rc={Rc} theta={th}: surface-H error {e:.1e}", flush=True)
    if e > 1e-4:
        print("VALIDATE FAIL (A)"); sys.exit(1)
print("(B) corner-run cells")
for c in R("corner_check_values.json"):
    r = run(c["a"], c["s"])
    e = [d(r[k], c[k]) for k in ("err_dbl_sc", "err_sclpwa_sc", "err_sc_exact")]; worst = max(worst, *e)
    print(f"  a={c['a']:5.2f} s={c['s']:.3f} ppw {r['ppw_used']}/{c['ppw_used']}: dbl ({e[0]:.1e}) LPWA ({e[1]:.1e}) SC-ex ({e[2]:.1e}) "
          f"| LAS2c {r['err_sclas_sc']:.3e} vs one-sided {r['err_dbl_sc']:.3e}", flush=True)
print("(C) cheapest coarse cells vs the map")
ref = R("map_cell_values.json")
for i, j, a, s in sorted([t for t in cell_tasks("coarse") if t[3] > 0], key=cost)[:2]:
    r = run(a, s); m = ref[f"{i:03d}_{j:03d}"]
    e = [d(r["err_sclpwa_sc"], m["err_sclpwa_sc"]), d(r["err_dbl_sc"], m["err_dbl_sc"])]; worst = max(worst, *e)
    print(f"  a={a:5.2f} s={s:.3f}: LPWA ({e[0]:.1e}) one-sided ({e[1]:.1e}) | LAS2c {r['err_sclas_sc']:.3e}", flush=True)
ok = worst < 2e-2
print(f"VALIDATE worst rel diff {worst:.1e} -> {'OK' if ok else 'FAIL (stop and report)'}"); sys.exit(0 if ok else 1)
