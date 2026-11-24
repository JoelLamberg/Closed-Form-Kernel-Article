"""Step 1 (run first, CPU, a few minutes). Two checks, writes nothing into ../results/.
 (A) Four cheap corner cells WITH back-facing pairs, already computed in the verified corner run: run_point_plus must
     reproduce their doubled (sigma = +1), SC+LPWA and SC-vs-exact errors.
 (B) The two cheapest cells of THIS run: SC+LPWA and SC-vs-exact must reproduce the 301x301 map's values (those do not
     depend on sigma); the doubled error is printed next to the map's two-sided value for information.
usage: python validate.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from corner_core import run_point_plus
from grid_cells import CFG, cell_tasks, cost

R = lambda f: json.load(open(os.path.join(HERE, "..", "reference", f)))
run = lambda a, s: run_point_plus(a, s, nobs=CFG["nobs"], tol=CFG["tol"], floor=CFG["floor"])
d = lambda x, y: abs(x - y) / abs(y)
worst = 0.0
print("(A) corner-run cells (sigma = +1 path, back-facing pairs present)")
for c in R("corner_check_values.json"):
    r = run(c["a"], c["s"])
    e = [d(r[k], c[k]) for k in ("err_dbl_sc", "err_sclpwa_sc", "err_sc_exact")]; worst = max(worst, *e)
    print(f"  a={c['a']:5.2f} s={c['s']:.3f} ppw {r['ppw_used']}/{c['ppw_used']}: dbl {r['err_dbl_sc']:.4e} vs {c['err_dbl_sc']:.4e} "
          f"({e[0]:.1e}) | LPWA ({e[1]:.1e}) | SC-ex ({e[2]:.1e})", flush=True)
print("(B) this run's two cheapest cells vs the 301x301 map")
ref = R("map_cell_values.json")
for i, j, a, s in sorted(cell_tasks(), key=cost)[:2]:
    r = run(a, s); m = ref[f"{i:03d}_{j:03d}"]
    e = [d(r["err_sclpwa_sc"], m["err_sclpwa_sc"])]; worst = max(worst, *e)
    print(f"  a={a:5.2f} s={s:.3f} ppw {r['ppw_used']}/{int(m['ppw_used'])}: LPWA {r['err_sclpwa_sc']:.4e} vs map {m['err_sclpwa_sc']:.4e} "
          f"({e[0]:.1e}) | SC-ex {r['err_sc_exact']:.2e} vs map {m['err_sc_exact']:.2e} | one-sided {r['err_dbl_sc']:.3e}, "
          f"map two-sided {m['err_cbim_sc']:.3e} (info)", flush=True)
ok = worst < 2e-2
print(f"VALIDATE worst rel diff {worst:.1e} -> {'OK' if ok else 'FAIL (stop and report)'}")
sys.exit(0 if ok else 1)
