"""Step 1 (run first, CPU, minutes): on corner cells where no observer sees the back of the cap (frac_sigma_neg = 0 in the
map), CBIM with sigma = sign equals the doubled formula, so err_dbl_sc must reproduce the map's err_cbim_sc, and
err_sclpwa_sc must reproduce the map's value. Writes nothing into ../results/.
usage: python validate.py [--n 6]"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from corner_core import run_point_plus
from grid_corner import CFG, corner_tasks, cost

p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=6); args = p.parse_args()
ref = json.load(open(os.path.join(HERE, "..", "reference", "map_corner_values.json")))
cand = sorted((t for t in corner_tasks() if ref[f"{t[0]:03d}_{t[1]:03d}"]["frac_sigma_neg"] == 0), key=cost)
pick = [cand[int(q * (len(cand) - 1) / max(1, args.n - 1) * 0.3)] for q in range(args.n)]   # cheap end, spread out
worst = 0.0
for i, j, a, s in pick:
    r = run_point_plus(a, s, nobs=CFG["nobs"], tol=CFG["tol"], floor=CFG["floor"]); m = ref[f"{i:03d}_{j:03d}"]
    d1 = abs(r["err_dbl_sc"] - m["err_cbim_sc"]) / m["err_cbim_sc"]
    d2 = abs(r["err_sclpwa_sc"] - m["err_sclpwa_sc"]) / m["err_sclpwa_sc"]
    worst = max(worst, d1, d2)
    print(f"a={a:5.2f} s={s:.3f} ppw {r['ppw_used']}/{m['ppw_used']}: dbl {r['err_dbl_sc']:.4e} vs map CBIM {m['err_cbim_sc']:.4e} ({d1:.1e})"
          f" | LPWA {r['err_sclpwa_sc']:.4e} vs map {m['err_sclpwa_sc']:.4e} ({d2:.1e})", flush=True)
ok = worst < 2e-2
print(f"VALIDATE worst rel diff {worst:.1e} -> {'OK' if ok else 'FAIL (stop and report)'}")
sys.exit(0 if ok else 1)
