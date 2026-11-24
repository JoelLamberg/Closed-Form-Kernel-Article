"""Rebuild data/map_cache.npz (all 90,601 map points as 301x301 arrays) straight from the results zip, no unzipping.
usage: python build_cache.py [path/to/results_301x301.zip]     (default: ../../results_301x301.zip, i.e. the Drive folder)"""
import json, os, sys, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_map import grid

ZIP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "results_301x301.zip")
KEYS = ["err_cbim_sc", "err_cbimplus_sc", "err_sclpwa_sc", "err_sc_exact", "levels", "est_rel_uncertainty",
        "err_cbim_sc_int", "err_sclpwa_sc_int", "err_cbim_sc_max", "frac_sigma_neg", "ppw_used", "Nsrc"]

z = zipfile.ZipFile(ZIP)
cfg = json.loads(z.read("results/config.json"))
A, S = grid(cfg)
D = {k: np.full((len(A), len(S)), np.nan) for k in KEYS}
dev = np.zeros((len(A), len(S)), np.int8)                        # 1 = computed on GPU, 0 = CPU
n = 0
for name in z.namelist():
    if not name.startswith("results/pt_"):
        continue
    r = json.loads(z.read(name)); n += 1
    for k in KEYS:
        D[k][r["i"], r["j"]] = r.get(k, np.nan)
    dev[r["i"], r["j"]] = r.get("device") == "gpu"
missing = int(np.isnan(D["err_cbim_sc"]).sum())
out = os.path.join(HERE, "..", "data", "map_cache.npz")
np.savez(out, A=A, S=S, gpu=dev, **D)
print(f"{n} points read from {ZIP}; missing cells: {missing}; wrote {out}")
