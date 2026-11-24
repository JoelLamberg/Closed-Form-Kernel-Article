"""After the run: coverage + headline numbers. Writes ../cells_summary.csv (one row per cell, map values alongside).
usage: python summarize.py"""
import csv, glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from grid_cells import cell_tasks
ref = json.load(open(os.path.join(HERE, "..", "reference", "map_cell_values.json")))
rows = []
for f in glob.glob(os.path.join(HERE, "..", "results", "pt_*.json")):
    r = json.load(open(f)); m = ref[f"{r['i']:03d}_{r['j']:03d}"]
    rows.append(dict(i=r["i"], j=r["j"], a=r["a"], s=r["s"], device=r["device"], ppw=r["ppw_used"], seconds=r["seconds"],
                     err_sc_exact=r["err_sc_exact"], err_dbl_sc=r["err_dbl_sc"], err_dbl_sc_int=r["err_dbl_sc_int"],
                     err_dbl_exact=r["err_dbl_exact"], err_sclpwa_sc=r["err_sclpwa_sc"], err_sclpwa_sc_int=r["err_sclpwa_sc_int"],
                     err_sclpwa_exact=r["err_sclpwa_exact"], unc_dbl=r.get("est_rel_uncertainty_dbl", float("nan")),
                     frac_sigma_neg=r["frac_sigma_neg"], map_err_cbim_sign=m["err_cbim_sc"],
                     map_err_cbim_omega_plus=m["err_cbimplus_sc"], map_err_sclpwa=m["err_sclpwa_sc"]))
print(f"cells done {len(rows)} / {len(cell_tasks())}")
if rows:
    with open(os.path.join(HERE, "..", "cells_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    vis = [r for r in rows if max(r["err_dbl_sc"], r["err_sclpwa_sc"]) > 10 * r["err_sc_exact"]]   # map plotting rule
    print(f"cells above 10x the SC-vs-exact floor: {len(vis)}")
    if vis:
        print(f"  one-sided (sigma=+1) beats SC+LPWA (complex):   {sum(r['err_dbl_sc'] < r['err_sclpwa_sc'] for r in vis)}")
        print(f"  one-sided (sigma=+1) beats SC+LPWA (intensity): {sum(r['err_dbl_sc_int'] < r['err_sclpwa_sc_int'] for r in vis)}")
        print(f"  min LPWA/one-sided ratio: {min(r['err_sclpwa_sc']/r['err_dbl_sc'] for r in vis):.1f}")
        print(f"  two-sided worse than one-sided by >1%: {sum(r['map_err_cbim_sign'] > 1.01*r['err_dbl_sc'] for r in vis)}, "
              f"better by >1%: {sum(r['map_err_cbim_sign'] < 0.99*r['err_dbl_sc'] for r in vis)}, "
              f"max two-sided/one-sided {max(r['map_err_cbim_sign']/r['err_dbl_sc'] for r in vis):.2f}")
    print(f"  max Richardson uncertainty (dbl): {max(r['unc_dbl'] for r in rows):.1e}")
    print(f"  total compute {sum(r['seconds'] for r in rows)/3600:.1f} h (sum over cells)")
