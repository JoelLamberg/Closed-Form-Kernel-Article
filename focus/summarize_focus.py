"""After the run: coverage, headline comparison per NA, CSV. usage: python summarize_focus.py"""
import csv, glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "..", "results", "pt_*.json"))]
ok = [r for r in rows if r["status"] == "ok"]
print(f"points {len(rows)} (computed {len(ok)}, skipped: " + ", ".join(f"{s} {sum(r['status']==s for r in rows)}" for s in sorted({r['status'] for r in rows} - {'ok'})) + ")")
if ok:
    keys = ["na", "a", "s", "concave", "F_from_rim", "ppw_used", "levels", "converged", "rim_ratio", "err_sc_exact", "err_dbl_sc",
            "err_lpwa_sc", "err_two_sc", "err_dbl_sc_int", "err_lpwa_sc_int", "err_dbl_exact", "err_lpwa_exact", "unc_dbl", "unc_lpwa",
            "err_dbl_sc_mid", "err_lpwa_sc_mid", "frac_sigma_neg", "seconds", "device"]
    with open(os.path.join(HERE, "..", "focus_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(sorted(ok, key=lambda r: (r["na"], r["a"], r["concave"], r["s"])))
    print(f"not converged within the ladder: {sum(not r['converged'] for r in ok)}; max Richardson unc dbl {max(r.get('unc_dbl',0) for r in ok):.1e}")
    print(f"{'NA':>4} {'n':>3} {'vis':>3} {'dbl<lpwa':>8} {'worst dbl/lpwa':>14} {'median':>8} {'two>1.01dbl':>11}")
    for na in sorted({r["na"] for r in ok}):
        sub = [r for r in ok if r["na"] == na]
        vis = [r for r in sub if max(r["err_dbl_sc"], r["err_lpwa_sc"]) > 10 * r["err_sc_exact"]]
        rat = sorted(r["err_dbl_sc"] / r["err_lpwa_sc"] for r in vis)
        print(f"{na:4.1f} {len(sub):3d} {len(vis):3d} {sum(x < 1 for x in rat):8d} {max(rat) if rat else float('nan'):14.3f} "
              f"{rat[len(rat)//2] if rat else float('nan'):8.3f} {sum(r['err_two_sc'] > 1.01*r['err_dbl_sc'] for r in vis):11d}")
    print(f"total compute {sum(r['seconds'] for r in rows)/3600:.1f} h (sum over points)")
