import json, glob, math, sys, os, statistics as st
R = sys.argv[1]; ref = json.load(open(sys.argv[2]))
rows = [json.load(open(f)) for f in glob.glob(os.path.join(R, 'pt_*.json'))]
keys = {(r['i'], r['j']) for r in rows}
print("files", len(rows), "unique", len(keys), "expected", len(ref))
print("missing", len(set(tuple(map(int, k.split('_'))) for k in ref) - keys))
bad = [r for r in rows if any(isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in r.values())]
print("NaN/inf cells", len(bad), "| nobs set", {r['nobs'] for r in rows}, "| devices", {d: sum(r['device'] == d for r in rows) for d in ('cpu', 'gpu')})
ppw = {}; [ppw.__setitem__(r['ppw_used'], ppw.get(r['ppw_used'], 0) + 1) for r in rows]; print("ppw used", dict(sorted(ppw.items())))
vis = [r for r in rows if max(r['err_dbl_sc'], r['err_sclpwa_sc']) > 10 * r['err_sc_exact']]
inv = [r for r in rows if r not in vis]
print("visible", len(vis), "| below plotting rule", len(inv), "a range of hidden", (min(r['a'] for r in inv), max(r['a'] for r in inv)) if inv else "")
rat = [r['err_dbl_sc'] / r['err_sclpwa_sc'] for r in vis]; rat_i = [r['err_dbl_sc_int'] / r['err_sclpwa_sc_int'] for r in vis]
print(f"dbl/LPWA complex: max {max(rat):.3f} median {st.median(rat):.3f} min {min(rat):.4f}; intensity max {max(rat_i):.3f} median {st.median(rat_i):.3f}")
# extrapolated values too
rx = [r['err_dbl_sc_extrap'] / r['err_sclpwa_sc_extrap'] for r in vis if 'err_dbl_sc_extrap' in r]
print(f"dbl/LPWA (Richardson-extrapolated): max {max(rx):.3f}")
u = [r.get('est_rel_uncertainty_dbl', 0) for r in vis]; print(f"Richardson unc dbl: max {max(u):.1e}, >1%: {sum(x > 0.01 for x in u)}")
# consistency with the map where sigma never flips
z = [(r, ref[f"{r['i']:03d}_{r['j']:03d}"]) for r in rows if ref[f"{r['i']:03d}_{r['j']:03d}"]['frac_sigma_neg'] == 0]
d = [abs(r['err_dbl_sc'] / m['err_cbim_sc'] - 1) for r, m in z]; dl = [abs(r['err_sclpwa_sc'] / m['err_sclpwa_sc'] - 1) for r, m in z]
print(f"frac_sigma_neg=0 cells {len(z)}: dbl vs map CBIM max rel diff {max(d):.1e}; LPWA vs map {max(dl):.1e}")
dl_all = [abs(r['err_sclpwa_sc'] / ref[f"{r['i']:03d}_{r['j']:03d}"]['err_sclpwa_sc'] - 1) for r in rows]
print(f"all cells: LPWA vs map max rel diff {max(dl_all):.1e} (>1e-3: {sum(x > 1e-3 for x in dl_all)})")
# the 538 crossover cells (map: CBIM sign worse than LPWA)
cx = [(r, ref[f"{r['i']:03d}_{r['j']:03d}"]) for r in vis if ref[f"{r['i']:03d}_{r['j']:03d}"]['err_cbim_sc'] > ref[f"{r['i']:03d}_{r['j']:03d}"]['err_sclpwa_sc']]
print(f"map crossover cells (sign loses to LPWA) among visible: {len(cx)}; there dbl/LPWA max {max(r['err_dbl_sc']/r['err_sclpwa_sc'] for r,_ in cx):.3f}, "
      f"sign/dbl median {st.median(m['err_cbim_sc']/r['err_dbl_sc'] for r,m in cx):.1f} max {max(m['err_cbim_sc']/r['err_dbl_sc'] for r,m in cx):.0f}")
sg = [ref[f"{r['i']:03d}_{r['j']:03d}"]['err_cbim_sc'] / r['err_dbl_sc'] for r in vis]
print(f"sign/dbl over all visible: min {min(sg):.3f} max {max(sg):.0f}; cells where sign better than dbl by >1%: {sum(x < 0.99 for x in sg)}")
op = [ref[f"{r['i']:03d}_{r['j']:03d}"]['err_cbimplus_sc'] / r['err_dbl_sc'] for r in vis]
print(f"Omega+/dbl: min {min(op):.3f} max {max(op):.0f}; Omega+ better by >1%: {sum(x < 0.99 for x in op)}")
w = max(vis, key=lambda r: r['err_dbl_sc'] / r['err_sclpwa_sc']); print("closest cell:", {k: w[k] for k in ('a', 's', 'err_dbl_sc', 'err_sclpwa_sc', 'err_sc_exact', 'ppw_used')})
