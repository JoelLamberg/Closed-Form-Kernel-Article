import json, glob, math, sys, os, statistics as st
R, REF, CELLS = sys.argv[1:4]
ref = json.load(open(REF)); cells = json.load(open(CELLS))
rows = [json.load(open(f)) for f in glob.glob(os.path.join(R, 'pt_*.json'))]
keys = {(r['i'], r['j']) for r in rows}
refk = {tuple(map(int, k.split('_'))) for k in ref}
print("cells.json keys", list(cells), "n", cells['n'])
print("files", len(rows), "unique", len(keys), "expected", len(ref), "missing", len(refk - keys), "extra", len(keys - refk))
bad = [r for r in rows if any(isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in r.values())]
print("NaN/inf", len(bad), "| nobs", {r['nobs'] for r in rows}, "| devices", {d: sum(r['device'] == d for r in rows) for d in ('cpu', 'gpu')})
ppw = {}
for r in rows: ppw[r['ppw_used']] = ppw.get(r['ppw_used'], 0) + 1
print("ppw used", dict(sorted(ppw.items())))
m = lambda r: ref[f"{r['i']:03d}_{r['j']:03d}"]
pm = [(r['ppw_used'], int(m(r)['ppw_used'])) for r in rows]
print("ppw same as map:", sum(a == b for a, b in pm), "| higher:", sum(a > b for a, b in pm), "| lower:", sum(a < b for a, b in pm))
print("a range", min(r['a'] for r in rows), max(r['a'] for r in rows), "| s range", min(r['s'] for r in rows), max(r['s'] for r in rows))
print("corner overlap (a<10 & s>0.3):", sum(r['a'] < 10 and r['s'] > 0.3 for r in rows))
vis = [r for r in rows if max(r['err_dbl_sc'], r['err_sclpwa_sc']) > 10 * r['err_sc_exact']]
print("visible", len(vis), "| dbl itself >10x floor:", sum(r['err_dbl_sc'] > 10 * r['err_sc_exact'] for r in rows))
rat = [r['err_sclpwa_sc'] / r['err_dbl_sc'] for r in vis]; rat_i = [r['err_sclpwa_sc_int'] / r['err_dbl_sc_int'] for r in vis]
print(f"LPWA/dbl complex: min {min(rat):.2f} median {st.median(rat):.1f} max {max(rat):.0f}; intensity min {min(rat_i):.2f} median {st.median(rat_i):.1f}")
print("vs exact field: dbl<lpwa", sum(r['err_dbl_exact'] < r['err_sclpwa_exact'] for r in rows))
u = [r['est_rel_uncertainty_dbl'] for r in rows]; ul = [r['est_rel_uncertainty_lpwa'] for r in rows]
print(f"Richardson unc dbl max {max(u):.1e} (>1%: {sum(x > .01 for x in u)}), lpwa max {max(ul):.1e}")
dl = [abs(r['err_sclpwa_sc'] / m(r)['err_sclpwa_sc'] - 1) for r in rows]
de = [abs(r['err_sc_exact'] / m(r)['err_sc_exact'] - 1) for r in rows]
print(f"LPWA vs map max rel diff {max(dl):.1e} (>1e-3: {sum(x > 1e-3 for x in dl)}); SC-exact vs map {max(de):.1e}")
fs = [abs(r['frac_sigma_neg'] / m(r)['frac_sigma_neg'] - 1) for r in rows if m(r)['frac_sigma_neg'] > 0]
print(f"frac_sigma_neg vs map: map>0 in {len(fs)}, max rel diff {max(fs):.1e}; map==0 in {sum(m(r)['frac_sigma_neg']==0 for r in rows)}")
sg = [m(r)['err_cbim_sc'] / r['err_dbl_sc'] for r in vis]; op = [m(r)['err_cbimplus_sc'] / r['err_dbl_sc'] for r in vis]
print(f"two-sided/one-sided: min {min(sg):.4f} median {st.median(sg):.4f} max {max(sg):.3f}; >1.01: {sum(x>1.01 for x in sg)}, <0.99: {sum(x<0.99 for x in sg)}")
print(f"Omega+/one-sided: min {min(op):.4f} max {max(op):.3f}; <0.99: {sum(x<0.99 for x in op)}")
print("frac_sigma_neg: median", st.median(r['frac_sigma_neg'] for r in rows), "max", max(r['frac_sigma_neg'] for r in rows))
w = min(vis, key=lambda r: r['err_sclpwa_sc'] / r['err_dbl_sc']); print("closest:", {k: w[k] for k in ('a','s','err_dbl_sc','err_sclpwa_sc','err_sc_exact','ppw_used')})
top = sorted(vis, key=lambda r: -m(r)['err_cbim_sc'] / r['err_dbl_sc'])[:5]
for r in top: print("  top 2s/1s", round(r['a'],2), round(r['s'],3), round(m(r)['err_cbim_sc']/r['err_dbl_sc'],4), "fneg", r['frac_sigma_neg'])
print("total compute h", sum(r['seconds'] for r in rows)/3600)
