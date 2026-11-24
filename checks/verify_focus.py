import json, glob, math, os, sys, statistics as st
rows = [json.load(open(f)) for f in glob.glob(os.path.join(sys.argv[1], 'pt_*.json'))]
ok = [r for r in rows if r['status'] == 'ok']
print("points", len(rows), "ok", len(ok), {s: sum(r['status'] == s for r in rows) for s in {r['status'] for r in rows}})
print("settings seen:", {(r['tol'], r['floor'], r['nobs'], r['half_width'], r['rim_target']) for r in ok})
bad = [r for r in ok if any(isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in r.values())]
print("NaN/inf", len(bad), "| devices", {d: sum(r['device'] == d for r in ok) for d in ('cpu', 'gpu')}, "| ppw_used", sorted({r['ppw_used'] for r in ok}))
print("rim ratio range", min(r['rim_ratio'] for r in ok), max(r['rim_ratio'] for r in ok))
print("SC vs exact (floor) range %.1e .. %.1e" % (min(r['err_sc_exact'] for r in ok), max(r['err_sc_exact'] for r in ok)))
for m in ('dbl', 'lpwa', 'two'):
    u = [r.get(f'unc_{m}', 0) for r in ok]
    print(f"Richardson unc {m}: max {max(u):.1e}, >5%: {sum(x > 0.05 for x in u)}, median {st.median(u):.1e}")
print("last step change SC: max %.1e" % max(r['last_step_change_sc'] for r in ok))
vis = [r for r in ok if max(r['err_dbl_sc'], r['err_lpwa_sc']) > 10 * r['err_sc_exact']]
print("visible", len(vis))
rat = [r['err_dbl_sc'] / r['err_lpwa_sc'] for r in vis]; rx = [r['err_dbl_sc_extrap'] / r['err_lpwa_sc_extrap'] for r in vis]
ri = [r['err_dbl_sc_int'] / r['err_lpwa_sc_int'] for r in vis]
print(f"dbl/lpwa complex: max {max(rat):.3f} median {st.median(rat):.4f} | extrap max {max(rx):.3f} | intensity max {max(ri):.3f}")
ex = [r['err_dbl_exact'] < r['err_lpwa_exact'] for r in ok]; print(f"vs exact beam: dbl better in {sum(ex)}/{len(ok)}; worst ratio {max(r['err_dbl_exact']/r['err_lpwa_exact'] for r in ok):.3f}")
mid = [r['err_dbl_sc_mid'] / r['err_lpwa_sc_mid'] for r in ok]; print(f"mid plane dbl/lpwa max {max(mid):.3f}")
# dbl visible itself (above 10x floor)?
dv = [r for r in ok if r['err_dbl_sc'] > 10 * r['err_sc_exact']]; print("points where dbl itself exceeds 10x floor:", len(dv))
tw = [(r['err_two_sc'] / r['err_dbl_sc'], r) for r in ok]
print("two/dbl: max %.2f, points >1.01: %d, frac_sigma_neg>0: %d" % (max(t for t, _ in tw), sum(t > 1.01 for t, _ in tw), sum(r['frac_sigma_neg'] > 0 for r in ok)))
for t, r in sorted(tw, key=lambda x: -x[0])[:6]:
    print(f"   two/dbl {t:.2f}: NA {r['na']} a {r['a']} s {r['s']} {'cc' if r['concave'] else 'cv'} frac_neg {r['frac_sigma_neg']:.2f}")
print("min two/dbl %.3f" % min(t for t, _ in tw))
