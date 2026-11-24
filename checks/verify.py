import sys, json, zipfile, math, collections
z = zipfile.ZipFile(sys.argv[1])
P = [json.loads(z.read(n)) for n in z.namelist() if '/pt_' in n and n.endswith('.json')]
print('points', len(P), 'unique ij', len({(p['i'], p['j']) for p in P}))
bad = [p for p in P if any(isinstance(v, float) and not math.isfinite(v) for v in p.values())]
print('non-finite', len(bad))
print('device', collections.Counter(p.get('device', 'cpu') for p in P))
print('levels', sorted(collections.Counter(p['levels'] for p in P).items()))
print('ppw', sorted(collections.Counter(p['ppw_used'] for p in P).items()))
ok = lambda e, p: e > 10 * p['err_sc_exact']
lp = [p for p in P if p['err_sclpwa_sc'] < p['err_cbim_sc']]
lp_plus = [p for p in P if p['err_sclpwa_sc'] < p['err_cbimplus_sc']]
print('SC+LPWA beats CBIM:', len(lp), '| vs Omega+:', len(lp_plus))
print('  of which CBIM err above 10x floor:', sum(ok(p['err_cbim_sc'], p) for p in lp))
print('  a range', min(p['a'] for p in lp), max(p['a'] for p in lp), ' s min', min(p['s'] for p in lp))
print('frac_sigma_neg==0:', sum(p['frac_sigma_neg'] == 0 for p in P))
r = [p['err_cbim_sc'] / p['err_cbimplus_sc'] for p in P if p['err_cbimplus_sc'] > 0]
print('both/Omega+ ratio min/max', min(r), max(r), ' cells ratio>1.01:', sum(x > 1.01 for x in r))
print('max est_rel_uncertainty', max(p['est_rel_uncertainty'] for p in P))
cb = [p for p in P if p['s'] > 0]
print('err_cbim_sc range (s>0)', min(p['err_cbim_sc'] for p in cb), max(p['err_cbim_sc'] for p in cb))
print('planar err_cbim_sc max', max(p['err_cbim_sc'] for p in P if p['s'] == 0))
# first-order in curvature check at a~20, small s
row = sorted([p for p in P if abs(p['a'] - 20) < 0.3 and 0 < p['s'] < 1e-2], key=lambda p: p['s'])
if len(row) > 2:
    p0, p1 = row[0], row[-1]
    print('slope dlog err/dlog s at a~%.1f:' % p0['a'], math.log(p1['err_cbim_sc'] / p0['err_cbim_sc']) / math.log(p1['s'] / p0['s']))
print('total CPU-h (wall per point sum)/3600', sum(p['seconds'] for p in P) / 3600)
