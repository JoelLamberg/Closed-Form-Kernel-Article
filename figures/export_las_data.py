"""Collect the stronger-H (LAS2c) results into fig_las_data.mat for fig_las_crossover.py."""
import json, glob, numpy as np, scipy.io as sio
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = 2 * np.pi
R = [json.load(open(f)) for f in glob.glob(os.path.join(REPO, 'data', 'stronger_H', 'unz', 'results_map', 'pt_*.json'))]
g = lambda k: np.array([r[k] for r in R], float)
a, s = g('a'), g('s'); e1, e2, e3, fl = g('err_dbl_sc'), g('err_sclpwa_sc'), g('err_sclas_sc'), g('err_sc_exact')
F = [json.load(open(f)) for f in glob.glob(os.path.join(REPO, 'data', 'stronger_H', 'unz', 'results_focus', 'pt_*.json'))]
F = [r for r in F if r['status'] == 'ok']
h = lambda k: np.array([float(r[k]) for r in F])
sio.savemat(os.path.join(REPO, 'data', 'figures', 'fig_las_data.mat'), dict(
    map_a=a, map_s=s, map_err_one=e1, map_err_lpwa=e2, map_err_las=e3, map_err_ref=fl,
    map_unc_one=g('unc_dbl'), map_unc_las=g('unc_sclas'),
    foc_na=h('na'), foc_s=h('s'), foc_concave=h('concave'), foc_err_one=h('err_dbl_sc'),
    foc_err_lpwa=h('err_lpwa_sc'), foc_err_las=h('err_las_sc'), foc_err_ref=h('err_sc_exact')))
print('map', len(R), 'focus', len(F))
