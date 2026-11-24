"""Fig. 8: one-sided CBIM versus Stratton-Chu with the second-order local estimate of H (LAS2c).
(a) map cells: error ratio LAS2c / one-sided against kRc/(kw0)^2, coloured by a/Rc.
(b) focused beams, a = 10 lambda: the same ratio against signed curvature, one colour per NA.
Data: fig_las_data.mat (export_las_data.py)."""
import os
import numpy as np, scipy.io as sio, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------- STYLE ----------------
FONT = 'DejaVu Sans'
FS_TICK, FS_LABEL, FS_PANEL, FS_LEGEND, FS_ANNOT, FS_CBAR = 8, 9, 10, 7, 7.5, 8
FIGSIZE = (7.0, 3.0)
MS_MAP, MS_FOC = 9, 26          # scatter marker areas
LW_REF, LW_LAW = 0.8, 1.0
CMAP = 'inferno'                # map cells by a/Rc
CMAP_NA = 'viridis'             # focused beams by NA
C_REF, C_LAW = '0.35', '#1f77b4'
LAW_COEF = 0.85                 # LAS2c/one-sided = 0.85 kRc/(kw0)^2 for a/Rc < 0.1
OUT = 'fig_las_crossover'
# ---------------------------------------
plt.rcParams.update({'font.family': FONT, 'font.size': FS_TICK})
d = sio.loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'figures', 'fig_las_data.mat'), squeeze_me=True)
K = 2 * np.pi

fig, (ax, bx) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw=dict(width_ratios=[1.25, 1]))

# (a) map
a, s, e1, e3, fl = d['map_a'], d['map_s'], d['map_err_one'], d['map_err_las'], d['map_err_ref']
m = (e1 > 10 * fl) & (e3 > 10 * fl) & (s > 0)
x = (K * a[m] / s[m]) / (K * a[m] / 5.5) ** 2
q = e3[m] / e1[m]
o = np.argsort(s[m])
sc = ax.scatter(x[o], q[o], c=s[m][o], cmap=CMAP, vmin=0, vmax=1, s=MS_MAP, lw=0)
xx = np.logspace(-1.5, 6, 50)
ax.plot(xx, np.ones_like(xx), color=C_REF, lw=LW_REF, ls='-')
ax.plot(xx, LAW_COEF * xx, color=C_LAW, lw=LW_LAW, ls='--', label=r'$0.85\,kR_c/(kw_0)^2$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.05, 2e4); ax.set_ylim(0.1, 1e4)
ax.set_xlabel(r'$kR_c/(kw_0)^2$', fontsize=FS_LABEL)
ax.set_ylabel('error ratio, LAS2c ÷ one-sided', fontsize=FS_LABEL)
ax.text(0.04, 0.95, 'one-sided CBIM more accurate', transform=ax.transAxes, fontsize=FS_ANNOT, va='top')
ax.text(0.96, 0.04, 'LAS2c more accurate', transform=ax.transAxes, fontsize=FS_ANNOT, ha='right')
ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.1), fontsize=FS_LEGEND, frameon=False)
cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05); cb.set_label(r'$a/R_c$', fontsize=FS_CBAR)
cb.ax.tick_params(labelsize=FS_TICK)
ax.text(-0.17, 1.03, '(a)', transform=ax.transAxes, fontsize=FS_PANEL, fontweight='bold')
ax.tick_params(labelsize=FS_TICK)

# (b) focused beams
na, fs, cc = d['foc_na'], d['foc_s'], d['foc_concave'].astype(bool)
f1, f3, ff = d['foc_err_one'], d['foc_err_las'], d['foc_err_ref']
sx = np.where(cc, fs, -fs)                       # concave positive, convex negative
qf = f3 / f1
res = (f1 > 10 * ff) | (f3 > 10 * ff)
cm = plt.get_cmap(CMAP_NA); nas = np.unique(na)
for k, v in enumerate(nas):
    col = cm(k / max(1, len(nas) - 1))
    mm = na == v
    bx.scatter(sx[mm & res], qf[mm & res], s=MS_FOC, color=col, lw=0)
    bx.scatter(sx[mm & ~res], qf[mm & ~res], s=MS_FOC, facecolors='none', edgecolors=col, lw=0.9)
bx.axhline(1, color=C_REF, lw=LW_REF)
bx.set_yscale('log'); bx.set_ylim(0.3, 1000); bx.set_xlim(-0.6, 1.0)
bx.set_xlabel(r'$a/R_c$ (convex $<0$, concave $>0$)', fontsize=FS_LABEL)
bx.set_ylabel('error ratio, LAS2c ÷ one-sided', fontsize=FS_LABEL)
h = [Line2D([], [], marker='o', ls='', color=cm(k / max(1, len(nas) - 1)), label=f'NA {v:.1f}', ms=4.5) for k, v in enumerate(nas)]
h += [Line2D([], [], marker='o', ls='', mfc='none', mec='0.3', label='below reference', ms=4.5)]
bx.legend(handles=h, fontsize=FS_LEGEND, frameon=False, loc='upper center', ncol=3, columnspacing=0.8, handletextpad=0.3)
bx.text(0.03, 0.03, 'LAS2c more accurate', transform=bx.transAxes, fontsize=FS_ANNOT, ha='left')
bx.text(0.03, 0.80, 'one-sided more accurate', transform=bx.transAxes, fontsize=FS_ANNOT, ha='left')
bx.text(-0.2, 1.03, '(b)', transform=bx.transAxes, fontsize=FS_PANEL, fontweight='bold')
bx.tick_params(labelsize=FS_TICK)

fig.tight_layout()
fig.savefig(OUT + '.png', dpi=300); fig.savefig(OUT + '.pdf')
print('wrote', OUT, 'map cells', m.sum(), 'focus', len(qf), 'resolved', res.sum())
