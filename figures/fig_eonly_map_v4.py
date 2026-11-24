"""Fig. 5 (v4, two panels): (a) one-sided CBIM (Smythe) and (b) Stratton-Chu with LPWA-estimated H, relative error
against Stratton-Chu with the exact surface fields over the 301x301 map. Each panel hatches its own points below
10x the accuracy of the reference. The former panel (c), two-sided / one-sided, is now Fig. S2 of the supplement.
Data: fig_eonly_data.mat (export_eonly_data.py)."""
import os
import numpy as np, scipy.io as sio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
# ---------------- STYLE
FS_BASE, FS_TITLE, FS_LABEL, FS_TICK, FS_CBAR = 9, 9, 9, 8, 9
FIGSIZE = (9.4, 4.3)
CMAP = "inferno"; HATCH_GREY = "#e4e3df"; INK, INK2, SURF = "#0b0b0b", "#52514e", "#ffffff"
plt.rcParams.update({"font.size": FS_BASE, "axes.labelsize": FS_LABEL, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF, "hatch.color": "#8a8984"})
# ---------------- data
d = sio.loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "figures", "fig_eonly_data.mat"), squeeze_me=True)
A, S, one, lp, fl = d["A"], d["S"], d["one"], d["lp"], d["fl"]
hide_a, hide_b = one <= 10 * fl, lp <= 10 * fl
def edges(v):
    lv = np.log(v); return np.exp(np.r_[lv[0] - (lv[1] - lv[0]) / 2, (lv[1:] + lv[:-1]) / 2, lv[-1] + (lv[-1] - lv[-2]) / 2])
ye = edges(A); xe = edges(S[1:]); xp = np.array([1.0e-5, 3.6e-5])          # planar strip left of the log axis
grey = LinearSegmentedColormap.from_list("g", [HATCH_GREY] * 2)
fig, ax = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)
en = LogNorm(1e-12, 1e-1)
for k, (Z, hide, title) in enumerate([(one, hide_a, "(a) one-sided CBIM (Smythe), E only"),
                                      (lp, hide_b, "(b) Stratton–Chu with LPWA-estimated H")]):
    Zm = np.ma.masked_where(hide | ~np.isfinite(Z), np.clip(Z, 1e-12, None))
    im = ax[k].pcolormesh(xe, ye, Zm[:, 1:], cmap=CMAP, norm=en, rasterized=True)
    ax[k].pcolormesh(xp, ye, Zm[:, :1], cmap=CMAP, norm=en, rasterized=True)
    H = np.ma.masked_where(~hide, np.ones_like(Z))
    ax[k].pcolor(xe, ye, H[:, 1:], cmap=grey, hatch="///", lw=0); ax[k].pcolor(xp, ye, H[:, :1], cmap=grey, hatch="///", lw=0)
    ax[k].axvline(6.0e-5, color=SURF, lw=4)
    ax[k].set_xscale("log"); ax[k].set_yscale("log"); ax[k].minorticks_off()
    ax[k].set_xlim(xp[0], xe[-1]); ax[k].set_ylim(ye[0], ye[-1])
    ax[k].set_xticks([1.9e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.95]); ax[k].set_xticklabels(["plane", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "0.1", "0.95"])
    ax[k].set_yticks([3, 5, 10, 20, 40, 60]); ax[k].set_yticklabels(["3", "5", "10", "20", "40", "60"])
    ax[k].set_xlabel("curvature  $a/R_c$"); ax[k].set_title(title, fontsize=FS_TITLE, loc="left")
    for sp in ax[k].spines.values(): sp.set_linewidth(0.6)
ax[0].set_ylabel("aperture radius  $a/\\lambda$")
cb = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.015); cb.set_label("relative error vs Stratton–Chu, exact H", fontsize=FS_CBAR)
fig.savefig("fig_eonly_map_v4.png", dpi=250, bbox_inches="tight"); fig.savefig("fig_eonly_map_v4.pdf", bbox_inches="tight")
print("saved fig_eonly_map_v4 | shown a/b", int((~hide_a).sum()), int((~hide_b).sum()))
