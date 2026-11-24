"""Fig. 4 (v2, two panels): meridional (y = 0) cross-sections of the two test setups. (a) map test, CSP Gaussian beam
through a convex cap, observation plane a beyond the vertex; (b) focusing test, focused beam through a concave cap to
the focal plane. The former panel (c), the two-sided sign geometry, is now Fig. S1 of the supplement.
Data: fig_setup_data.mat (export_setup_data.py)."""
import os
import numpy as np, scipy.io as sio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# ---------------- STYLE
FS_BASE, FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT, FS_CBAR = 9, 9, 9, 8, 8.5, 9
FIGSIZE = (8.6, 4.9)
CMAP = "inferno"; INK, INK2 = "#0b0b0b", "#52514e"; CAP, OBS, NRM = "#ffffff", "#5fd3f3", "#9be29b"
LW_CAP, LW_OBS, LW_NRM, L_NRM = 2.2, 2.2, 1.3, 2.0
plt.rcParams.update({"font.size": FS_BASE, "axes.labelsize": FS_LABEL, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.labelcolor": INK, "figure.facecolor": "white"})
# ---------------- data
d = sio.loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "figures", "fig_setup_data.mat"), squeeze_me=True); ZR = float(d["Z_RIM"])
def cap_curve(a, s, concave=False, n=400):
    x = np.linspace(-a, a, n); Rc = a / s; c0 = np.sqrt(Rc**2 - a**2)
    return x, (ZR - c0 + np.sqrt(Rc**2 - x**2)) if not concave else (ZR + c0 - np.sqrt(Rc**2 - x**2))
def normals(ax_, a, s, concave, xs, L):
    Rc = a / s; c0 = np.sqrt(Rc**2 - a**2)
    for x in xs:
        if not concave:
            z = ZR - c0 + np.sqrt(Rc**2 - x**2); n = np.array([x, z - (ZR - c0)]) / Rc
        else:
            z = ZR + c0 - np.sqrt(Rc**2 - x**2); n = np.array([-x, (ZR + c0) - z]) / Rc
        ax_.annotate("", xy=(z + L * n[1], x + L * n[0]), xytext=(z, x), arrowprops=dict(arrowstyle="-|>", color=NRM, lw=LW_NRM))
fig, ax = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True, gridspec_kw=dict(width_ratios=[0.95, 0.62]))
nrm = LogNorm(1e-4, 1)
# (a) map test
a, s = float(d["a_a"]), float(d["a_s"]); zv, zo, L = float(d["a_zv"]), float(d["a_zo"]), float(d["a_L"])
X, Z = np.meshgrid(d["a_x"], d["a_z"], indexing="ij")
ax[0].pcolormesh(Z, X, d["a_I"], cmap=CMAP, norm=nrm, shading="auto", rasterized=True)
cx, cz = cap_curve(a, s); ax[0].plot(cz, cx, color=CAP, lw=LW_CAP)
normals(ax[0], a, s, False, np.linspace(-0.8 * a, 0.8 * a, 5), L_NRM)
ax[0].plot([zo, zo], [-L, L], color=OBS, lw=LW_OBS, ls="--")
ax[0].annotate("", xy=(zv, -1.35 * a), xytext=(zo, -1.35 * a), arrowprops=dict(arrowstyle="<->", color="white", lw=1))
ax[0].text(0.5 * (zv + zo), -1.28 * a, "$a$", color="white", ha="center", va="bottom")
ax[0].annotate("", xy=(ZR - 0.3, 0), xytext=(ZR - 0.3, -a), arrowprops=dict(arrowstyle="<->", color="white", lw=1))
ax[0].text(ZR + 0.1, -0.55 * a, "$a$", color="white", ha="left", va="center")
ax[0].text(0.3, 1.33 * a, "convex cap $\\Omega$, prescribed $\\mathbf{E}$", color="white", fontsize=FS_ANNOT)
ax[0].text(zo + 0.5, L + 0.4, "observation\nplane", color=OBS, fontsize=FS_ANNOT)
ax[0].set_title("(a) map test: Gaussian beam", loc="left", fontsize=FS_TITLE)
# (b) focusing test
a2, s2, na, zf, zv2 = float(d["b_a"]), float(d["b_s"]), float(d["b_na"]), float(d["b_zf"]), float(d["b_zv"])
X, Z = np.meshgrid(d["b_x"], d["b_z"], indexing="ij")
ax[1].pcolormesh(Z, X, d["b_I"], cmap=CMAP, norm=nrm, shading="auto", rasterized=True)
cx, cz = cap_curve(a2, s2, True); ax[1].plot(cz, cx, color=CAP, lw=LW_CAP)
normals(ax[1], a2, s2, True, np.linspace(-0.8 * a2, 0.8 * a2, 5), L_NRM)
Lf = 3 / (np.pi * na); ax[1].plot([zf, zf], [-Lf, Lf], color=OBS, lw=LW_OBS, ls="--"); ax[1].plot([zf], [0], "o", ms=3, color=OBS)
ax[1].text(zf - 1.6, Lf + 0.5, "focal plane", color=OBS, fontsize=FS_ANNOT)
ax[1].text(zv2 - 2.7, 1.12 * a2, "concave cap $\\Omega$", color="white", fontsize=FS_ANNOT)
ax[1].set_title(f"(b) focusing test (NA {na:g})", loc="left", fontsize=FS_TITLE)
for k in (0, 1):
    ax[k].set_xlabel("$z/\\lambda$"); ax[k].set_ylabel("$x/\\lambda$" if k == 0 else ""); ax[k].set_aspect("equal")
    for sp in ax[k].spines.values(): sp.set_linewidth(0.6)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=CMAP), ax=ax, shrink=0.75, pad=0.01)
cb.set_label("$|\\mathbf{E}|^2$ (normalised), $y=0$", fontsize=FS_CBAR)
fig.savefig("fig_setup_v2.png", dpi=220, bbox_inches="tight"); fig.savefig("fig_setup_v2.pdf", bbox_inches="tight"); print("saved fig_setup_v2")
