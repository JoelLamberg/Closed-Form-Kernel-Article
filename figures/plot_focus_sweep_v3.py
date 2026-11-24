"""Focused-beam sweep figure: (a) error vs NA, (b) error vs curvature, (c) LPWA/Smythe factor for every visible point."""
import json, glob, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES, OUT = sys.argv[1], sys.argv[2]
rows = [json.load(open(f)) for f in glob.glob(RES + "/pt_*.json")]
ok = {(r["na"], r["a"], r["s"], r["concave"]): r for r in rows if r["status"] == "ok"}
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.labelcolor": INK, "figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.6})
fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
NAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def series(keyf, xs):
    pts = [(x, ok[keyf(x)]) for x in xs if keyf(x) in ok]
    return np.array([p[0] for p in pts]), [p[1] for p in pts]


# (a) error vs NA, a = 20
cases = [("plane", BLUE, lambda na: (na, 20.0, 0.0, False)), ("convex, $a/R_c$ = 0.3", ORANGE, lambda na: (na, 20.0, 0.3, False)),
         ("concave, $a/R_c$ = 0.7", AQUA, lambda na: (na, 20.0, 0.7, True))]
fl = []
for lab, col, kf in cases:
    x, rs = series(kf, NAS)
    ax[0].plot(x, [r["err_dbl_sc"] for r in rs], "-o", color=col, lw=2, ms=6, label=f"{lab}, Smythe")
    ax[0].plot(x, [r["err_lpwa_sc"] for r in rs], "--s", color=col, lw=2, ms=6, mfc="white", label=f"{lab}, LPWA H")
    fl += [(xx, r["err_sc_exact"]) for xx, r in zip(x, rs)]
fx = sorted({f[0] for f in fl}); fhi = [10 * max(v for xx, v in fl if xx == q) for q in fx]
ax[0].fill_between(fx, 1e-7, fhi, color=GREY, alpha=0.45, lw=0, zorder=0)
ax[0].set_yscale("log"); ax[0].set_ylim(1e-6, 1e-1); ax[0].set_xlabel("numerical aperture NA")
ax[0].set_ylabel("relative error vs Stratton–Chu, exact H"); ax[0].set_title("(a) error vs NA ($a = 20\\lambda$; convex $a/R_c$ = 0.3, concave 0.7)", loc="left", fontsize=9)

# (b) error vs curvature, NA 0.5, a = 20
SS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
for lab, col, cc in (("convex", ORANGE, False), ("concave", AQUA, True)):
    x, rs = series(lambda s: (0.5, 20.0, s, cc if s > 0 else False), SS)
    ax[1].plot(x, [r["err_dbl_sc"] for r in rs], "-o", color=col, lw=2, ms=6, label=f"{lab}, Smythe")
    ax[1].plot(x, [r["err_lpwa_sc"] for r in rs], "--s", color=col, lw=2, ms=6, mfc="white", label=f"{lab}, LPWA H")
x, rs = series(lambda s: (0.5, 20.0, s, True if s > 0 else False), SS)
ax[1].fill_between(x, 1e-7, [10 * r["err_sc_exact"] for r in rs], color=GREY, alpha=0.45, lw=0, zorder=0)
ax[1].annotate("concave cap follows the\nconverging wavefront\n($R_c \\approx$ focal distance)", xy=(0.9, 4e-6), xytext=(0.47, 1.4e-6),
               fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
ax[1].set_yscale("log"); ax[1].set_ylim(1e-6, 1e-1); ax[1].set_xlabel("curvature $a/R_c$")
ax[1].set_title("(b) focal-plane error vs curvature (NA 0.5, $a = 20\\lambda$)", loc="left", fontsize=9)

# (c) factor LPWA / Smythe, every visible point
for lab, col, sel in (("plane", BLUE, lambda r: r["s"] == 0), ("convex", ORANGE, lambda r: r["s"] > 0 and not r["concave"]),
                      ("concave", AQUA, lambda r: r["concave"])):
    pts = [r for r in ok.values() if sel(r) and max(r["err_dbl_sc"], r["err_lpwa_sc"]) > 10 * r["err_sc_exact"]]
    off = {"plane": -0.018, "convex": 0.0, "concave": 0.018}[lab]
    ax[2].scatter([r["na"] + off for r in pts], [r["err_lpwa_sc"] / r["err_dbl_sc"] for r in pts], s=26, color=col,
                  edgecolor="white", linewidth=0.8, label=lab, zorder=3)
ax[2].axhline(1, color=INK2, lw=1); ax[2].text(0.08, 1.15, "equal accuracy", fontsize=7.5, color=INK2)
allf = [r["err_lpwa_sc"] / r["err_dbl_sc"] for r in ok.values() if max(r["err_dbl_sc"], r["err_lpwa_sc"]) > 10 * r["err_sc_exact"]]
ax[2].axhline(min(allf), color=INK2, lw=0.8, ls=":"); ax[2].text(0.08, min(allf) * 0.72, f"smallest factor {min(allf):.0f}", fontsize=7.5, color=INK2)
ax[2].set_yscale("log"); ax[2].set_ylim(0.5, 3e3); ax[2].set_xlabel("numerical aperture NA")
ax[2].set_ylabel("error ratio  (SC, LPWA H) ÷ (one-sided CBIM)"); ax[2].legend(fontsize=7.5, frameon=False, loc="upper right")
ax[2].set_title(f"(c) error ratio, one dot per geometry ({len(allf)} of {len(ok)})", loc="left", fontsize=9)
ax[2].annotate("", xy=(0.735, 12), xytext=(0.735, 1.6), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
ax[2].text(0.72, 4.4, "one-sided CBIM\nmore accurate", fontsize=7.5, color=INK2, ha="right", va="center")
for a_ in ax[:2]:
    a_.set_xticks(NAS if a_ is ax[0] else SS)
ax[2].set_xticks(NAS)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
METHOD_GREY = "#7a7975"                               # neutral: the method key shows line style only, not a curve
H1 = [Line2D([], [], color=BLUE, lw=2.5, label="plane"), Line2D([], [], color=ORANGE, lw=2.5, label="convex cap"),
      Line2D([], [], color=AQUA, lw=2.5, label="concave cap")]
H2 = [Line2D([], [], color=METHOD_GREY, lw=2, marker="o", ms=6, label="one-sided CBIM (Smythe), E only"),
      Line2D([], [], color=METHOD_GREY, lw=2, ls="--", marker="s", ms=6, mfc="white", label="Stratton–Chu, LPWA-estimated H"),
      Patch(color=GREY, alpha=0.45, label="below 10× the accuracy of the reference")]
L1 = fig.legend(handles=H1, title="surface (colour)", loc="upper right", bbox_to_anchor=(0.36, -0.01), ncol=3,
                frameon=False, fontsize=8, title_fontsize=8, alignment="left")
L2 = fig.legend(handles=H2, title="method (line and marker style)", loc="upper left", bbox_to_anchor=(0.39, -0.01), ncol=3,
                frameon=False, fontsize=8, title_fontsize=8, alignment="left")
ax[2].get_legend().remove()
fig.savefig(OUT + ".png", dpi=220, bbox_inches="tight"); fig.savefig(OUT + ".pdf", bbox_inches="tight"); print("saved", OUT, "min factor", min(allf))
