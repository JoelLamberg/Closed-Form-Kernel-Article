"""Focal-plane image figure: per point (row) the reference intensity, then |E_method - E_SC| / max|E_SC| for Smythe
(one-sided CBIM) and SC with LPWA H, on one log scale per row. Prints convergence (ppw 20 vs 28) of the difference fields.
usage: python plot_focal_fields.py FIELDS_DIR OUT_STEM [--comp all|z]"""
import sys, glob, os, argparse
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
p = argparse.ArgumentParser(); p.add_argument("dir"); p.add_argument("out"); p.add_argument("--comp", default="all"); a = p.parse_args()
files = sorted(glob.glob(os.path.join(a.dir, "fields_*.npz")), key=lambda f: ("cc" in f, "s0.0" not in f))
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.labelcolor": INK, "figure.facecolor": "white"})
inf = plt.get_cmap("inferno")
fig, ax = plt.subplots(len(files), 3, figsize=(10.2, 3.15 * len(files)), constrained_layout=True, squeeze=False)
ci = {"x": 0, "y": 1, "z": 2}.get(a.comp)
sel = (lambda E: E) if ci is None else (lambda E: E[..., ci:ci + 1])
for r, f in enumerate(files):
    d = np.load(f); X, Y = d["X"], d["Y"]; ppws = sorted(int(k.split("ppw")[1]) for k in d.files if k.startswith("E_sc_ppw"))
    hi = ppws[-1]; Esc = d[f"E_sc_ppw{hi}"]; nrm = np.max(np.linalg.norm(Esc, axis=-1))
    I = np.sum(np.abs(sel(Esc))**2, -1) / nrm**2
    errs = {m: np.linalg.norm(sel(d[f"E_{m}_ppw{hi}"] - Esc), axis=-1) / nrm for m in ("dbl", "lpwa")}
    l2 = {m: np.sqrt(np.sum(np.abs(sel(d[f"E_{m}_ppw{hi}"] - Esc))**2) / np.sum(np.abs(sel(Esc))**2)) for m in ("dbl", "lpwa")}
    for m in ("dbl", "lpwa"):                                  # convergence of the difference field between ppw levels
        if len(ppws) > 1:
            D1 = d[f"E_{m}_ppw{ppws[-2]}"] - d[f"E_sc_ppw{ppws[-2]}"]; D2 = d[f"E_{m}_ppw{hi}"] - Esc
            print(f"{os.path.basename(f)} {m}: |diff(ppw{ppws[-2]}) - diff(ppw{hi})| / |diff| = {np.linalg.norm(D1-D2)/np.linalg.norm(D2):.1e}")
    sc_ex = np.sqrt(np.sum(np.abs(Esc - d["E_exact"])**2) / np.sum(np.abs(d["E_exact"])**2))
    print(f"{os.path.basename(f)}: L2 dbl {l2['dbl']:.2e}  lpwa {l2['lpwa']:.2e}  SC vs exact {sc_ex:.1e}")
    vmax = max(e.max() for e in errs.values()); vmin = vmax * 1e-4
    x0, y0 = X / 1.0, Y / 1.0; ext = [x0.min(), x0.max(), y0.min(), y0.max()]
    cc = "concave" if bool(d["concave"]) else ("plane" if float(d["s"]) == 0 else "convex")
    lab = f"NA {float(d['na']):.1f}, $a$ = {float(d['a']):.0f}λ, $a/R_c$ = {float(d['s']):.1f} ({cc})"
    im0 = ax[r, 0].imshow(I.T, origin="lower", extent=ext, cmap=inf, norm=Normalize(0, I.max()), interpolation="nearest")
    ax[r, 0].set_title(("intensity" if ci is None else f"$|E_{a.comp}|^2$") + f", Stratton–Chu (exact H)\n{lab}", loc="left", fontsize=8.5)
    for k, (m, name) in enumerate((("dbl", "Smythe (one-sided CBIM), E only"), ("lpwa", "Stratton–Chu, LPWA H, E only")), 1):
        imk = ax[r, k].imshow(np.clip(errs[m], vmin, None).T, origin="lower", extent=ext, cmap=inf, norm=LogNorm(vmin, vmax), interpolation="nearest")
        dl = "|\\Delta E|" if ci is None else f"|\\Delta E_{a.comp}|"
        rl = "relative L2" if ci is None else f"L2 relative to $E_{a.comp}$"
        ax[r, k].set_title(f"{name}\n${dl}$, {rl} = {l2[m]:.1e}", loc="left", fontsize=8.5)
    fig.colorbar(im0, ax=ax[r, 0], shrink=0.85, pad=0.02, label="normalised")
    fig.colorbar(imk, ax=ax[r, 1:], shrink=0.85, pad=0.02, label=("$|E - E_{SC}|$" if ci is None else f"$|E_{a.comp} - E_{{SC,{a.comp}}}|$") + " / max $|E_{SC}|$")
    for k in range(3):
        ax[r, k].set_xlabel("$x/\\lambda$"); ax[r, k].set_ylabel("$y/\\lambda$" if k == 0 else "")
        for sp in ax[r, k].spines.values(): sp.set_linewidth(0.6)
fig.savefig(a.out + ".png", dpi=220, bbox_inches="tight"); fig.savefig(a.out + ".pdf", bbox_inches="tight"); print("saved", a.out)
