"""Merge 301x301 map + corner run + back-facing run into one one-sided map; recompute every Sec. 4.1 number."""
import sys, os, glob, json, zipfile
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
CACHE = os.path.join(REPO, 'data', 'map', 'map_cache.npz')
ZIP = os.path.join(REPO, 'data', 'map', 'results_301x301.zip')
RUNS = {"corner": (os.path.join(REPO, 'data', 'corner', 'unz', 'results'), os.path.join(REPO, 'map', 'corner', 'reference', 'map_corner_values.json')),
        "backface": (os.path.join(REPO, 'data', 'backface', 'unz', 'results'), os.path.join(REPO, 'map', 'backface', 'reference', 'map_cell_values.json'))}
OUT = sys.argv[1]
d = np.load(CACHE); A, S = d["A"], d["S"]
one, lp, fl = d["err_cbim_sc"].copy(), d["err_sclpwa_sc"].copy(), d["err_sc_exact"].copy()
one_i, lp_i = d["err_cbim_sc_int"].copy(), d["err_sclpwa_sc_int"].copy()
two, omp = d["err_cbim_sc"].copy(), d["err_cbimplus_sc"].copy()
unc = d["est_rel_uncertainty"].copy()
src = np.zeros(one.shape, int)            # 0 = map (no back-facing pairs), 1 = corner run, 2 = back-facing run
ex1, ex2 = np.full(one.shape, np.nan), np.full(one.shape, np.nan)
z = zipfile.ZipFile(ZIP)
for nm in z.namelist():
    if nm.endswith(".json") and "config" not in nm:
        r = json.loads(z.read(nm)); ex1[r["i"], r["j"]] = r["err_cbim_exact"]; ex2[r["i"], r["j"]] = r["err_sclpwa_exact"]
for k, (R, REF) in enumerate(RUNS.values(), 1):
    for f in glob.glob(os.path.join(R, "pt_*.json")):
        r = json.load(open(f)); i, j = r["i"], r["j"]
        one[i, j], lp[i, j], fl[i, j] = r["err_dbl_sc"], r["err_sclpwa_sc"], r["err_sc_exact"]
        one_i[i, j], lp_i[i, j] = r["err_dbl_sc_int"], r["err_sclpwa_sc_int"]
        ex1[i, j], ex2[i, j] = r["err_dbl_exact"], r["err_sclpwa_exact"]
        unc[i, j] = r["est_rel_uncertainty_dbl"]; src[i, j] = k
print("cells: map-only", (src == 0).sum(), "corner", (src == 1).sum(), "backface", (src == 2).sum())
vis = np.maximum(one, lp) > 10 * fl
print("visible", vis.sum(), "of", vis.size, "| finite", np.isfinite(one).all(), "| max unc on visible", unc[vis].max())
q = lp / one; Sg = np.broadcast_to(S[None, :], one.shape); Ag = np.broadcast_to(A[:, None], one.shape)
print(f"LPWA/one-sided over visible: min {q[vis].min():.2f} median {np.median(q[vis]):.0f} max {q[vis].max():.3g}")
for lo, hi in ((0.3, 9), (0.1, 0.3), (0, 0.1)):
    m = vis & (Sg >= lo) & (Sg < hi); print(f"  a/Rc in [{lo},{hi}): n {m.sum()} median {np.median(q[m]):.0f} range {q[m].min():.1f} .. {q[m].max():.3g}")
print(f"intensity one/LPWA worst ratio {np.max(one_i[vis] / lp_i[vis]):.3f}")
print(f"vs exact beam: one-sided worse in {(ex1[vis] >= ex2[vis]).sum()} of {vis.sum()}, worst ratio {np.nanmax(ex1[vis] / ex2[vis]):.3f}")
print(f"one-sided visible max {one[vis].max():.2e} at a={Ag[vis][one[vis].argmax()]:.2f} s={Sg[vis][one[vis].argmax()]:.3f}")
i60 = np.argmin(abs(A - 60)); j1 = 1; j62 = np.argmin(abs(S - 0.62))
print(f"a=60: s={S[j1]:.1e} -> {one[i60, j1]:.2e} (src {src[i60, j1]}); s={S[j62]:.3f} -> {one[i60, j62]:.2e} (src {src[i60, j62]}, map 2-sided {two[i60, j62]:.2e})")
lpv = lp[vis]; print(f"LPWA range on visible {lpv.min():.2e} .. {lpv.max():.2e}")
# slope: log one-sided vs log a/Rc, per row over curved visible cells with a/Rc<=0.1, median slope
sl = []
for i in range(len(A)):
    m = vis[i, 1:] & (S[1:] <= 0.1) & (one[i, 1:] > 10 * fl[i, 1:])
    if m.sum() > 10: sl.append(np.polyfit(np.log(S[1:][m]), np.log(one[i, 1:][m]), 1)[0])
print(f"slope per row (a/Rc<=0.1): median {np.median(sl):.3f} range {min(sl):.3f}..{max(sl):.3f} over {len(sl)} rows")
m = vis[:, 1:] & (one[:, 1:] > 10 * fl[:, 1:]); X = np.log(Sg[:, 1:][m]); Y = np.log(one[:, 1:][m] * Ag[:, 1:][m] ** 0)
print(f"global slope all curved visible: {np.polyfit(X, Y, 1)[0]:.3f}")
# two-sided vs one-sided where computed directly
c = (src > 0) & vis; r2 = two / one; ro = omp / one
for nm, mm in (("corner", (src == 1) & vis), ("backface", (src == 2) & vis), ("both", c)):
    print(f"{nm}: n {mm.sum()} two/one max {r2[mm].max():.3g} min {r2[mm].min():.4f} (<0.99: {(r2[mm] < 0.99).sum()}, min ratio -> advantage {100*(1-r2[mm].min()):.2f}%), >1.01: {(r2[mm] > 1.01).sum()}; Omega+/one max {ro[mm].max():.3g} min {ro[mm].min():.4f} (<0.99: {(ro[mm] < 0.99).sum()})")
cx = c & (two > lp); print(f"crossover (two-sided worse than LPWA): {cx.sum()} cells, all corner: {(src[cx] == 1).all()}, two/one median {np.median(r2[cx]):.0f} max {r2[cx].max():.0f}")
np.savez(OUT, A=A, S=S, one=one, lp=lp, fl=fl, one_int=one_i, lp_int=lp_i, two=two, omega_plus=omp, src=src, ratio=np.where(src > 0, r2, np.nan), vis=vis)
print("saved", OUT)
