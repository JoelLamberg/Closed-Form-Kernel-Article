"""Cost of the E-only methods on the SAME source-observation pairs (extends the handoff's bench_cost.py, whose
CBIM-closed and Stratton-Chu kernels are reused verbatim in style):
  cbim_sign     CBIM closed form, sigma = sign(d.n)                      (handoff bench_cost.py 'cbim_closed')
  cbim_subcell  CBIM closed form, exact sub-cell sigma fraction          (what the map run used)
  sc_exactH     Stratton-Chu with exact H                                (handoff 'stratton_chu'; reference)
  sc_lpwaH      Stratton-Chu with LPWA-estimated H = same kernel + one-off O(Nsrc) H estimate from E
Single core numpy (OMP=1) and, if CuPy is present, the same kernels on the GPU. Writes bench_lpwa.json.
usage: python bench_lpwa.py [a] [s] [ppw] [nobs]        (default = handoff probe 14 0.5 5 48)"""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # cbim_sc_core.py sits next to this file
from cbim_sc_core import K, cap, csp_fields, obs_plane, lpwa_H

av = [float(x) for x in sys.argv[1:]]
A_, S_, PPW, NOBS = (av + [14.0, 0.5, 5, 48][len(av):])[:4]
PPW, NOBS = int(PPW), int(NOBS)
w0 = A_ / 5.5; b = K * w0**2 / 2
P, N, W, zv, h, hTx, hTy, kappa = cap(A_, S_, PPW)
E, H = csp_fields(P, b)
Ro, wo = obs_plane(zv, A_, w0, NOBS)
Eex = csp_fields(Ro, b)[0]


def kernels(xp):
    def frac_positive(f0, ga, gb):
        a = xp.maximum(xp.maximum(ga, gb), 1e-300); bb = xp.maximum(xp.minimum(ga, gb), 1e-300)
        v = -f0; lo, mid = -(a + bb) / 2, -(a - bb) / 2
        cdf = xp.where(v <= lo, 0.0, xp.where(v <= mid, (v - lo) ** 2 / (2 * a * bb),
              xp.where(v <= -mid, (v + a / 2) / a, xp.where(v < -lo, 1 - (-lo - v) ** 2 / (2 * a * bb), 1.0))))
        return 1.0 - cdf

    def geom(r, P):
        d = r[:, None, :] - P[None]
        return d, xp.sqrt(xp.einsum("ijk,ijk->ij", d, d))

    def cbim(r, g, subcell):
        P, N, wE = g["P"], g["N"], g["wE"]
        d, t = geom(r, P)
        kt = K * t
        I = (1 - 1j * kt) * xp.exp(1j * kt) / t**2
        f0 = xp.einsum("ijk,jk->ij", d, N)
        if subcell:
            ga = xp.abs(kappa * xp.einsum("ijk,jk->ij", d, g["hTx"]) - g["NhTx"])
            gb = xp.abs(kappa * xp.einsum("ijk,jk->ij", d, g["hTy"]) - g["NhTy"])
            sig = 2 * frac_positive(f0, ga, gb) - 1
        else:
            sig = xp.sign(f0)
        c = sig * I / (2 * np.pi * t)
        return xp.einsum("ij,ijk->ik", c, xp.cross(wE[None], d))

    def sc(r, g, wJ):
        P, wE = g["P"], g["wE"]
        d, t = geom(r, P)
        kt = K * t; e = xp.exp(1j * kt)
        EM = xp.einsum("ij,ijk->ik", (1 - 1j * kt) * e / (4 * np.pi * t**3), xp.cross(wE[None], d))
        gg = e / (4 * np.pi * t)
        A2 = 1 + 1j / kt - 1 / kt**2; B2 = -1 - 3j / kt + 3 / kt**2
        dJ = xp.einsum("ijk,jk->ij", d, wJ) / t
        EJ = 1j * K * (xp.einsum("ij,jk->ik", gg * A2, wJ) + xp.einsum("ij,ijk->ik", gg * B2 * dJ / t, d))
        return EM + EJ
    return cbim, sc


def run(xp, name, chunk, sync=lambda: None):
    asx = (lambda x: xp.asarray(x))
    cbim, sc = kernels(xp)
    t0 = time.perf_counter()
    H2 = lpwa_H(P, N, E, b)                                   # one-off O(Nsrc) step: H estimated from E
    t_lpwa = time.perf_counter() - t0
    g = dict(P=asx(P), N=asx(N), wE=asx(W[:, None] * np.cross(N, E)), hTx=asx(hTx), hTy=asx(hTy))
    g["NhTx"] = xp.einsum("jk,jk->j", g["N"], g["hTx"])[None]; g["NhTy"] = xp.einsum("jk,jk->j", g["N"], g["hTy"])[None]
    wJ = asx(W[:, None] * np.cross(N, H)); wJ2 = asx(W[:, None] * np.cross(N, H2))
    R = asx(Ro); m, Ns = Ro.shape[0], P.shape[0]; rows = max(1, chunk // Ns)
    fns = {"cbim_sign": lambda r: cbim(r, g, False), "cbim_subcell": lambda r: cbim(r, g, True),
           "sc_exactH": lambda r: sc(r, g, wJ), "sc_lpwaH": lambda r: sc(r, g, wJ2)}
    out, fields = {}, {}
    for k, fn in fns.items():
        best = 1e99
        for rep in range(3):
            sync(); t0 = time.perf_counter()
            F = xp.concatenate([fn(R[i:i + rows]) for i in range(0, m, rows)])
            sync(); best = min(best, time.perf_counter() - t0)
        out[k] = best / (m * Ns) * 1e9
        fields[k] = F.get() if hasattr(F, "get") else F
    out["lpwa_setup_s"] = t_lpwa
    return out, fields


def rel(a, ref):
    return float(np.sqrt(np.sum(wo[:, None] * np.abs(a - ref) ** 2) / np.sum(wo[:, None] * np.abs(ref) ** 2)))


if __name__ == "__main__":
    Ns, m = P.shape[0], Ro.shape[0]
    print(f"probe a={A_} s={S_} ppw={PPW} nobs={NOBS}: Nsrc={Ns}, Nobs={m}, pairs={Ns * m:.3e}", flush=True)
    res = dict(probe=dict(a=A_, s=S_, ppw=PPW, nobs=NOBS, Nsrc=Ns, pairs=Ns * m))
    cpu, F = run(np, "cpu", 1_000_000)
    res["cpu_1core_ns_per_pair"] = cpu
    res["error_vs_SC_exactH"] = {k: rel(F[k], F["sc_exactH"]) for k in ("cbim_sign", "cbim_subcell", "sc_lpwaH")}
    res["SC_exactH_vs_exact_field"] = rel(F["sc_exactH"], Eex)
    try:
        import cupy as cp
        gpu, G = run(cp, "gpu", 20_000_000, sync=cp.cuda.Stream.null.synchronize)
        res["gpu_ns_per_pair"] = gpu
        res["gpu_vs_cpu_maxrel"] = {k: float(np.abs(G[k] - F[k]).max() / np.abs(F[k]).max()) for k in F}
    except ImportError:
        pass
    print(json.dumps(res, indent=1))
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_bench"), exist_ok=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_bench",
                                     f"bench_lpwa_a{A_:g}_s{S_:g}.json"), "w"), indent=1)
