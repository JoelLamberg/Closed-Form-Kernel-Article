"""GPU feasibility: cbim_sc_core.radiate() ported line-for-line to CuPy (complex128), timed and compared with the
CPU result on the same point. usage: python gpu_test.py [a] [s] [ppw] [nobs] [chunk_elems]"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # cbim_sc_core.py sits next to this file
import numpy as np
import cupy as cp
from cbim_sc_core import K, cap, csp_fields, obs_plane, lpwa_H, radiate


def frac_positive_x(xp, f0, ga, gb):
    a = xp.maximum(xp.maximum(ga, gb), 1e-300); b = xp.maximum(xp.minimum(ga, gb), 1e-300)
    v = -f0
    lo, mid = -(a + b) / 2, -(a - b) / 2
    cdf = xp.where(v <= lo, 0.0,
          xp.where(v <= mid, (v - lo) ** 2 / (2 * a * b),
          xp.where(v <= -mid, (v + a / 2) / a,
          xp.where(v < -lo, 1 - (-lo - v) ** 2 / (2 * a * b), 1.0))))
    return 1.0 - cdf


def radiate_gpu(Ro, P, N, w, nxE, nxH, hTx, hTy, kappa, nxH2, chunk_elems):
    xp = cp
    Ro, P, N, w, nxE, nxH, hTx, hTy, nxH2 = (cp.asarray(x) for x in (Ro, P, N, w, nxE, nxH, hTx, hTy, nxH2))
    m, Ns = Ro.shape[0], P.shape[0]
    Esc = xp.zeros((m, 3), xp.complex128); Ecb = xp.zeros_like(Esc); Ecp = xp.zeros_like(Esc); Es2 = xp.zeros_like(Esc)
    step = max(1, chunk_elems // Ns)
    wJ2, wJ, wE = w[:, None] * nxH2, w[:, None] * nxH, w[:, None] * nxE
    NhTx, NhTy = xp.einsum("jk,jk->j", N, hTx)[None], xp.einsum("jk,jk->j", N, hTy)[None]
    for i0 in range(0, m, step):
        r = Ro[i0:i0 + step]
        d = r[:, None, :] - P[None, :, :]
        t = xp.sqrt(xp.einsum("ijk,ijk->ij", d, d))
        kt = K * t
        e = xp.exp(1j * kt)
        cM = (1 - 1j * kt) * e / (4 * np.pi * t**3)
        EM = cM[..., None] * xp.cross(wE[None, :, :], d)
        g = e / (4 * np.pi * t)
        A = 1 + 1j / kt - 1 / kt**2
        Bc = -1 - 3j / kt + 3 / kt**2
        dJ = xp.einsum("ijk,jk->ij", d, wJ) / t
        EJ = (1j * K * g)[..., None] * (A[..., None] * wJ[None] + (Bc * dJ / t)[..., None] * d)
        f0 = xp.einsum("ijk,jk->ij", d, N)
        if kappa == 0.0:
            fp = (f0 > 0).astype(float)
        else:
            ga = xp.abs(kappa * xp.einsum("ijk,jk->ij", d, hTx) - NhTx)
            gb = xp.abs(kappa * xp.einsum("ijk,jk->ij", d, hTy) - NhTy)
            fp = frac_positive_x(xp, f0, ga, gb)
        Esc[i0:i0 + step] = (EM + EJ).sum(axis=1)
        dJ2 = xp.einsum("ijk,jk->ij", d, wJ2) / t
        EJ2 = (1j * K * g)[..., None] * (A[..., None] * wJ2[None] + (Bc * dJ2 / t)[..., None] * d)
        Es2[i0:i0 + step] = (EM + EJ2).sum(axis=1)
        Ecb[i0:i0 + step] = (2 * (2 * fp - 1)[..., None] * EM).sum(axis=1)
        Ecp[i0:i0 + step] = (2 * fp[..., None] * EM).sum(axis=1)
    cp.cuda.Stream.null.synchronize()
    return [cp.asnumpy(x) for x in (Esc, Ecb, Ecp, Es2)]


if __name__ == "__main__":
    av = [float(x) for x in sys.argv[1:]]
    a, s, ppw, nobs = (av + [10.0, 0.5, 7, 25][len(av):])[:4]
    ppw, nobs = int(ppw), int(nobs)
    chunk = int(av[4]) if len(av) > 4 else 10_000_000
    w0 = a / 5.5; b = K * w0**2 / 2
    P, N, w, zv, h, hTx, hTy, kappa = cap(a, s, ppw)
    E, H = csp_fields(P, b); H2 = lpwa_H(P, N, E, b)
    Ro, wo = obs_plane(zv, a, w0, nobs)
    args = (Ro, P, N, w, np.cross(N, E), np.cross(N, H), hTx, hTy, kappa, np.cross(N, H2))
    print(f"a={a} s={s} ppw={ppw} nobs={nobs}: Nsrc={P.shape[0]} pairs={P.shape[0] * Ro.shape[0]:.3e} chunk={chunk:.0e}", flush=True)
    radiate_gpu(*args, chunk)                                      # warm-up (kernel compile)
    t0 = time.perf_counter(); G = radiate_gpu(*args, chunk); tg = time.perf_counter() - t0
    print(f"GPU: {tg:.2f} s  ({tg / (P.shape[0] * Ro.shape[0]) * 1e9:.0f} ns/pair, all three methods)", flush=True)
    t0 = time.perf_counter(); C = radiate(*args[:8], nxH2=args[9]); tc = time.perf_counter() - t0
    print(f"CPU (1 core, box loaded): {tc:.2f} s  ({tc / (P.shape[0] * Ro.shape[0]) * 1e9:.0f} ns/pair)", flush=True)
    for name, g, c in zip(("SC", "CBIM", "CBIM+", "SC+LPWA"), G, C):
        print(f"  {name:8s} max rel diff GPU vs CPU {np.abs(g - c).max() / np.abs(c).max():.1e}")
    print(f"speed-up vs one loaded core: {tc / tg:.0f}x  -> equivalent to ~{tc / tg:.0f} of the 92 workers", flush=True)
