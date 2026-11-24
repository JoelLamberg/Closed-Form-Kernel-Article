"""Map run with a stronger magnetic-field estimate. Same beam, cap, observation plane, adaptive ladder and Richardson
as the 301 x 301 map / corner / back-facing runs (cbim_sc_core.py and corner_core.py are byte-identical copies). Adds
Stratton-Chu with H from the second-order local angular-spectrum estimate with curvature term (LAS2c, las_core.py).
Methods per point: SC exact H (reference), one-sided CBIM = Smythe (dbl), SC + LPWA H, SC + LAS2c H."""
import numpy as np
from cbim_sc_core import K, Z_RIM, cap, csp_fields, obs_plane, lpwa_H, rel, rel_int, rel_max, PPW_LADDER
from corner_core import _richardson
from las_core import las_H


def radiate_las(Ro, P, N, w, nxE, nxH, nxH2, nxH3, chunk_elems=1_000_000):
    """E_SC (exact H), E_dbl (sigma = +1), E_SC+LPWA, E_SC+LAS2c at Ro (m,3)."""
    m, Ns = Ro.shape[0], P.shape[0]
    Esc = np.zeros((m, 3), np.complex128); Edb = np.zeros_like(Esc); Es2 = np.zeros_like(Esc); Es3 = np.zeros_like(Esc)
    step = max(1, chunk_elems // Ns)
    wE = w[:, None] * nxE; wJs = [w[:, None] * x for x in (nxH, nxH2, nxH3)]
    for i0 in range(0, m, step):
        r = Ro[i0:i0 + step]
        d = r[:, None, :] - P[None, :, :]
        t = np.sqrt(np.einsum("ijk,ijk->ij", d, d)); kt = K * t; e = np.exp(1j * kt)
        sM = (((1 - 1j * kt) * e / (4 * np.pi * t**3))[..., None] * np.cross(wE[None, :, :], d)).sum(axis=1)
        g = e / (4 * np.pi * t); A = 1 + 1j / kt - 1 / kt**2; Bc = -1 - 3j / kt + 3 / kt**2
        out = []
        for wJ in wJs:
            dJ = np.einsum("ijk,jk->ij", d, wJ) / t
            out.append(((1j * K * g)[..., None] * (A[..., None] * wJ[None] + (Bc * dJ / t)[..., None] * d)).sum(axis=1))
        Esc[i0:i0 + step] = sM + out[0]; Es2[i0:i0 + step] = sM + out[1]; Es3[i0:i0 + step] = sM + out[2]
        Edb[i0:i0 + step] = 2 * sM
    return Esc, Edb, Es2, Es3


def surface_H_estimates(P, N, E, b, a, s):
    H2 = lpwa_H(P, N, E, b)
    center = None if s == 0 else np.array([0.0, 0.0, Z_RIM - np.sqrt((a / s)**2 - a**2)])
    H3 = las_H(P, N, E, lambda Q: csp_fields(Q, b)[0], center, None if s == 0 else a / s)
    return H2, H3


def run_point_las(a, s, nobs=49, tol=1e-3, floor=1e-8, w0_over_a=1/5.5, ladder=PPW_LADDER):
    w0 = w0_over_a * a; b = K * w0**2 / 2
    hist = []
    for ppw in ladder:
        P, N, w, zv, h, hTx, hTy, kappa = cap(a, s, ppw)
        E, H = csp_fields(P, b)
        Ro, wo = obs_plane(zv, a, w0, nobs)
        H2, H3 = surface_H_estimates(P, N, E, b, a, s)
        nr_s = lambda X: np.sqrt(np.sum(w[:, None] * np.abs(np.cross(N, X))**2))
        hErr2, hErr3 = nr_s(H2 - H) / nr_s(H), nr_s(H3 - H) / nr_s(H)
        Esc, Edb, Es2, Es3 = radiate_las(Ro, P, N, w, np.cross(N, E), np.cross(N, H), np.cross(N, H2), np.cross(N, H3))
        hist.append(dict(ppw=ppw, Nsrc=int(P.shape[0]), Esc=Esc, Edb=Edb, Es2=Es2, Es3=Es3))
        if len(hist) >= 2:
            p0, p1 = hist[-2], hist[-1]
            nrm = lambda X: np.sqrt(np.sum(wo[:, None] * np.abs(X)**2)); nE = nrm(p1["Esc"])
            refs = {k: max(nrm(p1[k] - p1["Esc"]), floor * nE) for k in ("Edb", "Es2", "Es3")}
            ch = [nrm(p1["Esc"] - p0["Esc"]) / min(refs.values())] + [nrm(p1[k] - p0[k]) / refs[k] for k in refs]
            if max(ch) < tol:
                break
    Eex, _ = csp_fields(Ro, b)
    last = hist[-1]; Esc = last["Esc"]
    out = dict(a=a, s=s, nobs=nobs, w0=w0, ppw_used=last["ppw"], Nsrc=last["Nsrc"], levels=len(hist),
               surfH_err_lpwa=hErr2, surfH_err_las=hErr3, err_sc_exact=rel(Esc, Eex, wo))
    for key, name in (("Edb", "dbl"), ("Es2", "sclpwa"), ("Es3", "sclas")):
        out[f"err_{name}_sc"] = rel(last[key], Esc, wo); out[f"err_{name}_sc_int"] = rel_int(last[key], Esc, wo)
        out[f"err_{name}_exact"] = rel(last[key], Eex, wo)
        if len(hist) >= 2:
            Ex, p = _richardson(hist, key, Esc, wo); ex = rel(Ex, Esc, wo)
            out[f"err_{name}_sc_extrap"] = ex; out[f"richardson_order_{name}"] = p
            out[f"unc_{name}"] = abs(out[f"err_{name}_sc"] - ex) / max(ex, 1e-300)
    if len(hist) >= 2:
        out["last_step_change_sc"] = rel(hist[-1]["Esc"], hist[-2]["Esc"], wo)
    return out
