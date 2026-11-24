"""Focused-beam sweep with the stronger magnetic-field estimate. focus_core.py is a byte-identical copy of the focus
sweep's module; this file adds Stratton-Chu with H from the second-order local angular-spectrum estimate with curvature
term (LAS2c, las_core.py) next to SC exact, one-sided (dbl), SC+LPWA (analytic phase gradient) and the two-sided form."""
import numpy as np
from focus_core import (K, Z_RIM, cap_oriented, beam_for, rim_ratio, lpwa_H, focal_plane, _richardson, choose_focus,
                        frac_positive, rel, rel_int, LADDER)
from las_core import las_H


def radiate_focus_las(Ro, P, N, w, nxE, nxH, nxH2, nxH3, hTx, hTy, kappa, chunk_elems=1_000_000):
    """E_SC, E_dbl, E_SC+LPWA, E_two (sub-cell sigma), E_SC+LAS2c at Ro (m,3)."""
    m, Ns = Ro.shape[0], P.shape[0]
    Esc, Edb, Es2, Etw, Es3 = (np.zeros((m, 3), np.complex128) for _ in range(5))
    step = max(1, chunk_elems // Ns)
    wE = w[:, None] * nxE; wJs = [w[:, None] * x for x in (nxH, nxH2, nxH3)]
    curved = kappa != 0.0
    NhTx, NhTy = np.einsum("jk,jk->j", N, hTx)[None], np.einsum("jk,jk->j", N, hTy)[None]
    for i0 in range(0, m, step):
        r = Ro[i0:i0 + step]
        d = r[:, None, :] - P[None, :, :]
        t = np.sqrt(np.einsum("ijk,ijk->ij", d, d)); kt = K * t; e = np.exp(1j * kt)
        EM = ((1 - 1j * kt) * e / (4 * np.pi * t**3))[..., None] * np.cross(wE[None, :, :], d)
        g = e / (4 * np.pi * t); A = 1 + 1j / kt - 1 / kt**2; Bc = -1 - 3j / kt + 3 / kt**2
        EJ = []
        for wJ in wJs:
            dJ = np.einsum("ijk,jk->ij", d, wJ) / t
            EJ.append(((1j * K * g)[..., None] * (A[..., None] * wJ[None] + (Bc * dJ / t)[..., None] * d)).sum(axis=1))
        f0 = np.einsum("ijk,jk->ij", d, N)
        if curved:
            ga = np.abs(kappa * np.einsum("ijk,jk->ij", d, hTx) - NhTx)
            gb = np.abs(kappa * np.einsum("ijk,jk->ij", d, hTy) - NhTy)
            fp = frac_positive(f0, ga, gb)
        else:
            fp = (f0 > 0).astype(float)
        sM = EM.sum(axis=1)
        Esc[i0:i0 + step] = sM + EJ[0]; Es2[i0:i0 + step] = sM + EJ[1]; Es3[i0:i0 + step] = sM + EJ[2]
        Edb[i0:i0 + step] = 2 * sM
        Etw[i0:i0 + step] = (2 * (2 * fp - 1)[..., None] * EM).sum(axis=1)
    return Esc, Edb, Es2, Etw, Es3


def cap_center(a, s, concave):
    if s == 0:
        return None, None
    Rc = a / s; c = np.sqrt(Rc**2 - a**2)
    return np.array([0.0, 0.0, Z_RIM + c if concave else Z_RIM - c]), Rc


def run_point_focus_las(a, s, na, concave, nobs=49, half_width=3.0, tol=1e-4, floor=1e-10, ladder=LADDER, nobs_mid=25,
                        rim_target=1e-4, zf=None):
    if zf is None:
        zf, status = choose_focus(a, s, concave, na, rim_target)
    else:
        status = "ok"
    base = dict(a=a, s=s, na=na, concave=bool(concave), status=status, rim_target=rim_target)
    if status != "ok":
        return base
    beam = beam_for(a, s, concave, na, zf)
    w0 = 1.0 / (np.pi * na)
    center, Rc = cap_center(a, s, concave)
    keys = ("Edb", "Es2", "Etw", "Es3")
    hist = []
    for ppw in ladder:
        P, N, w, zv, h, hTx, hTy, kappa = cap_oriented(a, s, ppw, concave)
        Ro, wo = focal_plane(zf, w0, nobs, half_width)
        E, H, G = beam.fields(P, want_grad=True)
        H2 = lpwa_H(P, N, E, G)
        H3 = las_H(P, N, E, lambda Q: beam.fields(Q)[0], center, Rc)
        nr_s = lambda X: np.sqrt(np.sum(w[:, None] * np.abs(np.cross(N, X))**2))
        hErr2, hErr3 = nr_s(H2 - H) / nr_s(H), nr_s(H3 - H) / nr_s(H)
        Esc, Edb, Es2, Etw, Es3 = radiate_focus_las(Ro, P, N, w, np.cross(N, E), np.cross(N, H), np.cross(N, H2),
                                                    np.cross(N, H3), hTx, hTy, kappa)
        hist.append(dict(ppw=ppw, Nsrc=int(P.shape[0]), Esc=Esc, Edb=Edb, Es2=Es2, Etw=Etw, Es3=Es3))
        if len(hist) >= 2:
            p0, p1 = hist[-2], hist[-1]
            nrm = lambda X: np.sqrt(np.sum(wo[:, None] * np.abs(X)**2)); nE = nrm(p1["Esc"])
            refs = {k: max(nrm(p1[k] - p1["Esc"]), floor * nE) for k in keys}
            ch = [nrm(p1["Esc"] - p0["Esc"]) / min(refs.values())] + [nrm(p1[k] - p0[k]) / refs[k] for k in refs]
            if max(ch) < tol:
                break
    last = hist[-1]; Esc = last["Esc"]; Eex, _, _ = beam.fields(Ro)
    res = dict(base, zf=zf, w0=w0, nobs=nobs, half_width=half_width, tol=tol, floor=floor,
               rim_ratio=rim_ratio(a, s, concave, na, zf), ppw_used=last["ppw"], Nsrc=last["Nsrc"], levels=len(hist),
               converged=bool(len(hist) < len(ladder)), surfH_err_lpwa=hErr2, surfH_err_las=hErr3,
               err_sc_exact=rel(Esc, Eex, wo))
    for key, name in (("Edb", "dbl"), ("Es2", "lpwa"), ("Etw", "two"), ("Es3", "las")):
        res[f"err_{name}_sc"] = rel(last[key], Esc, wo); res[f"err_{name}_sc_int"] = rel_int(last[key], Esc, wo)
        res[f"err_{name}_exact"] = rel(last[key], Eex, wo)
        if len(hist) >= 2:
            Ex, p = _richardson(hist, key, Esc, wo); ex = rel(Ex, Esc, wo)
            res[f"err_{name}_sc_extrap"] = ex; res[f"richardson_order_{name}"] = p
            res[f"unc_{name}"] = abs(res[f"err_{name}_sc"] - ex) / max(ex, 1e-300)
    return res
