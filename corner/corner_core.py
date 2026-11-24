"""Corner run: the doubled magnetic-current formula (CBIM with sigma = +1, i.e. Stratton-Chu with the n x H term
dropped and the n x E term doubled = Smythe) against Stratton-Chu with exact H, plus Stratton-Chu with LPWA-estimated H,
on the same adaptive ladder as the 301 x 301 map. cbim_sc_core.py is the map's module, UNCHANGED; this file only adds
radiate_plus() (no sigma, no sub-cell fraction) and run_point_plus().
Units as in cbim_sc_core: lambda = 1, k = 2 pi, Z0 = 1, e^{-i w t}."""
import numpy as np
from cbim_sc_core import K, cap, csp_fields, obs_plane, lpwa_H, rel, rel_int, rel_max, PPW_LADDER


def radiate_plus(Ro, P, N, w, nxE, nxH, nxH2, chunk_elems=1_000_000):
    """Return E_SC (exact H), E_dbl = 2 sum E_M (sigma = +1) and E_SC+LPWA at observation points Ro (m,3)."""
    m, Ns = Ro.shape[0], P.shape[0]
    Esc = np.zeros((m, 3), np.complex128); Edb = np.zeros_like(Esc); Es2 = np.zeros_like(Esc)
    step = max(1, chunk_elems // Ns)
    wJ, wJ2, wE = w[:, None] * nxH, w[:, None] * nxH2, w[:, None] * nxE
    for i0 in range(0, m, step):
        r = Ro[i0:i0 + step]
        d = r[:, None, :] - P[None, :, :]
        t = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
        kt = K * t
        e = np.exp(1j * kt)
        EM = ((1 - 1j * kt) * e / (4 * np.pi * t**3))[..., None] * np.cross(wE[None, :, :], d)
        g = e / (4 * np.pi * t)
        A = 1 + 1j / kt - 1 / kt**2
        Bc = -1 - 3j / kt + 3 / kt**2
        dJ = np.einsum("ijk,jk->ij", d, wJ) / t
        EJ = (1j * K * g)[..., None] * (A[..., None] * wJ[None] + (Bc * dJ / t)[..., None] * d)
        dJ2 = np.einsum("ijk,jk->ij", d, wJ2) / t
        EJ2 = (1j * K * g)[..., None] * (A[..., None] * wJ2[None] + (Bc * dJ2 / t)[..., None] * d)
        sM = EM.sum(axis=1)
        Esc[i0:i0 + step] = sM + EJ.sum(axis=1)
        Es2[i0:i0 + step] = sM + EJ2.sum(axis=1)
        Edb[i0:i0 + step] = 2 * sM
    return Esc, Edb, Es2


def _richardson(hist, key, Esc, wo):
    """Extrapolated field from the last two levels; order fitted from three levels when possible (else p = 2)."""
    h1, h2 = 1 / hist[-2]["ppw"], 1 / hist[-1]["ppw"]
    p = 2.0
    if len(hist) >= 3:
        e0, e1, e2 = (rel(hist[i][key], Esc, wo) for i in (-3, -2, -1))
        d1, d2 = e1 - e0, e2 - e1
        q = (hist[-2]["ppw"] / hist[-3]["ppw"] * hist[-1]["ppw"] / hist[-2]["ppw"]) ** 0.5
        if d1 != 0 and d2 != 0 and d2 / d1 > 0:
            p = float(np.clip(np.log(d1 / d2) / np.log(q), 1.0, 4.0))
    r = (h2 / h1) ** p
    return hist[-1][key] + (hist[-1][key] - hist[-2][key]) * r / (1 - r), p


def run_point_plus(a, s, nobs=49, tol=1e-3, floor=1e-8, w0_over_a=1/5.5, ladder=PPW_LADDER):
    """Same stopping rule as cbim_sc_core.run_point_adaptive, applied to SC, the doubled formula and SC+LPWA."""
    w0 = w0_over_a * a
    b = K * w0**2 / 2
    hist = []
    for ppw in ladder:
        P, N, w, zv, h, hTx, hTy, kappa = cap(a, s, ppw)
        E, H = csp_fields(P, b)
        Ro, wo = obs_plane(zv, a, w0, nobs)
        H2 = lpwa_H(P, N, E, b)
        Esc, Edb, Es2 = radiate_plus(Ro, P, N, w, np.cross(N, E), np.cross(N, H), np.cross(N, H2))
        hist.append(dict(ppw=ppw, Nsrc=int(P.shape[0]), Esc=Esc, Edb=Edb, Es2=Es2))
        if len(hist) >= 2:
            p0, p1 = hist[-2], hist[-1]
            nrm = lambda X: np.sqrt(np.sum(wo[:, None] * np.abs(X)**2))
            nE = nrm(p1["Esc"])
            ref_d = max(nrm(p1["Edb"] - p1["Esc"]), floor * nE)
            ref_2 = max(nrm(p1["Es2"] - p1["Esc"]), floor * nE)
            dsc = nrm(p1["Esc"] - p0["Esc"]) / min(ref_d, ref_2)
            ddb = nrm(p1["Edb"] - p0["Edb"]) / ref_d
            dc2 = nrm(p1["Es2"] - p0["Es2"]) / ref_2
            if max(dsc, ddb, dc2) < tol:
                break
    Eex, _ = csp_fields(Ro, b)
    last = hist[-1]; Esc = last["Esc"]
    out = dict(a=a, s=s, nobs=nobs, w0=w0, ppw_used=last["ppw"], Nsrc=last["Nsrc"], levels=len(hist),
               err_sc_exact=rel(Esc, Eex, wo),
               err_dbl_sc=rel(last["Edb"], Esc, wo), err_dbl_sc_int=rel_int(last["Edb"], Esc, wo),
               err_dbl_sc_max=rel_max(last["Edb"], Esc), err_dbl_exact=rel(last["Edb"], Eex, wo),
               err_sclpwa_sc=rel(last["Es2"], Esc, wo), err_sclpwa_sc_int=rel_int(last["Es2"], Esc, wo),
               err_sclpwa_exact=rel(last["Es2"], Eex, wo))
    if len(hist) >= 2:
        Edx, p = _richardson(hist, "Edb", Esc, wo)
        E2x, p2 = _richardson(hist, "Es2", Esc, wo)
        out.update(err_dbl_sc_extrap=rel(Edx, Esc, wo), err_dbl_sc_int_extrap=rel_int(Edx, Esc, wo),
                   err_sclpwa_sc_extrap=rel(E2x, Esc, wo), richardson_order_dbl=p, richardson_order_lpwa=p2,
                   est_rel_uncertainty_dbl=abs(out["err_dbl_sc"] - rel(Edx, Esc, wo)) / max(rel(Edx, Esc, wo), 1e-300),
                   est_rel_uncertainty_lpwa=abs(out["err_sclpwa_sc"] - rel(E2x, Esc, wo)) / max(rel(E2x, Esc, wo), 1e-300),
                   last_step_change_sc=rel(hist[-1]["Esc"], hist[-2]["Esc"], wo))
    out["frac_sigma_neg"] = float(np.mean(np.einsum("ijk,jk->ij",
        Ro[::max(1, Ro.shape[0] // 64), None, :] - P[None], N) < 0))
    return out
