"""Focusing-beam test: an exact focused beam passes through a spherical cap and comes to a focus beyond it.
Compared on the focal plane, all driven by the field on the cap:
  SC      Stratton-Chu with exact E and H on the cap (reference; also checked against the exact beam)
  dbl     one-sided CBIM (sigma = +1) = Smythe's formula = 2 x magnetic-current term, E only
  lpwa    Stratton-Chu with H estimated from E by the local plane-wave approximation, E only
  two     two-sided CBIM, sigma = sign(d.n), exact sub-cell treatment of the sign jump (as in the map), E only
Field: aplanatic x-polarised focused beam written as a FINITE sum of plane waves (each an exact Maxwell solution, so the
sum is exact everywhere, with no singularity), focused at (0,0,zf):
  E(r) = sum_q c_q p_q exp(i k s_q.(r - r_f)),  H = sum_q c_q (s_q x p_q) exp(...)   (Z0 = 1)
  p = cos(phi) theta_hat - sin(phi) phi_hat,  c ~ sqrt(cos th) exp(-(sin th / NA)^2) bump(sin th / sin th_max)
  bump(x) = exp(1 - 1/(1 - x^2)) (smooth cut-off),  sin th_max = min(0.97, 1.8 NA).
NA is the 1/e half-width of the angular amplitude in sin(theta). Units as in cbim_sc_core: lambda = 1, k = 2 pi, e^{-i w t}."""
import numpy as np
from cbim_sc_core import K, Z_RIM, frac_positive, rel, rel_int, rel_max

LADDER = (5, 7, 10, 14, 20, 28, 40)


class FocusedBeam:
    """Exact focused beam as a finite plane-wave sum, sampled finely enough to be a smooth beam within radius D of focus."""
    def __init__(self, na, zf, D):
        self.na, self.zf = na, zf
        self.smax = min(0.97, 1.8 * na); thmax = np.arcsin(self.smax)
        nth = int(np.ceil(1.3 * K * D * thmax / np.pi)) + 24
        nph = int(np.ceil(2.2 * K * D * self.smax)) + 32
        u, wu = np.polynomial.legendre.leggauss(nth)
        cmin = np.cos(thmax); ct = 0.5 * (u + 1) * (1 - cmin) + cmin; wct = 0.5 * (1 - cmin) * wu
        st = np.sqrt(1 - ct**2); ph = 2 * np.pi * np.arange(nph) / nph
        CT, PH = np.meshgrid(ct, ph, indexing="ij"); ST = np.sqrt(1 - CT**2)
        x = ST / self.smax
        bump = np.where(x < 1, np.exp(1 - 1 / np.clip(1 - x**2, 1e-300, None)), 0.0)
        amp = np.sqrt(CT) * np.exp(-(ST / na)**2) * bump
        W = (wct[:, None] * (2 * np.pi / nph)) * amp
        s = np.stack([ST * np.cos(PH), ST * np.sin(PH), CT], -1)
        th_hat = np.stack([CT * np.cos(PH), CT * np.sin(PH), -ST], -1)
        ph_hat = np.stack([-np.sin(PH), np.cos(PH), np.zeros_like(PH)], -1)
        p = np.cos(PH)[..., None] * th_hat - np.sin(PH)[..., None] * ph_hat
        keep = W.ravel() > 1e-16 * np.abs(W).max()
        self.s = s.reshape(-1, 3)[keep]; self.p = p.reshape(-1, 3)[keep]; self.c = W.ravel()[keep].astype(complex)
        self.q = np.cross(self.s, self.p)
        self.nwaves = int(keep.sum())
        E0, _, _ = self.fields(np.array([[0.0, 0.0, zf]]))
        self.c /= np.linalg.norm(E0[0])                       # |E(focus)| = 1

    def fields(self, P, want_grad=False, chunk=4_000_000):
        """E, H and (optionally) grad E [i, j, l] = d E_j / d x_l at points P (n,3)."""
        n = P.shape[0]; E = np.zeros((n, 3), complex); H = np.zeros((n, 3), complex)
        G = np.zeros((n, 3, 3), complex) if want_grad else None
        step = max(1, chunk // self.nwaves); rf = np.array([0.0, 0.0, self.zf])
        for i0 in range(0, n, step):
            ph = np.exp(1j * K * ((P[i0:i0 + step] - rf) @ self.s.T)) * self.c[None, :]
            E[i0:i0 + step] = ph @ self.p; H[i0:i0 + step] = ph @ self.q
            if want_grad:
                G[i0:i0 + step] = 1j * K * np.einsum("nq,qj,ql->njl", ph, self.p, self.s, optimize=True)
        return E, H, G


def cap_oriented(a, s, ppw, concave):
    """Spherical cap, aperture radius a, s = a/Rc, rim at z = Z_RIM, normal towards +z (towards the focus).
    convex: bulges towards +z (as in the map). concave: bowl opening towards +z.
    Returns P, N, w, z_vertex, h, hTx, hTy, kappa (kappa = d n / d o along the surface, signed)."""
    nzmin = np.sqrt(1 - s**2)
    h = nzmin / ppw
    n1 = int(np.ceil(a / h))
    x = np.arange(-n1, n1 + 1) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho2 = X**2 + Y**2; m = rho2 <= a**2
    X, Y, rho2 = X[m], Y[m], rho2[m]
    if s == 0:
        Z = np.full_like(X, Z_RIM); N = np.zeros((X.size, 3)); N[:, 2] = 1
        w = np.full(X.size, h * h); zv = Z_RIM; kappa = 0.0; zx = zy = np.zeros_like(X)
    else:
        Rc = a / s; root = np.sqrt(Rc**2 - rho2); c = np.sqrt(Rc**2 - a**2)
        if not concave:
            zc = Z_RIM - c; Z = zc + root; N = np.stack([X, Y, Z - zc], axis=1) / Rc
            zv = zc + Rc; kappa = 1.0 / Rc; zx, zy = -X / root, -Y / root
        else:
            zc = Z_RIM + c; Z = zc - root; N = np.stack([-X, -Y, zc - Z], axis=1) / Rc
            zv = zc - Rc; kappa = -1.0 / Rc; zx, zy = X / root, Y / root
        w = h * h * Rc / root
    P = np.stack([X, Y, Z], axis=1)
    hTx = h * np.stack([np.ones_like(X), np.zeros_like(X), zx], axis=1)
    hTy = h * np.stack([np.zeros_like(X), np.ones_like(X), zy], axis=1)
    return P, N, w, zv, h, hTx, hTy, kappa


def z_vertex(a, s, concave):
    return Z_RIM if s == 0 else Z_RIM + (a / s - np.sqrt((a / s)**2 - a**2)) * (-1 if concave else 1)


def beam_for(a, s, concave, na, zf):
    zv = z_vertex(a, s, concave)
    D = max(np.hypot(a, zf - Z_RIM), zf - zv) + 1.0
    return FocusedBeam(na, zf, D)


def rim_ratio(a, s, concave, na, zf):
    """max |E| on the rim circle / max |E| along cap radii (the beam peaks near the axis; 4 radii x 96 points)."""
    beam = beam_for(a, s, concave, na, zf)
    phi = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    rim = np.stack([a * np.cos(phi), a * np.sin(phi), np.full_like(phi, Z_RIM)], 1)
    rho = np.linspace(0, a, 96)
    zr = np.full_like(rho, Z_RIM) if s == 0 else (
        Z_RIM + (np.sqrt((a / s)**2 - rho**2) - np.sqrt((a / s)**2 - a**2)) * (-1 if concave else 1))
    rad = np.concatenate([np.stack([rho * np.cos(q), rho * np.sin(q), zr], 1) for q in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)])
    Er, _, _ = beam.fields(rim); Ec, _, _ = beam.fields(rad)
    return np.linalg.norm(Er, axis=1).max() / np.linalg.norm(Ec, axis=1).max()


def choose_focus(a, s, concave, na, rim_target=1e-4, gap=2.0):
    """Largest focal distance F (from the rim plane) at which the field on the rim is still below rim_target x the cap
    maximum, i.e. the beam fills the cap as far as the rim level allows. A high-NA beam cannot vanish faster at a finite
    rim (its spectrum must reach large angles), so rim_target is a parameter, and the Stratton-Chu-vs-exact error of every
    point is recorded as its floor. Returns (zf, status): 'ok', 'focus_inside_cap' or 'no_solution'."""
    zv = z_vertex(a, s, concave)
    Fmin = max(0.5, zv - Z_RIM + gap)
    smax = min(0.97, 1.8 * na); Fgeo = a * np.sqrt(1 - smax**2) / smax       # rim exactly at the spectral cut-off
    if rim_ratio(a, s, concave, na, Z_RIM + Fmin) > rim_target:
        return Z_RIM + Fmin, ("focus_inside_cap" if Fgeo < Fmin else "no_solution")
    lo, hi = Fmin, Fmin * 1.5
    while rim_ratio(a, s, concave, na, Z_RIM + hi) <= rim_target:
        lo, hi = hi, hi * 1.5
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if rim_ratio(a, s, concave, na, Z_RIM + mid) <= rim_target: lo = mid
        else: hi = mid
    return Z_RIM + lo, "ok"


def lpwa_H(P, N, E, G):
    """cbim_sc_core.lpwa_H with the phase gradient taken analytically: s = grad_t(phi)/k + n sqrt(1-|.|^2), H = s x E.
    phi = phase of the dominant Cartesian component of E; G = grad E [i, j, l]."""
    j = np.argmax(np.abs(E), axis=1); idx = np.arange(E.shape[0])
    g = np.imag(G[idx, j, :] / E[idx, j][:, None])
    gt = g - np.sum(g * N, axis=1)[:, None] * N
    st = gt / K; sn = np.sqrt(np.clip(1 - np.sum(st * st, axis=1), 0, None))
    return np.cross(st + sn[:, None] * N, E)


def radiate_focus(Ro, P, N, w, nxE, nxH, nxH2, hTx, hTy, kappa, chunk_elems=1_000_000):
    """Return E_SC, E_dbl (sigma=+1), E_SC+LPWA, E_two (sigma=sign, exact sub-cell fraction) at Ro (m,3)."""
    m, Ns = Ro.shape[0], P.shape[0]
    Esc = np.zeros((m, 3), np.complex128); Edb = np.zeros_like(Esc); Es2 = np.zeros_like(Esc); Etw = np.zeros_like(Esc)
    step = max(1, chunk_elems // Ns)
    wJ, wJ2, wE = w[:, None] * nxH, w[:, None] * nxH2, w[:, None] * nxE
    curved = kappa != 0.0
    NhTx, NhTy = np.einsum("jk,jk->j", N, hTx)[None], np.einsum("jk,jk->j", N, hTy)[None]
    for i0 in range(0, m, step):
        r = Ro[i0:i0 + step]
        d = r[:, None, :] - P[None, :, :]
        t = np.sqrt(np.einsum("ijk,ijk->ij", d, d)); kt = K * t; e = np.exp(1j * kt)
        EM = ((1 - 1j * kt) * e / (4 * np.pi * t**3))[..., None] * np.cross(wE[None, :, :], d)
        g = e / (4 * np.pi * t); A = 1 + 1j / kt - 1 / kt**2; Bc = -1 - 3j / kt + 3 / kt**2
        dJ = np.einsum("ijk,jk->ij", d, wJ) / t
        EJ = (1j * K * g)[..., None] * (A[..., None] * wJ[None] + (Bc * dJ / t)[..., None] * d)
        dJ2 = np.einsum("ijk,jk->ij", d, wJ2) / t
        EJ2 = (1j * K * g)[..., None] * (A[..., None] * wJ2[None] + (Bc * dJ2 / t)[..., None] * d)
        f0 = np.einsum("ijk,jk->ij", d, N)
        if curved:
            ga = np.abs(kappa * np.einsum("ijk,jk->ij", d, hTx) - NhTx)
            gb = np.abs(kappa * np.einsum("ijk,jk->ij", d, hTy) - NhTy)
            fp = frac_positive(f0, ga, gb)
        else:
            fp = (f0 > 0).astype(float)
        sM = EM.sum(axis=1)
        Esc[i0:i0 + step] = sM + EJ.sum(axis=1)
        Es2[i0:i0 + step] = sM + EJ2.sum(axis=1)
        Edb[i0:i0 + step] = 2 * sM
        Etw[i0:i0 + step] = (2 * (2 * fp - 1)[..., None] * EM).sum(axis=1)
    return Esc, Edb, Es2, Etw


def focal_plane(zf, w0, nobs, half_width):
    L = half_width * w0; u = np.linspace(-L, L, nobs); X, Y = np.meshgrid(u, u, indexing="ij")
    wt = np.ones(nobs); wt[1:-1:2] = 4; wt[2:-1:2] = 2; wt /= 3          # Simpson (nobs odd)
    return np.stack([X.ravel(), Y.ravel(), np.full(X.size, zf)], axis=1), np.outer(wt, wt).ravel()


def _richardson(hist, key, Esc, wo):
    h1, h2 = 1 / hist[-2]["ppw"], 1 / hist[-1]["ppw"]; p = 2.0
    if len(hist) >= 3:
        e0, e1, e2 = (rel(hist[i][key], Esc, wo) for i in (-3, -2, -1)); d1, d2 = e1 - e0, e2 - e1
        q = (hist[-2]["ppw"] / hist[-3]["ppw"] * hist[-1]["ppw"] / hist[-2]["ppw"]) ** 0.5
        if d1 != 0 and d2 != 0 and d2 / d1 > 0:
            p = float(np.clip(np.log(d1 / d2) / np.log(q), 1.0, 4.0))
    r = (h2 / h1) ** p
    return hist[-1][key] + (hist[-1][key] - hist[-2][key]) * r / (1 - r), p


def run_point_focus(a, s, na, concave, nobs=49, half_width=3.0, tol=1e-4, floor=1e-10, ladder=LADDER, nobs_mid=25,
                    rim_target=1e-4, zf=None):
    """zf: precomputed focus (reference/focus_choice.json); chosen here if None."""
    if zf is None:
        zf, status = choose_focus(a, s, concave, na, rim_target)
    else:
        status = "ok"
    base = dict(a=a, s=s, na=na, concave=bool(concave), status=status, sin_theta_max=min(0.97, 1.8 * na),
                rim_target=rim_target)
    if status != "ok":
        return base
    beam = beam_for(a, s, concave, na, zf)
    w0 = 1.0 / (np.pi * na)                                   # focal spot scale (Gaussian waist for amplitude 1/e at NA)
    hist = []
    for ppw in ladder:
        P, N, w, zv, h, hTx, hTy, kappa = cap_oriented(a, s, ppw, concave)
        Ro, wo = focal_plane(zf, w0, nobs, half_width)
        zm = 0.5 * (zv + zf)                                   # second plane, between cap and focus
        wm = max(w0, (zf - zm) * na / np.sqrt(1 - na**2))     # beam radius (amplitude 1/e) there
        Rm, wm_ = focal_plane(zm, 1.0, nobs_mid, 2.5 * wm)
        E, H, G = beam.fields(P, want_grad=True)
        H2 = lpwa_H(P, N, E, G)
        out = radiate_focus(np.vstack([Ro, Rm]), P, N, w, np.cross(N, E), np.cross(N, H), np.cross(N, H2), hTx, hTy, kappa)
        nf = Ro.shape[0]
        Esc, Edb, Es2, Etw = (x[:nf] for x in out)
        hist.append(dict(ppw=ppw, Nsrc=int(P.shape[0]), Esc=Esc, Edb=Edb, Es2=Es2, Etw=Etw,
                         mid=dict(Esc=out[0][nf:], Edb=out[1][nf:], Es2=out[2][nf:], Etw=out[3][nf:])))
        if len(hist) >= 2:
            p0, p1 = hist[-2], hist[-1]
            nrm = lambda X: np.sqrt(np.sum(wo[:, None] * np.abs(X)**2)); nE = nrm(p1["Esc"])
            refs = {k: max(nrm(p1[k] - p1["Esc"]), floor * nE) for k in ("Edb", "Es2", "Etw")}
            ch = [nrm(p1["Esc"] - p0["Esc"]) / min(refs.values())] + [nrm(p1[k] - p0[k]) / refs[k] for k in refs]
            if max(ch) < tol:
                break
    last = hist[-1]; Esc = last["Esc"]
    Eex, _, _ = beam.fields(Ro); Emid_ex, _, _ = beam.fields(Rm); mid = last["mid"]
    res = dict(base, zf=zf, F_from_rim=zf - Z_RIM, z_vertex=zv, w0=w0, nwaves=beam.nwaves, nobs=nobs, half_width=half_width,
               tol=tol, floor=floor, rim_ratio=rim_ratio(a, s, concave, na, zf),
               ppw_used=last["ppw"], Nsrc=last["Nsrc"], levels=len(hist), converged=bool(len(hist) < len(ladder)),
               err_dbl_sc=rel(last["Edb"], Esc, wo), err_dbl_sc_int=rel_int(last["Edb"], Esc, wo),
               err_lpwa_sc=rel(last["Es2"], Esc, wo), err_lpwa_sc_int=rel_int(last["Es2"], Esc, wo),
               err_two_sc=rel(last["Etw"], Esc, wo), err_two_sc_int=rel_int(last["Etw"], Esc, wo),
               err_sc_exact=rel(Esc, Eex, wo), err_dbl_exact=rel(last["Edb"], Eex, wo), err_lpwa_exact=rel(last["Es2"], Eex, wo),
               err_sc_exact_mid=rel(mid["Esc"], Emid_ex, wm_), err_dbl_sc_mid=rel(mid["Edb"], mid["Esc"], wm_),
               err_lpwa_sc_mid=rel(mid["Es2"], mid["Esc"], wm_), err_two_sc_mid=rel(mid["Etw"], mid["Esc"], wm_),
               peak_SC=float(np.max(np.linalg.norm(Esc, axis=1))))
    if len(hist) >= 2:
        res["last_step_change_sc"] = rel(hist[-1]["Esc"], hist[-2]["Esc"], wo)
        for key, name in (("Edb", "dbl"), ("Es2", "lpwa"), ("Etw", "two")):
            Ex, p = _richardson(hist, key, Esc, wo)
            ex = rel(Ex, Esc, wo); res[f"err_{name}_sc_extrap"] = ex; res[f"richardson_order_{name}"] = p
            res[f"unc_{name}"] = abs(res[f"err_{name}_sc"] - ex) / max(ex, 1e-300)
    res["frac_sigma_neg"] = float(np.mean(np.einsum("ijk,jk->ij", Ro[::max(1, Ro.shape[0] // 64), None, :] - P[None], N) < 0))
    return res
