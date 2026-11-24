"""CBIM (single-current, closed form) vs Stratton-Chu on curved caps.

Test field : exact complex-source-point (CSP) beam, x-directed dipole at (0,0,i*b),
             waist at z=0, w0^2 = 2b/k. Exact E and H everywhere in z>0.
Surface    : spherical cap, aperture radius a, curvature parameter s = a/RoC
             (s=0 -> plane), rim at z = Z_RIM, bulging towards +z.
Methods    : Stratton-Chu / Love form  E = sum w [ E_M(n x E) + E_J(n x H) ]
             CBIM closed form          E = sum w 2*sigma*E_M(n x E),  sigma = sign(d.n)
Units      : lambda = 1, k = 2*pi, Z0 = 1 (so curl E = i k H), time e^{-i w t}.
"""
import numpy as np

K = 2.0 * np.pi
Z_RIM = 0.5          # rim height (wavelengths) above the CSP waist plane / branch disk


# ---------------------------------------------------------------- exact field
def csp_fields(P, b):
    """Exact E, H of an x-dipole at complex point (0,0,i b). P: (...,3) real."""
    Rv = P.astype(np.complex128)
    Rv[..., 2] -= 1j * b
    R = np.sqrt(np.sum(Rv * Rv, axis=-1))          # principal branch: Re R > 0 for z > 0
    Rh = Rv / R[..., None]
    kR = K * R
    g = np.exp(1j * kR - K * b) / (4 * np.pi * R)   # e^{-kb} normalisation
    A = 1 + 1j / kR - 1 / kR**2
    B = -1 - 3j / kR + 3 / kR**2
    E = (g * B * Rh[..., 0])[..., None] * Rh
    E[..., 0] += g * A
    gradg = (g * (1j * K * R - 1) / R)[..., None] * Rh
    M = np.zeros_like(E)                             # grad g x x_hat = (0, gz, -gy)
    M[..., 1] = gradg[..., 2]
    M[..., 2] = -gradg[..., 1]
    return E, M / (1j * K)


def beam_w(z, w0):
    zR = np.pi * w0**2
    return w0 * np.sqrt(1 + (z / zR) ** 2)


# ---------------------------------------------------------------- geometry
def cap(a, s, ppw):
    """Source points, outward normals (towards +z side), quadrature weights."""
    nzmin = np.sqrt(1 - s**2)
    h = nzmin / ppw                                  # projected spacing -> ~1/ppw along surface
    n1 = int(np.ceil(a / h))
    x = (np.arange(-n1, n1 + 1)) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho2 = X**2 + Y**2
    m = rho2 <= a**2
    X, Y, rho2 = X[m], Y[m], rho2[m]
    if s == 0:
        Z = np.full_like(X, Z_RIM)
        N = np.zeros((X.size, 3)); N[:, 2] = 1
        w = np.full(X.size, h * h)
        zv = Z_RIM
        kappa = 0.0
        zx = zy = np.zeros_like(X)
    else:
        Rc = a / s
        root = np.sqrt(Rc**2 - rho2)
        zc = Z_RIM - np.sqrt(Rc**2 - a**2)          # sphere centre
        Z = zc + root
        N = np.stack([X, Y, Z - zc], axis=1) / Rc
        w = h * h * Rc / root
        zv = zc + Rc
        kappa = 1.0 / Rc
        zx, zy = -X / root, -Y / root
    P = np.stack([X, Y, Z], axis=1)
    hTx = h * np.stack([np.ones_like(X), np.zeros_like(X), zx], axis=1)   # cell edge vectors
    hTy = h * np.stack([np.zeros_like(X), np.ones_like(X), zy], axis=1)
    return P, N, w, zv, h, hTx, hTy, kappa


# ---------------------------------------------------------------- integrals
def frac_positive(f0, ga, gb):
    """Fraction of a parameter cell where f = f0 + u > 0, with u the sum of two uniform
    variables of widths ga = |grad f . hTx|, gb = |grad f . hTy| (f linear over the cell).
    Exact for linear f; turns the sigma step into the exact cell average."""
    a = np.maximum(np.maximum(ga, gb), 1e-300); b = np.maximum(np.minimum(ga, gb), 1e-300)
    v = -f0                                            # P(u > v) = 1 - CDF(v)
    lo, mid = -(a + b) / 2, -(a - b) / 2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cdf = _trap_cdf(v, a, b, lo, mid)
    return 1.0 - cdf


def _trap_cdf(v, a, b, lo, mid):
    return np.where(v <= lo, 0.0,
          np.where(v <= mid, (v - lo) ** 2 / (2 * a * b),
          np.where(v <= -mid, (v + a / 2) / a,
          np.where(v < -lo, 1 - (-lo - v) ** 2 / (2 * a * b), 1.0))))


def radiate(Ro, P, N, w, nxE, nxH, hTx=None, hTy=None, kappa=0.0, chunk_elems=1_000_000, nxH2=None):
    """Return E_SC, E_CBIM (sigma = sign(d.n), both sides) and E_CBIM+ (Omega_+ only:
    source points with d.n > 0) at observation points Ro (m,3)."""
    m, Ns = Ro.shape[0], P.shape[0]
    Esc = np.zeros((m, 3), np.complex128)
    Ecb = np.zeros((m, 3), np.complex128)
    Ecp = np.zeros((m, 3), np.complex128)
    Es2 = np.zeros((m, 3), np.complex128)
    step = max(1, chunk_elems // Ns)
    wJ2 = None if nxH2 is None else w[:, None] * nxH2
    wJ = w[:, None] * nxH
    wE = w[:, None] * nxE
    for i0 in range(0, m, step):
        r = Ro[i0:i0 + step]
        d = r[:, None, :] - P[None, :, :]                     # (c,Ns,3)
        t = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
        kt = K * t
        e = np.exp(1j * kt)
        # magnetic-current term  E_M = I/(4 pi t) (n x E) x d,  I = (1-ikt)e^{ikt}/t^2
        cM = (1 - 1j * kt) * e / (4 * np.pi * t**3)
        cr = np.cross(wE[None, :, :], d)                      # (wE x d)
        EM = cM[..., None] * cr
        # electric-current term  E_J = i k Gbar(d) . (n x H)
        g = e / (4 * np.pi * t)
        A = 1 + 1j / kt - 1 / kt**2
        Bc = -1 - 3j / kt + 3 / kt**2
        dJ = np.einsum("ijk,jk->ij", d, wJ) / t               # dhat . wJ
        EJ = (1j * K * g)[..., None] * (A[..., None] * wJ[None] + (Bc * dJ / t)[..., None] * d)
        f0 = np.einsum("ijk,jk->ij", d, N)                    # d . n  (sigma = sign f0)
        if hTx is None or kappa == 0.0:
            fp = (f0 > 0).astype(float)                       # plane: f constant over a cell
        else:
            # grad_o (d.n) = -n + kappa d ; its components along the cell edges
            ga = np.abs(kappa * np.einsum("ijk,jk->ij", d, hTx) - np.einsum("jk,jk->j", N, hTx)[None])
            gb = np.abs(kappa * np.einsum("ijk,jk->ij", d, hTy) - np.einsum("jk,jk->j", N, hTy)[None])
            fp = frac_positive(f0, ga, gb)                    # exact cell-average of [f > 0]
        Esc[i0:i0 + step] = (EM + EJ).sum(axis=1)
        if wJ2 is not None:                                   # Stratton-Chu with estimated H
            dJ2 = np.einsum("ijk,jk->ij", d, wJ2) / t
            EJ2 = (1j * K * g)[..., None] * (A[..., None] * wJ2[None] + (Bc * dJ2 / t)[..., None] * d)
            Es2[i0:i0 + step] = (EM + EJ2).sum(axis=1)
        Ecb[i0:i0 + step] = (2 * (2 * fp - 1)[..., None] * EM).sum(axis=1)
        Ecp[i0:i0 + step] = (2 * fp[..., None] * EM).sum(axis=1)
    if nxH2 is not None:
        return Esc, Ecb, Ecp, Es2
    return Esc, Ecb, Ecp


def lpwa_H(P, N, E, b, delta=1e-4):
    """Local plane-wave estimate of H from E ON THE SURFACE ONLY (what a user with only E
    would do): s = grad_t(phi)/k + n sqrt(1 - |grad_t(phi)/k|^2), H = s x E (Z0 = 1).
    phi = phase of the dominant Cartesian component; the tangential phase gradient is taken
    from the exact field by central differences projected onto the tangent plane (i.e. the
    gradient of the prescribed surface data, not of the off-surface field)."""
    j = np.argmax(np.abs(E), axis=1)
    idx = np.arange(E.shape[0])
    g = np.zeros_like(P)
    for ax in range(3):
        dP = np.zeros(3); dP[ax] = delta
        Ep, _ = csp_fields(P + dP, b); Em, _ = csp_fields(P - dP, b)
        g[:, ax] = np.imag((Ep[idx, j] - Em[idx, j]) / (2 * delta) / E[idx, j])
    gt = g - np.sum(g * N, axis=1)[:, None] * N              # tangential phase gradient
    st = gt / K
    sn = np.sqrt(np.clip(1 - np.sum(st * st, axis=1), 0, None))
    s_hat = st + sn[:, None] * N
    return np.cross(s_hat, E)


def obs_plane(zv, a, w0, nobs):
    """Observation plane at distance a beyond the cap vertex, window +-2.5 w(z).
    Returns points (nobs^2, 3) and 2-D quadrature weights for the L2 metrics
    (Simpson for odd nobs, trapezoid for even)."""
    zo = zv + a
    L = 2.5 * beam_w(zo, w0)
    u = np.linspace(-L, L, nobs)
    X, Y = np.meshgrid(u, u, indexing="ij")
    if nobs % 2 == 1 and nobs >= 3:                    # Simpson (4th order) for odd nobs
        wt = np.ones(nobs); wt[1:-1:2] = 4; wt[2:-1:2] = 2; wt /= 3
    else:                                              # trapezoid (2nd order) otherwise
        wt = np.ones(nobs); wt[[0, -1]] = 0.5
    return np.stack([X.ravel(), Y.ravel(), np.full(X.size, zo)], axis=1), np.outer(wt, wt).ravel()


def rel(a, b, wo=None):
    """relative L2 norm over the observation window (trapezoidal weights wo)."""
    wo = np.ones(a.shape[0]) if wo is None else wo
    return float(np.sqrt(np.sum(wo[:, None] * np.abs(a - b)**2) / np.sum(wo[:, None] * np.abs(b)**2)))


def rel_max(a, b):
    """max over observation points of |a-b|, relative to max |b| (vector norms per point)."""
    return float(np.max(np.linalg.norm(a - b, axis=1)) / np.max(np.linalg.norm(b, axis=1)))


def rel_int(a, b, wo=None):
    """relative L2 error of the intensity |E|^2 (trapezoidal weights wo)."""
    wo = np.ones(a.shape[0]) if wo is None else wo
    Ia, Ib = np.sum(np.abs(a)**2, axis=1), np.sum(np.abs(b)**2, axis=1)
    return float(np.sqrt(np.sum(wo * (Ia - Ib)**2) / np.sum(wo * Ib**2)))


def run_point(a, s, ppw=5, nobs=48, w0_over_a=1/5.5):
    """One map point. Returns dict with errors and sizes."""
    w0 = w0_over_a * a
    b = K * w0**2 / 2
    P, N, w, zv, h, hTx, hTy, kappa = cap(a, s, ppw)
    E, H = csp_fields(P, b)
    nxE = np.cross(N, E)
    nxH = np.cross(N, H)
    Ro, wo = obs_plane(zv, a, w0, nobs)
    Eex, _ = csp_fields(Ro, b)
    Esc, Ecb, Ecp = radiate(Ro, P, N, w, nxE, nxH, hTx, hTy, kappa)
    return dict(a=a, s=s, ppw=ppw, nobs=nobs, Nsrc=int(P.shape[0]), w0=w0,
                err_cbim_sc=rel(Ecb, Esc, wo),        # main quantity (L2, all components)
                err_cbim_sc_max=rel_max(Ecb, Esc),
                err_cbim_sc_int=rel_int(Ecb, Esc, wo),
                err_cbimplus_sc=rel(Ecp, Esc, wo),    # Omega_+ variant of CBIM
                err_sc_exact=rel(Esc, Eex, wo),       # validation floor
                err_cbim_exact=rel(Ecb, Eex, wo),
                frac_sigma_neg=float(np.mean(np.sign(np.einsum("ijk,jk->ij",
                    Ro[::max(1, Ro.shape[0]//64), None, :] - P[None], N)) < 0)))


# ---------------------------------------------------------------- adaptive, self-certified point
PPW_LADDER = (5, 7, 10, 14, 20, 28)


def run_point_adaptive(a, s, nobs=48, tol=1e-3, floor=1e-8, w0_over_a=1/5.5, ladder=PPW_LADDER):
    """Refine surface sampling until the CBIM and SC fields change by < tol relative to the
    CBIM-SC difference (floored at floor*|E_SC|) between consecutive ladder steps, then Richardson-extrapolate the error metrics (order fitted
    from the last three levels when available, else p = 2). Returns metrics + certificate."""
    w0 = w0_over_a * a
    b = K * w0**2 / 2
    hist = []
    for ppw in ladder:
        P, N, w, zv, h, hTx, hTy, kappa = cap(a, s, ppw)
        E, H = csp_fields(P, b)
        Ro, wo = obs_plane(zv, a, w0, nobs)
        H2 = lpwa_H(P, N, E, b)
        Esc, Ecb, Ecp, Es2 = radiate(Ro, P, N, w, np.cross(N, E), np.cross(N, H), hTx, hTy, kappa,
                                     nxH2=np.cross(N, H2))
        hist.append(dict(ppw=ppw, Nsrc=int(P.shape[0]), Esc=Esc, Ecb=Ecb, Ecp=Ecp, Es2=Es2))
        if len(hist) >= 2:
            p0, p1 = hist[-2], hist[-1]
            nrm = lambda X: np.sqrt(np.sum(wo[:, None] * np.abs(X)**2))
            nE = nrm(p1["Esc"])
            # changes are measured against the quantity reported: the CBIM - SC difference
            ref_b = max(nrm(p1["Ecb"] - p1["Esc"]), floor * nE)
            ref_p = max(nrm(p1["Ecp"] - p1["Esc"]), floor * nE)
            dsc = nrm(p1["Esc"] - p0["Esc"]) / min(ref_b, ref_p)
            dcb = nrm(p1["Ecb"] - p0["Ecb"]) / ref_b
            dcp = nrm(p1["Ecp"] - p0["Ecp"]) / ref_p
            ref_2 = max(nrm(p1["Es2"] - p1["Esc"]), floor * nE)
            dc2 = nrm(p1["Es2"] - p0["Es2"]) / ref_2
            if max(dsc, dcb, dcp, dc2) < tol:
                break
    Eex, _ = csp_fields(Ro, b)
    last = hist[-1]
    Esc = last["Esc"]

    def metrics(Ecb, Ecp):
        return dict(err_cbim_sc=rel(Ecb, Esc, wo), err_cbim_sc_max=rel_max(Ecb, Esc),
                    err_cbim_sc_int=rel_int(Ecb, Esc, wo), err_cbimplus_sc=rel(Ecp, Esc, wo),
                    err_cbim_exact=rel(Ecb, Eex, wo))

    out = dict(a=a, s=s, nobs=nobs, w0=w0, ppw_used=last["ppw"], Nsrc=last["Nsrc"],
               levels=len(hist), err_sc_exact=rel(Esc, Eex, wo))
    out.update(metrics(last["Ecb"], last["Ecp"]))
    # third method: Stratton-Chu with H estimated by the local plane-wave approximation
    out["err_sclpwa_sc"] = rel(last["Es2"], Esc, wo)
    out["err_sclpwa_sc_int"] = rel_int(last["Es2"], Esc, wo)
    out["err_sclpwa_exact"] = rel(last["Es2"], Eex, wo)
    # Richardson on the CBIM fields (SC is spectrally converged; its change is reported)
    if len(hist) >= 2:
        h1, h2 = 1 / hist[-2]["ppw"], 1 / hist[-1]["ppw"]
        p = 2.0                                     # default order (sub-cell jump treatment)
        if len(hist) >= 3:                          # fit order from the last three levels
            e0, e1, e2 = (rel(hist[i]["Ecb"], Esc, wo) for i in (-3, -2, -1))
            d1, d2 = e1 - e0, e2 - e1
            q = (hist[-2]["ppw"] / hist[-3]["ppw"] * hist[-1]["ppw"] / hist[-2]["ppw"]) ** 0.5
            if d1 != 0 and d2 != 0 and d2 / d1 > 0:
                p = float(np.clip(np.log(d1 / d2) / np.log(q), 1.0, 4.0))
        r = (h2 / h1) ** p
        Ecb_x = hist[-1]["Ecb"] + (hist[-1]["Ecb"] - hist[-2]["Ecb"]) * r / (1 - r)
        Ecp_x = hist[-1]["Ecp"] + (hist[-1]["Ecp"] - hist[-2]["Ecp"]) * r / (1 - r)
        mx = metrics(Ecb_x, Ecp_x)
        out.update({k + "_extrap": v for k, v in mx.items()})
        out["richardson_order"] = p
        out["est_rel_uncertainty"] = abs(mx["err_cbim_sc"] - out["err_cbim_sc"]) / max(mx["err_cbim_sc"], 1e-300)
        out["last_step_change_sc"] = rel(hist[-1]["Esc"], hist[-2]["Esc"], wo)
    out["frac_sigma_neg"] = float(np.mean(np.einsum("ijk,jk->ij",
        Ro[::max(1, Ro.shape[0] // 64), None, :] - P[None], N) < 0))
    return out
