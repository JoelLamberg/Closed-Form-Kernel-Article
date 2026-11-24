"""
cbim_core.py -- closed-form, transformation-free CBIM kernels.

Conventions (fixed once, used everywhere):
  time dependence      e^{-i omega t}
  wavenumber           k = 2 pi / lambda,  lambda = 1 throughout
  master integrals     I(t;k) = (1 - ikt) e^{ikt} / t^2
                       J(t;k) = (k^2 t^2 + 2ikt - 2) e^{ikt} / t^2
  separation vector    d = r - o        (observation minus source),  t = |d|
  local basis          (e1, e2, e3) right handed, e2 = e3 x e1, e3 = surface normal
  curl field           M = curl E = i k Z0 H     ->   H = M / (i k Z0)

Transformation-free kernels (no Theta rotation matrix; everything global):

    E(r;o) = 2 pi * sgn * (I/t) * (e2 x d)                       [curl kernel]
    M(r;o) = 2 pi * sgn * Ghat . e2                              [dyadic kernel]
    Ghat   = (I-J)/t^3 * d d^T + (I+J)/t * Identity
    sgn    = sign[(r - o) . e3]

and the surface integrals

    E(r) = 1/(4 pi^2) int_Omega E0(o) E(r;o) dOmega
    M(r) = 1/(4 pi^2) int_Omega E0(o) M(r;o) dOmega

Identity used in the paper:  Ghat = 4 pi k^2 * Gbar, with Gbar the classical
free-space dyadic Green function.  Verified to machine precision in
make_figures.py (Fig. 2).
"""
import numpy as np

LAM = 1.0
K = 2.0 * np.pi / LAM
Z0 = 1.0                                   # normalised impedance


# ----------------------------------------------------------------------
# master integrals
# ----------------------------------------------------------------------
def IJ(t, k=K):
    """Closed-form master integrals I(t;k), J(t;k).  Taylor-safe as t -> 0
    is NOT needed here: every use is off-surface, t > 0."""
    s = k * t
    e = np.exp(1j * s)
    I = (1 - 1j * s) * e / t**2
    J = (s * s + 2j * s - 2) * e / t**2
    return I, J


def IJ_prop(t, k=K):
    """Propagating-only (rho < k) master integrals.  Provided for reference
    ONLY -- see Sec. 6 of the paper: the radial reduction that produces these
    does not survive spectral truncation, so they are not the propagating part
    of the kernel away from the axis.  Not used in any result."""
    s = k * t
    e = np.exp(1j * s)
    return ((1 - 1j * s) * e - 1) / t**2, ((s * s + 2j * s - 2) * e + 2) / t**2


# ----------------------------------------------------------------------
# transformation-free kernels
# ----------------------------------------------------------------------
def E_kernel(d, e2, e3, k=K, signed=True):
    """E(r;o)/(2 pi) for a stack of separation vectors d (N,3).
    Returns (N,3) complex."""
    t = np.linalg.norm(d, axis=-1)
    I, _ = IJ(t, k)
    out = (I / t)[..., None] * np.cross(e2, d)
    if signed:
        out = out * np.sign(np.einsum('...i,...i->...', d, e3))[..., None]
    return 2 * np.pi * out


def Ghat(d, k=K):
    """Dyadic kernel Ghat = (I-J)/t^3 d d^T + (I+J)/t Id.  d: (N,3) -> (N,3,3)."""
    t = np.linalg.norm(d, axis=-1)
    I, J = IJ(t, k)
    dd = d[..., :, None] * d[..., None, :]
    eye = np.broadcast_to(np.eye(3), dd.shape)
    return ((I - J) / t**3)[..., None, None] * dd + ((I + J) / t)[..., None, None] * eye


def M_kernel(d, e2, e3, k=K, signed=True):
    """M(r;o)/(2 pi) -- the curl field, M = curl E = i k Z0 H."""
    G = Ghat(d, k)
    out = np.einsum('...ij,...j->...i', G, e2)
    if signed:
        out = out * np.sign(np.einsum('...i,...i->...', d, e3))[..., None]
    return 2 * np.pi * out


def Gbar(d, k=K):
    """Classical free-space dyadic Green function (for the Ghat = 4 pi k^2 Gbar
    check).  d: (N,3) -> (N,3,3)."""
    t = np.linalg.norm(d, axis=-1)
    a = k * t
    g = np.exp(1j * a) / (4 * np.pi * t)
    rh = d / t[..., None]
    rr = rh[..., :, None] * rh[..., None, :]
    eye = np.broadcast_to(np.eye(3), rr.shape)
    c1 = (1 + 1j / a - 1 / a**2)
    c2 = (-1 - 3j / a + 3 / a**2)
    return g[..., None, None] * (c1[..., None, None] * eye + c2[..., None, None] * rr)


# ----------------------------------------------------------------------
# local-coordinate (paper Eq. 25-29) forms, for cross-checking
# ----------------------------------------------------------------------
def kernels_local(xb, yb, zb, k=K):
    """The five kernels E1,E3,M1,M2,M3 in local coordinates."""
    t = np.sqrt(xb**2 + yb**2 + zb**2)
    e = np.exp(1j * k * t)
    p = (1 - 1j * k * t) * e
    q = (3 - 3j * k * t - (k * t) ** 2) * e
    E1 = 2 * np.pi * zb / t**3 * p
    E3 = -2 * np.pi * xb / t**3 * p
    M1 = 2 * np.pi * xb * yb / t**5 * q
    M2 = 2 * np.pi / t**5 * (2 * (1 - 1j * k * t) * t**2
                             + (xb**2 + zb**2) * ((k * t) ** 2 + 3j * k * t - 3)) * e
    M3 = 2 * np.pi * yb * zb / t**5 * q
    return E1, E3, M1, M2, M3


# ----------------------------------------------------------------------
# surfaces
# ----------------------------------------------------------------------
def spherical_cap(Rc, alpha, nth, nph):
    """Spherical cap, curvature radius Rc, half-angle alpha, apex at the origin,
    opening toward +z.  Gauss-Legendre in cos(theta), uniform in azimuth.
    Returns O (Ns,3), dA (Ns,), e3 (Ns,3) unit normals pointing toward +z."""
    x, w = np.polynomial.legendre.leggauss(nth)
    cmin = np.cos(alpha)
    cth = 0.5 * (1 - cmin) * x + 0.5 * (1 + cmin)
    wth = 0.5 * (1 - cmin) * w
    psi = (np.arange(nph) + 0.5) * 2 * np.pi / nph
    dpsi = 2 * np.pi / nph
    CT, PS = np.meshgrid(cth, psi, indexing='ij')
    ST = np.sqrt(1 - CT**2)
    C = np.array([0.0, 0.0, Rc])
    O = C + Rc * np.stack([ST * np.cos(PS), ST * np.sin(PS), -CT], -1).reshape(-1, 3)
    WTH, _ = np.meshgrid(wth, psi, indexing='ij')
    dA = (Rc**2 * WTH * dpsi).reshape(-1)
    e3 = (C - O) / Rc
    return O, dA, e3


def flat_disk(a, nr, nph):
    """Flat circular aperture of radius a in the z = 0 plane."""
    x, w = np.polynomial.legendre.leggauss(nr)
    r = a * 0.5 * (x + 1)
    wr = a * 0.5 * w
    psi = (np.arange(nph) + 0.5) * 2 * np.pi / nph
    dpsi = 2 * np.pi / nph
    R, PS = np.meshgrid(r, psi, indexing='ij')
    WR, _ = np.meshgrid(wr, psi, indexing='ij')
    O = np.stack([R * np.cos(PS), R * np.sin(PS), np.zeros_like(R)], -1).reshape(-1, 3)
    dA = (R * WR * dpsi).reshape(-1)
    e3 = np.tile([0.0, 0.0, 1.0], (len(O), 1))
    return O, dA, e3


def full_sphere(R, nth, nph):
    """Closed sphere, outward normals -- used for the two-current validation."""
    x, w = np.polynomial.legendre.leggauss(nth)
    cth = x
    psi = (np.arange(nph) + 0.5) * 2 * np.pi / nph
    dpsi = 2 * np.pi / nph
    CT, PS = np.meshgrid(cth, psi, indexing='ij')
    W, _ = np.meshgrid(w, psi, indexing='ij')
    ST = np.sqrt(1 - CT**2)
    n = np.stack([ST * np.cos(PS), ST * np.sin(PS), CT], -1).reshape(-1, 3)
    return R * n, (R**2 * W * dpsi).reshape(-1), n


def tangent_basis(O, e3, pol=np.array([1.0, 0.0, 0.0])):
    """Local (e1, e2) from a global polarisation direction projected onto the
    tangent plane.  e1 along the electric field, e2 = e3 x e1."""
    e1 = pol - e3 * (e3 @ pol)[:, None]
    nrm = np.linalg.norm(e1, axis=1)
    bad = nrm < 1e-12                              # e3 parallel to pol
    if bad.any():                                  # fall back to any tangent dir
        alt = np.array([0.0, 1.0, 0.0])
        e1[bad] = alt - e3[bad] * (e3[bad] @ alt)[:, None]
        nrm = np.linalg.norm(e1, axis=1)
    e1 = e1 / nrm[:, None]
    return e1, np.cross(e3, e1)


# ----------------------------------------------------------------------
# forward operator (the object the optimiser differentiates)
# ----------------------------------------------------------------------
def forward_matrix(O, dA, e3, targets, e2=None, k=K, inc=None, signed=True,
                   chunk=None):
    """G[p, j, c]:  contribution of unit source amplitude at o_j to the c-th
    Cartesian component of E at target p, including the area weight and the
    1/(4 pi^2) prefactor.  `inc` is an optional per-source complex illumination
    factor (e.g. e^{i k s.o} for an incident plane wave).

    E(r_p) = sum_j G[p, j, :] * E0_j
    """
    if e2 is None:
        _, e2 = tangent_basis(O, e3)
    w = dA / (4 * np.pi**2)
    if inc is not None:
        w = w * inc
    P, Ns = len(targets), len(O)
    G = np.empty((P, Ns, 3), complex)
    for p in range(P):
        d = targets[p] - O
        G[p] = w[:, None] * E_kernel(d, e2, e3, k, signed)
    return G


def propagate(O, dA, e3, E0, targets, e2=None, k=K, signed=True, chunk_targets=4000):
    """Memory-lean evaluation of E(r) on many targets.  Same model as
    forward_matrix but without storing the full operator."""
    if e2 is None:
        _, e2 = tangent_basis(O, e3)
    w = (dA / (4 * np.pi**2)) * E0
    out = np.empty((len(targets), 3), complex)
    for p0 in range(0, len(targets), chunk_targets):
        tg = targets[p0:p0 + chunk_targets]
        d = tg[:, None, :] - O[None, :, :]
        t = np.linalg.norm(d, axis=-1)
        I, _ = IJ(t, k)
        c = (I / t)
        if signed:
            c = c * np.sign(np.einsum('pji,ji->pj', d, e3))
        e2b = np.broadcast_to(e2, d.shape)
        out[p0:p0 + chunk_targets] = np.einsum(
            'pj,pjc->pc', 2 * np.pi * c * w, np.cross(e2b, d))
    return out
