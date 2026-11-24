"""Second-order local angular-spectrum estimate of H from E on the surface (LAS2).

The tangent-plane angular spectrum gives the normal derivative of each Cartesian field
component through the Dirichlet-to-Neumann symbol k_z(q) = sqrt(k^2 - |q|^2). Expanding it
around the local tangential wavevector kappa (phase gradient of the dominant component) to
second order in q - kappa gives, with U = E e^{-i kappa.x},
    d_n E = i kz0 U - (kappa.grad U)/kz0 + i lap U/(2 kz0) + i (kappa.grad)^2 U/(2 kz0^3)
at the point (x = 0). Order 0 is the local plane wave, order 2 is exact for any plane wave
and removes the leading angular-spread error of LPWA. Only surface data are used: first and
second derivatives along the surface in geodesic normal coordinates (exact sphere map), by
central differences of the prescribed surface field. curl E = t1 x E_u + t2 x E_v + n x d_n E,
H = curl E / (i k) (Z0 = 1). The curvature of the surface is not included in the DtN symbol,
which is the approximation."""
import numpy as np


def tangent_frame(N):
    a = np.where(np.abs(N[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = a - np.sum(a * N, 1)[:, None] * N; t1 /= np.linalg.norm(t1, axis=1)[:, None]
    return t1, np.cross(N, t1)


def surf_points(P, N, t1, t2, u, v, center, Rc):
    """Point at geodesic normal coordinates (u, v) from P on the sphere (centre, Rc), or on the plane if Rc is None."""
    if Rc is None:
        return P + u * t1 + v * t2
    m0 = (P - center) / Rc                       # unit vector from centre to P (= +-N)
    r = np.hypot(u, v)
    if r == 0:
        return P.copy()
    dirv = (u * t1 + v * t2) / r
    return center + Rc * (m0 * np.cos(r / Rc) + dirv * np.sin(r / Rc))


def las_H(P, N, E, field_E, center=None, Rc=None, order=2, curvature=True, delta=1e-3, K=2 * np.pi, kappa_max=0.995, guard=1.0):
    """H estimated from surface E. field_E(points) -> E (complex, (n,3)) on the surface.
    curvature: add the first-order curvature term of the DtN map for a sphere, -(c/Rc)(1 + |kappa|^2/(2 kz0^2)),
    c = +1 if the surface bends away from N (convex towards N), -1 if towards it; exact for a spherical wave from
    the centre. kappa (intensity-weighted phase gradient) is clamped to kappa_max k. guard: where the expansion terms exceed guard
    times the leading term the point falls back to order 0 (the expansion is not valid there; matters only near field
    zeros and for very non-paraxial beams)."""
    t1, t2 = tangent_frame(N)
    F = lambda u, v: field_E(surf_points(P, N, t1, t2, u, v, center, Rc))
    Eu = (F(delta, 0) - F(-delta, 0)) / (2 * delta)
    Ev = (F(0, delta) - F(0, -delta)) / (2 * delta)
    if order >= 2:
        Euu = (F(delta, 0) - 2 * E + F(-delta, 0)) / delta**2
        Evv = (F(0, delta) - 2 * E + F(0, -delta)) / delta**2
        Euv = (F(delta, delta) - F(delta, -delta) - F(-delta, delta) + F(-delta, -delta)) / (4 * delta**2)
    nE2 = np.maximum(np.sum(np.abs(E)**2, axis=1), 1e-300)
    ku = np.imag(np.sum(np.conj(E) * Eu, axis=1)) / nE2         # intensity-weighted local wavevector (robust near
    kv = np.imag(np.sum(np.conj(E) * Ev, axis=1)) / nE2         # zeros of any single component)
    k2 = ku**2 + kv**2
    sc = np.minimum(1.0, kappa_max * K / np.sqrt(np.maximum(k2, 1e-300))); ku, kv = ku * sc, kv * sc
    k2 = ku**2 + kv**2
    kz0 = np.sqrt(np.clip(K**2 - k2, 1e-12, None))
    c = lambda x: x[:, None]
    dn = 1j * c(kz0) * E
    dn0 = dn.copy()
    if order >= 1:
        Uu, Uv = Eu - 1j * c(ku) * E, Ev - 1j * c(kv) * E
        dn = dn - (c(ku) * Uu + c(kv) * Uv) / c(kz0)
    if order >= 2:
        kgradE = c(ku) * Eu + c(kv) * Ev
        lapU = Euu + Evv - 2j * kgradE - c(k2) * E
        kk2U = (c(ku**2) * Euu + 2 * c(ku * kv) * Euv + c(kv**2) * Evv) - 2j * c(k2) * kgradE - c(k2**2) * E
        dn = dn + 1j * lapU / (2 * c(kz0)) + 1j * kk2U / (2 * c(kz0)**3)
    if guard and order >= 1:                                       # expansion not valid where the correction is
        corr = np.linalg.norm(dn - dn0, axis=1)                      # not small against the leading term: fall back
        bad = corr > guard * np.linalg.norm(dn0, axis=1)             # to the local plane wave at those points
        dn[bad] = dn0[bad]
    if curvature and Rc is not None:
        csgn = np.sign(np.sum((P - center) * N, axis=1))          # +1: N points away from the centre (convex)
        dn = dn - c(csgn / Rc * (1 + k2 / (2 * kz0**2))) * E
    curl = np.cross(t1, Eu) + np.cross(t2, Ev) + np.cross(N, dn)
    return curl / (1j * K)
