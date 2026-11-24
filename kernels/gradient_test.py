"""Gradient test of Sec. 3.1 (main text) and Supplement Sec. 4 (Gradient test).

A concave spherical cap (radius of curvature Rc = 20 lambda, aperture radius a = 14 lambda,
vertex at the origin, normals towards the target) is illuminated by an x-polarised plane wave along its axis. The phase on the cap is
optimised by Adam with the closed-form gradient so as to maximise |E|^2 at a target point
42 lambda from the vertex. The result is compared with the optimum I_opt from fixed-point
iteration of the vector problem and with the exact co-phase of the main text.
Lengths in wavelengths, k = 2 pi.
"""
import numpy as np

k = 2 * np.pi
Rc, a, f = 20.0, 14.0, 42.0
NTH, NPH = 44, 120          # Gauss-Legendre nodes in cos(theta), azimuthal nodes
LR, B1, B2, EPS, NIT = 0.2, 0.9, 0.999, 1e-8, 900


def cap():
    """Sample points, unit normals and area weights on the cap (vertex at the origin)."""
    thmax = np.arcsin(a / Rc)
    x, w = np.polynomial.legendre.leggauss(NTH)
    c = np.cos(thmax) + (1 - np.cos(thmax)) * (x + 1) / 2      # cos(theta) nodes
    wc = w * (1 - np.cos(thmax)) / 2
    ph = 2 * np.pi * np.arange(NPH) / NPH
    C, P = np.meshgrid(c, ph, indexing="ij")
    S = np.sqrt(1 - C**2)
    o = np.stack([Rc * S * np.cos(P), Rc * S * np.sin(P), Rc * (1 - C)], -1).reshape(-1, 3)
    n = np.stack([-S * np.cos(P), -S * np.sin(P), C], -1).reshape(-1, 3)   # towards +z
    wts = (Rc**2 * np.outer(wc, np.full(NPH, 2 * np.pi / NPH))).ravel()
    return o, n, wts


def contributions(o, n, wts, rf):
    """G_j: field at rf from each source of the illuminated cap (step 3 of Sec. 2.5)."""
    Et = np.array([1.0, 0, 0]) - n * n[:, :1]                   # tangential part of x
    Et = Et * np.exp(1j * k * o[:, 2:3])                        # plane wave along z
    d = rf - o
    t = np.linalg.norm(d, axis=1, keepdims=True)
    I = (1 - 1j * k * t) * np.exp(1j * k * t) / t**2
    return wts[:, None] / (2 * np.pi) * (I / t) * np.cross(np.cross(n, Et), d)


def intensity(G, phi):
    S = (G * np.exp(1j * phi)[:, None]).sum(0)
    return np.vdot(S, S).real, S


def optimum(G, phi, nit=2000):
    """Fixed-point iteration u = S/|S|, phi_j = -arg(G_j . u*)."""
    for _ in range(nit):
        _, S = intensity(G, phi)
        u = S / np.linalg.norm(S)
        phi = -np.angle(G @ u.conj())
    return intensity(G, phi)[0]


def adam(G, phi):
    m = np.zeros_like(phi); v = np.zeros_like(phi)
    for it in range(1, NIT + 1):
        _, S = intensity(G, phi)
        g = 2 * np.imag(np.exp(-1j * phi) * (G.conj() @ S))     # dF/dphi_j
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g**2
        phi = phi + LR * (m / (1 - B1**it)) / (np.sqrt(v / (1 - B2**it)) + EPS)
    return intensity(G, phi)[0]


if __name__ == "__main__":
    o, n, wts = cap()
    rho = np.hypot(o[:, 0], o[:, 1])
    phi0 = -k * (np.sqrt(rho**2 + f**2) - f)                    # flat thin-lens phase
    print(f"{len(o)} source points, sag {Rc - np.sqrt(Rc**2 - a**2):.2f} lambda")
    for x0 in (0.0, 2.0, 5.0, 9.0):
        rf = np.array([x0, 0.0, f])
        G = contributions(o, n, wts, rf)
        Iopt = optimum(G, phi0.copy())
        t = np.linalg.norm(rf - o, axis=1)
        phic = -np.angle((1 - 1j * k * t) * np.exp(1j * k * t)) - k * o[:, 2]
        Ic = intensity(G, phic)[0]
        I0 = intensity(G, phi0)[0]
        Ia = adam(G, phi0.copy())
        print(f"x0 = {x0:3.0f}: initial deficit {Iopt / I0:8.1f}x, "
              f"Adam |I/Iopt-1| = {abs(Ia / Iopt - 1):.1e}, "
              f"co-phase |Ic/Iopt-1| = {abs(Ic / Iopt - 1):.1e}")
