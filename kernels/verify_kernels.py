"""
verify_kernels.py -- Fig. 1 of the paper.

Three independent checks of Sec. 3 and Sec. 4:
  (a) closed form  vs  direct angular-spectrum quadrature of Eq. (4)
  (b) Ghat = 4 pi k^2 Gbar                       (Proposition 2)
  (c) transformation-free global assembly vs Theta-rotated local form,
      for BOTH E and M                            (Proposition 1)

MATLAB mirror: ../matlab/verify_kernels.m
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import j0

from cbim_core import K, Ghat, Gbar, E_kernel, M_kernel, kernels_local

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'output'); os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'lines.linewidth': 1.2, 'figure.dpi': 200,
                     'savefig.bbox': 'tight', 'legend.frameon': False,
                     'legend.fontsize': 7})
C = ['#1f77b4', '#d62728', '#2ca02c']


# ----------------------------------------------------------------------
def E1_spectral(s, z, k=K, nprop=600, nev=6000):
    """Direct numerical evaluation of  E1 = 2 pi int_0^inf J0(rho s)
    e^{i|z| kz} rho drho.  The branch point at rho = k is handled by splitting
    the integral and substituting rho = k sin(eta) below the cut and
    rho = k cosh(eta) above it, which makes both integrands smooth."""
    # propagating part, rho = k sin(eta), eta in [0, pi/2]
    x, w = np.polynomial.legendre.leggauss(nprop)
    eta = 0.25 * np.pi * (x + 1)
    we = 0.25 * np.pi * w
    rho = k * np.sin(eta)
    Ip = np.sum(j0(rho * s) * np.exp(1j * z * k * np.cos(eta))
                * rho * (k * np.cos(eta)) * we)
    # evanescent part, rho = k cosh(u); truncate where e^{-zk sinh u} ~ 1e-22
    umax = np.arcsinh(50.0 / (z * k))
    x, w = np.polynomial.legendre.leggauss(nev)
    u = 0.5 * umax * (x + 1)
    wu = 0.5 * umax * w
    rho = k * np.cosh(u)
    Ie = np.sum(j0(rho * s) * np.exp(-z * k * np.sinh(u))
                * rho * (k * np.sinh(u)) * wu)
    return 2 * np.pi * (Ip + Ie)


def main():
    R = {}

    # --- (a) closed form vs spectral quadrature ------------------------
    ts = np.logspace(np.log10(0.05), 1.0, 18)
    thetas = [15.0, 45.0, 75.0]
    errs = {}
    for th in thetas:
        thr = np.radians(th)
        e = []
        for t in ts:
            xb, zb = t * np.sin(thr), t * np.cos(thr)
            num = E1_spectral(xb, zb)
            cf = kernels_local(xb, 0.0, zb)[0]
            e.append(abs(num - cf) / abs(cf))
        errs[th] = np.array(e)
    R['a_max_relerr'] = float(max(v.max() for v in errs.values()))

    # --- (b) Ghat = 4 pi k^2 Gbar --------------------------------------
    rng = np.random.default_rng(1)
    d = rng.normal(size=(400, 3))
    d = d / np.linalg.norm(d, axis=1)[:, None] * np.logspace(-1, 1.2, 400)[:, None]
    dy = (np.linalg.norm(Ghat(d) - 4 * np.pi * K**2 * Gbar(d), axis=(1, 2))
          / np.linalg.norm(Ghat(d), axis=(1, 2)))
    R['b_max_relerr'] = float(dy.max())

    # --- (c) global vs Theta-rotated -----------------------------------
    eE, eM = [], []
    for _ in range(400):
        e1 = rng.normal(size=3); e1 /= np.linalg.norm(e1)
        e3 = np.cross(e1, rng.normal(size=3)); e3 /= np.linalg.norm(e3)
        e2 = np.cross(e3, e1)
        dv = rng.normal(size=3) * rng.uniform(0.2, 5.0)
        xb, yb, zb = dv @ e1, dv @ e2, dv @ e3
        E1c, E3c, M1c, M2c, M3c = kernels_local(xb, yb, zb)
        Eloc = E1c * e1 + E3c * e3
        Mloc = M1c * e1 + M2c * e2 + M3c * e3
        Eglb = E_kernel(dv[None], e2[None], e3[None], signed=False)[0]
        Mglb = M_kernel(dv[None], e2[None], e3[None], signed=False)[0]
        eE.append(np.linalg.norm(Eloc - Eglb) / np.linalg.norm(Eloc))
        eM.append(np.linalg.norm(Mloc - Mglb) / np.linalg.norm(Mloc))
    R['c_max_relerr_E'] = float(np.max(eE))
    R['c_max_relerr_M'] = float(np.max(eM))

    # --- plot ----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.15))
    for i, th in enumerate(thetas):
        ax[0].loglog(ts, np.maximum(errs[th], 1e-17), 'o-', ms=2.5, color=C[i],
                     label=rf'$\theta={th:.0f}^\circ$')
    ax[0].set_xlabel(r'$t/\lambda$'); ax[0].set_ylabel('relative error')
    ax[0].set_title('(a) closed form vs. spectral quadrature', fontsize=8)
    ax[0].set_ylim(1e-17, 1e-9); ax[0].legend()

    ax[1].loglog(np.linalg.norm(d, axis=1), np.maximum(dy, 1e-18), '.', ms=2,
                 color=C[0])
    ax[1].set_xlabel(r'$t/\lambda$'); ax[1].set_ylabel('relative error')
    ax[1].set_title(r'(b) $\hat{G}=4\pi k^2\bar{\bar{G}}$', fontsize=8)
    ax[1].set_ylim(1e-18, 1e-9)

    ax[2].semilogy(np.maximum(eE, 1e-18), '.', ms=2, color=C[0], label=r'$\mathbf{E}$')
    ax[2].semilogy(np.maximum(eM, 1e-18), '.', ms=2, color=C[1], label=r'$\mathbf{M}$')
    ax[2].set_xlabel('random frame / separation')
    ax[2].set_ylabel('relative error')
    ax[2].set_title(r'(c) global vs. $\Theta$-rotated', fontsize=8)
    ax[2].set_ylim(1e-18, 1e-9); ax[2].legend()
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_kernels.png'); plt.close(fig)

    print(json.dumps(R, indent=2))
    return R


if __name__ == '__main__':
    main()
