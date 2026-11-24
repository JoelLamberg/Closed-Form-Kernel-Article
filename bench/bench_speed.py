"""
bench_speed.py -- Fig. 3 of the paper.

Closed-form kernel versus the angular-spectrum quadrature it replaces, at equal
accuracy, both vectorised on one core.  The reference point is the near-field
map of Opt. Express 32, 28231 (2024), Sec. 5.1: 1800 source points to 500x500
observation points, reported there as approximately four hours.

The quadrature baseline must be a *competent* implementation, or the comparison
is against a straw man.  Two schemes for the same radial integral over
rho in [0, 8k]:

  naive   plain Gauss-Legendre across the whole range.  The integrand has a
          square-root kink at the branch point rho = k, where kz = sqrt(k^2-rho^2)
          turns imaginary, so GL converges only ALGEBRAICALLY: measured
          err ~ nq^(-1.63), and nq = 200 gives 8.6e-1 -- 86 % wrong.  An earlier
          version of this script used exactly that as the baseline, which made
          the reported speedup meaningless.  Retained below only as a documented
          negative control.

  split   substitute rho = k sin(eta) below the branch point and rho = k cosh(u)
          above it (the scheme verify_kernels.py already uses).  Each piece is
          smooth, so GL converges EXPONENTIALLY: 75 nodes per branch = 150 total
          reaches 1.8e-14 and stops improving.  This is the honest baseline.

accuracy_check() measures both against a converged reference and prints them, so
the "equal accuracy" claim in the caption is verified at run time rather than
asserted.

MATLAB mirror: ../matlab/bench_speed.m
"""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import j0

from cbim_core import K, E_kernel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_bench'); os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'lines.linewidth': 1.2, 'figure.dpi': 200,
                     'savefig.bbox': 'tight', 'legend.frameon': False,
                     'legend.fontsize': 7})
C = ['#1f77b4', '#d62728']
W_OE = 1800 * 250000                    # the published benchmark
H_OE = 4.0                              # hours reported in Ref. [2]


def t_closed(Ns, Np, rng, reps=3):
    O = rng.normal(size=(Ns, 3))
    e2 = np.tile([0, 1.0, 0], (Ns, 1))
    e3 = np.tile([0, 0, 1.0], (Ns, 1))
    tg = rng.normal(size=(Np, 3)) * 3 + np.array([0, 0, 12.0])
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        acc = np.zeros((Np, 3), complex)
        ch = max(1, int(2.5e7 // Np))
        for i in range(0, Ns, ch):
            d = tg[None, :, :] - O[i:i + ch, None, :]
            acc += E_kernel(d,
                            np.broadcast_to(e2[i:i + ch, None, :], d.shape),
                            np.broadcast_to(e3[i:i + ch, None, :], d.shape)).sum(0)
        best = min(best, time.perf_counter() - t0)
    return best


HI = 8.0          # upper limit of the radial integral, in units of k
NB = 75           # nodes PER BRANCH for the split scheme (150 total) -> 1.8e-14


def nodes_split(nb=NB, hi=HI):
    """Branch-split radial nodes: rho = k sin(eta) below k, k cosh(u) above.
    Returns (rho, weight, kz) with kz complex (imaginary on the evanescent arm)."""
    x, w = np.polynomial.legendre.leggauss(nb)
    eta = 0.25 * np.pi * (x + 1)
    we = 0.25 * np.pi * w
    rho_p = K * np.sin(eta)
    w_p = rho_p * K * np.cos(eta) * we
    kz_p = K * np.cos(eta) + 0j
    umax = np.arccosh(hi)
    u = 0.5 * (x + 1) * umax
    wu = 0.5 * umax * w
    rho_e = K * np.cosh(u)
    w_e = rho_e * K * np.sinh(u) * wu
    kz_e = 1j * K * np.sinh(u)
    return (np.concatenate([rho_p, rho_e]),
            np.concatenate([w_p, w_e]),
            np.concatenate([kz_p, kz_e]))


def nodes_naive(nq=200, hi=HI):
    """Plain GL across the branch point -- the negative control (86 % error)."""
    xg, wg = np.polynomial.legendre.leggauss(nq)
    rho = 0.5 * (xg + 1) * (hi * K)
    return rho, 0.5 * (hi * K) * wg * rho, np.sqrt(K * K - rho**2 + 0j)


def radial(s, z, rho, wq, kz):
    """The radial angular-spectrum integral, vectorised over (s, z)."""
    return (j0(np.outer(s, rho)) * np.exp(1j * np.outer(z, kz)) * wq).sum(-1)


def accuracy_check(rng, npts=120):
    """Verify the equal-accuracy premise of Fig. 3 instead of assuming it."""
    d = rng.normal(size=(npts, 3)) * 3 + np.array([0, 0, 12.0])
    s = np.linalg.norm(d[:, :2], axis=1)
    z = np.abs(d[:, 2])
    ref = radial(s, z, *nodes_split(600))
    err = lambda v: float(np.linalg.norm(v - ref) / np.linalg.norm(ref))
    out = {'split_nodes': 2 * NB,
           'err_split': err(radial(s, z, *nodes_split(NB))),
           'err_naive_200': err(radial(s, z, *nodes_naive(200)))}
    print(f"accuracy of the radial integral vs a converged reference:")
    print(f"   branch-split, {2*NB:4d} nodes : {out['err_split']:.2e}   <- baseline used")
    print(f"   naive GL,      200 nodes : {out['err_naive_200']:.2e}   (negative control)")
    return out


def t_quad(Ns, Np, rng, reps=2):
    """Same kernel by branch-split Gauss-Legendre quadrature of the radial
    angular spectrum -- the cost model of the original implementation, at an
    accuracy matched to the closed form (see accuracy_check)."""
    rho, wq, kz = nodes_split()
    O = rng.normal(size=(Ns, 3))
    tg = rng.normal(size=(Np, 3)) * 3 + np.array([0, 0, 12.0])
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        ch = max(1, int(3e5 // Np))
        for i in range(0, Ns, ch):
            d = tg[None, :, :] - O[i:i + ch, None, :]
            s = np.linalg.norm(d[..., :2], axis=-1)
            z = np.abs(d[..., 2])
            (j0(s[..., None] * rho) * np.exp(1j * z[..., None] * kz)
             * wq).sum(-1).sum()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(2)
    acc = accuracy_check(rng)
    print()
    sizes = [(400, 400), (800, 800), (1200, 1500), (1800, 3000), (1800, 8000)]
    work, tc, tq = [], [], []
    for Ns, Np in sizes:
        work.append(Ns * Np)
        tc.append(t_closed(Ns, Np, rng))
        tq.append(t_quad(Ns, Np, rng))
        print(f'Ns={Ns:5d} Np={Np:6d}: closed {tc[-1]:7.3f} s   quad {tq[-1]:8.3f} s')
    work = np.array(work); tc = np.array(tc); tq = np.array(tq)
    ns_c = tc[-1] / work[-1] * 1e9
    ns_q = tq[-1] / work[-1] * 1e9
    R = dict(ns_per_pair_closed=float(ns_c), ns_per_pair_quad=float(ns_q),
             speedup_same_implementation=float(ns_q / ns_c),
             t_closed_OE_case_s=float(ns_c * 1e-9 * W_OE),
             OE_reported_hours=H_OE,
             speedup_vs_published=float(H_OE * 3600 / (ns_c * 1e-9 * W_OE)),
             quad_nodes=2 * NB, **acc)

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.loglog(work, tq, 's-', color=C[1], ms=4, label='angular-spectrum quadrature')
    ax.loglog(work, tc, 'o-', color=C[0], ms=4, label='closed form (this work)')
    xe = np.array([work[0], W_OE])
    ax.loglog(xe, ns_q * 1e-9 * xe, '--', color=C[1], lw=0.8, alpha=0.6)
    ax.loglog(xe, ns_c * 1e-9 * xe, '--', color=C[0], lw=0.8, alpha=0.6)
    ax.plot([W_OE], [H_OE * 3600], '*', color='k', ms=11, zorder=5)
    ax.annotate('Ref. [2] Sec. 5.1\nreported, 4 h', (W_OE, H_OE * 3600),
                textcoords='offset points', xytext=(-8, -30), ha='right', fontsize=6.5)
    ax.plot([W_OE], [R['t_closed_OE_case_s']], '*', color=C[0], ms=11, zorder=5)
    ax.annotate(f"{R['t_closed_OE_case_s']:.0f} s", (W_OE, R['t_closed_OE_case_s']),
                textcoords='offset points', xytext=(-6, 6), ha='right',
                fontsize=6.5, color=C[0])
    ax.set_xlabel(r'source points $\times$ evaluation points')
    ax.set_ylabel('wall-clock time (s)')
    ax.legend(loc='upper left')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_speed.png'); plt.close(fig)
    print(json.dumps(R, indent=2))
    return R


if __name__ == '__main__':
    main()
