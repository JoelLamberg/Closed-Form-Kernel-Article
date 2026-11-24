"""Save focal-plane FIELDS (not just metrics) for a few points, for the image panels.
usage: python save_fields.py OUTDIR [--workers 12] [--ppw 20 28] [--nobs 121] [--half 4]
Writes OUTDIR/fields_<point>.npz with X, Y and E_sc, E_dbl, E_lpwa, E_two, E_exact (nobs x nobs x 3) for each ppw."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, json, sys, time
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import focus_core as fc

POINTS = [(0.6, 10.0, 0.0, False), (0.5, 10.0, 0.3, False), (0.6, 10.0, 0.7, True)]
G = {}


def _init(d):
    G.update(d)


def _rad(Ro):
    return fc.radiate_focus(Ro, G["P"], G["N"], G["w"], G["nxE"], G["nxH"], G["nxH2"], G["hTx"], G["hTy"], G["kappa"])


def _fld(P):
    return G["beam"].fields(P, want_grad=True)


def chunks(A, n):
    return [A[i:i + n] for i in range(0, A.shape[0], n)]


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("out"); p.add_argument("--workers", type=int, default=12)
    p.add_argument("--ppw", type=int, nargs="+", default=[20, 28]); p.add_argument("--nobs", type=int, default=121)
    p.add_argument("--half", type=float, default=4.0); args = p.parse_args()
    choice = json.load(open(os.path.join(HERE, "..", "reference", "focus_choice.json")))
    os.makedirs(args.out, exist_ok=True)
    for na, a, s, cc in POINTS:
        st, F = choice[f"{na}_{a}_{s}_{int(cc)}"]; assert st == "ok"
        zf = fc.Z_RIM + F; beam = fc.beam_for(a, s, cc, na, zf); w0 = 1 / (np.pi * na)
        u = np.linspace(-args.half * w0, args.half * w0, args.nobs); X, Y = np.meshgrid(u, u, indexing="ij")
        Ro = np.stack([X.ravel(), Y.ravel(), np.full(X.size, zf)], 1)
        save = dict(X=X, Y=Y, na=na, a=a, s=s, concave=cc, zf=zf, w0=w0, nwaves=beam.nwaves)
        save["E_exact"] = beam.fields(Ro)[0].reshape(args.nobs, args.nobs, 3)
        for ppw in args.ppw:
            t0 = time.time()
            P, N, w, zv, h, hTx, hTy, kappa = fc.cap_oriented(a, s, ppw, cc)
            with Pool(args.workers, initializer=_init, initargs=(dict(beam=beam),)) as pool:
                parts = pool.map(_fld, chunks(P, max(1, P.shape[0] // (4 * args.workers))))
            E = np.concatenate([q[0] for q in parts]); H = np.concatenate([q[1] for q in parts]); Gd = np.concatenate([q[2] for q in parts])
            H2 = fc.lpwa_H(P, N, E, Gd)
            d = dict(P=P, N=N, w=w, nxE=np.cross(N, E), nxH=np.cross(N, H), nxH2=np.cross(N, H2), hTx=hTx, hTy=hTy, kappa=kappa)
            with Pool(args.workers, initializer=_init, initargs=(d,)) as pool:
                parts = pool.map(_rad, chunks(Ro, max(1, Ro.shape[0] // (4 * args.workers))))
            for i, name in enumerate(("sc", "dbl", "lpwa", "two")):
                save[f"E_{name}_ppw{ppw}"] = np.concatenate([q[i] for q in parts]).reshape(args.nobs, args.nobs, 3)
            print(f"NA {na} a {a} s {s} {'cc' if cc else 'cv'} ppw {ppw}: Nsrc {P.shape[0]}, waves {beam.nwaves}, {time.time()-t0:.0f} s", flush=True)
        tag = f"na{na:.1f}_a{a:.0f}_s{s:.1f}_{'cc' if cc else 'cv'}"
        np.savez_compressed(os.path.join(args.out, f"fields_{tag}.npz"), **save)
    print("done", flush=True)
