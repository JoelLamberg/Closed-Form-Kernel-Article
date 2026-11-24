"""GPU runner (CuPy, complex128) for the focused-beam sweep with LAS2c -- written on the PC (the handoff has a CPU runner
only; Joel 2026-11-17: "run it using GPU, our CPU is quite busy"). Same points, same JSON files in ../results_focus/ as
run_focus_cpu.py, same physics: focus_las_core.run_point_focus_las unchanged, with two functions swapped for CuPy ports:
  FocusedBeam.fields   -> fields_gpu   (verbatim from the verified focus-sweep GPU runner, GPU = CPU to 5.6e-10)
  radiate_focus_las    -> radiate_focus_las_gpu (line-for-line port of focus_las_core.radiate_focus_las)
las_H calls beam.fields for its finite-difference stencil, so it runs on the GPU too.
usage:  python run_focus_gpu.py --check 2                (GPU vs CPU, short ladder 5/7 as in validate_focus; writes nothing)
        python run_focus_gpu.py --subset a10             (largest points first)"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, sys, time
import numpy as np
import cupy as cp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import focus_core
import focus_las_core
from cbim_sc_core import K
from grid_focus import tasks, fpath, cost
from settings import CFG
OUT = os.path.join(HERE, "..", "results_focus")
CHOICE = json.load(open(os.path.join(HERE, "..", "reference", "focus_choice.json")))
CHUNK = 12_000_000
_cpu_fields, _cpu_radiate = focus_core.FocusedBeam.fields, focus_las_core.radiate_focus_las


def fields_gpu(self, P, want_grad=False, chunk=None):
    s, p, q, c = (cp.asarray(x) for x in (self.s, self.p, self.q, self.c))
    P = cp.asarray(P); n = P.shape[0]; rf = cp.asarray([0.0, 0.0, self.zf])
    E = cp.zeros((n, 3), cp.complex128); H = cp.zeros_like(E); G = cp.zeros((n, 3, 3), cp.complex128) if want_grad else None
    step = max(1, CHUNK // self.nwaves)
    for i0 in range(0, n, step):
        ph = cp.exp(1j * K * ((P[i0:i0 + step] - rf) @ s.T)) * c[None, :]
        E[i0:i0 + step] = ph @ p; H[i0:i0 + step] = ph @ q
        if want_grad:
            for j in range(3):
                G[i0:i0 + step, j, :] = 1j * K * ((ph * p[None, :, j]) @ s)
    return cp.asnumpy(E), cp.asnumpy(H), (cp.asnumpy(G) if want_grad else None)


def _frac_positive(f0, ga, gb):
    a = cp.maximum(cp.maximum(ga, gb), 1e-300); b = cp.maximum(cp.minimum(ga, gb), 1e-300); v = -f0
    lo, mid = -(a + b) / 2, -(a - b) / 2
    cdf = cp.where(v <= lo, 0.0, cp.where(v <= mid, (v - lo) ** 2 / (2 * a * b),
          cp.where(v <= -mid, (v + a / 2) / a, cp.where(v < -lo, 1 - (-lo - v) ** 2 / (2 * a * b), 1.0))))
    return 1.0 - cdf


def radiate_focus_las_gpu(Ro, P, N, w, nxE, nxH, nxH2, nxH3, hTx, hTy, kappa, chunk_elems=None):
    Ro, P, N, w, nxE, nxH, nxH2, nxH3, hTx, hTy = (cp.asarray(x) for x in (Ro, P, N, w, nxE, nxH, nxH2, nxH3, hTx, hTy))
    m, Ns = Ro.shape[0], P.shape[0]
    Esc, Edb, Es2, Etw, Es3 = (cp.zeros((m, 3), cp.complex128) for _ in range(5))
    step = max(1, CHUNK // Ns)
    wE = w[:, None] * nxE; wJs = [w[:, None] * x for x in (nxH, nxH2, nxH3)]
    curved = kappa != 0.0
    NhTx, NhTy = cp.einsum("jk,jk->j", N, hTx)[None], cp.einsum("jk,jk->j", N, hTy)[None]
    for i0 in range(0, m, step):
        r = Ro[i0:i0 + step]
        d = r[:, None, :] - P[None, :, :]
        t = cp.sqrt(cp.einsum("ijk,ijk->ij", d, d)); kt = K * t; e = cp.exp(1j * kt)
        EM = ((1 - 1j * kt) * e / (4 * np.pi * t**3))[..., None] * cp.cross(wE[None, :, :], d)
        g = e / (4 * np.pi * t); A = 1 + 1j / kt - 1 / kt**2; Bc = -1 - 3j / kt + 3 / kt**2
        EJ = []
        for wJ in wJs:
            dJ = cp.einsum("ijk,jk->ij", d, wJ) / t
            EJ.append(((1j * K * g)[..., None] * (A[..., None] * wJ[None] + (Bc * dJ / t)[..., None] * d)).sum(axis=1))
        f0 = cp.einsum("ijk,jk->ij", d, N)
        if curved:
            ga = cp.abs(kappa * cp.einsum("ijk,jk->ij", d, hTx) - NhTx)
            gb = cp.abs(kappa * cp.einsum("ijk,jk->ij", d, hTy) - NhTy)
            fp = _frac_positive(f0, ga, gb)
        else:
            fp = (f0 > 0).astype(float)
        sM = EM.sum(axis=1)
        Esc[i0:i0 + step] = sM + EJ[0]; Es2[i0:i0 + step] = sM + EJ[1]; Es3[i0:i0 + step] = sM + EJ[2]
        Edb[i0:i0 + step] = 2 * sM
        Etw[i0:i0 + step] = (2 * (2 * fp - 1)[..., None] * EM).sum(axis=1)
    return tuple(cp.asnumpy(x) for x in (Esc, Edb, Es2, Etw, Es3))


def use_gpu(on):
    focus_core.FocusedBeam.fields = fields_gpu if on else _cpu_fields
    focus_las_core.radiate_focus_las = radiate_focus_las_gpu if on else _cpu_radiate


def key(t):
    return f"{t[0]}_{t[1]}_{t[2]}_{int(t[3])}"


def compute(t, **kw):
    global CHUNK
    st, F = CHOICE[key(t)]
    if st != "ok":
        return dict(na=t[0], a=t[1], s=t[2], concave=t[3], status=st)
    for _ in range(3):
        try:
            return focus_las_core.run_point_focus_las(t[1], t[2], t[0], t[3], zf=focus_core.Z_RIM + F, **{**CFG, **kw})
        except cp.cuda.memory.OutOfMemoryError:
            cp.get_default_memory_pool().free_all_blocks(); CHUNK //= 2
            print(f"GPU out of memory: chunk -> {CHUNK:.1e}", flush=True)
    raise RuntimeError(f"GPU out of memory 3x at {t}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="a10", choices=("a10", "all"))
    p.add_argument("--chunk", type=float, default=CHUNK); p.add_argument("--check", type=int, default=0)
    p.add_argument("--slot", type=int, default=0); p.add_argument("--nslots", type=int, default=1)
    args = p.parse_args(); CHUNK = int(args.chunk)
    T = [t for t in tasks() if args.subset == "all" or t[1] == 10.0]
    if args.check:                                   # validate_focus's two points first, then curved/concave ones
        pts = [(0.4, 10.0, 0.0, False), (0.5, 10.0, 0.3, False), (0.3, 10.0, 0.7, True), (0.7, 10.0, 0.5, True)]
        pts = [t for t in pts if CHOICE[key(t)][0] == "ok"][:args.check]
        worst = 0.0
        for t in pts:
            use_gpu(True); g = compute(t, ladder=(5, 7)); use_gpu(False); c = compute(t, ladder=(5, 7))
            ks = [k for k in c if (k.startswith("err_") or k.startswith("surfH_")) and isinstance(c[k], float) and c[k] > 1e-12]
            dev = max(abs(g[k] - c[k]) / abs(c[k]) for k in ks); worst = max(worst, dev)
            print(f"check {t}: ppw {c['ppw_used']}/{g['ppw_used']} max rel diff {dev:.1e} over {len(ks)} metrics "
                  f"(las {g['err_las_sc']:.3e} / {c['err_las_sc']:.3e})", flush=True)
        print(f"GPU FOCUS CHECK worst rel diff {worst:.1e} -> {'OK' if worst < 1e-6 else 'FAIL'}"); sys.exit(0 if worst < 1e-6 else 1)
    use_gpu(True); os.makedirs(OUT, exist_ok=True)
    mine = sorted(T, key=lambda t: -cost(t))[args.slot::args.nslots]
    print(f"GPU focus slot {args.slot}/{args.nslots}: {len(mine)} points (subset {args.subset}), chunk {CHUNK:.1e}, {CFG}", flush=True)
    t0, n = time.time(), 0
    for t in mine:
        f = fpath(OUT, t)
        if os.path.exists(f):
            continue
        t1 = time.time(); r = compute(t); r.update(seconds=time.time() - t1, device="gpu")
        with open(f + ".gpu.tmp", "w") as fh:
            json.dump(r, fh)
        os.replace(f + ".gpu.tmp", f); n += 1
        msg = (f"dbl={r['err_dbl_sc']:.2e} lpwa={r['err_lpwa_sc']:.2e} las={r['err_las_sc']:.2e} SC-ex={r['err_sc_exact']:.0e} "
               f"ppw={r['ppw_used']}" if r.get("status") == "ok" else r.get("status"))
        print(f"[gpu {n}] NA={t[0]} a={t[1]} s={t[2]} {'cc' if t[3] else 'cv'}: {msg} {r['seconds']:.0f}s "
              f"elapsed {(time.time()-t0)/3600:.2f} h", flush=True)
    print(f"GPU focus slot done: {n} points", flush=True)
