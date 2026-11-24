"""GPU runner (CuPy, complex128). Same cells, same JSON files in ../results/, SMALLEST cells first so it meets the CPU
runner in the middle; both skip finished cells (a collision computes one cell twice, identical result, atomic replace).
usage:  python run_cells_gpu.py                     (needs cupy; on the RTX PRO 5000 the map used chunk 2.5e7)
        python run_cells_gpu.py --check 5           (GPU vs CPU on 5 small cells, prints max rel diff, writes nothing)"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, sys, time
import numpy as np
import cupy as cp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import corner_core
from cbim_sc_core import K
from grid_cells import CFG, cell_tasks, cost, fname

OUT = os.path.join(HERE, "..", "results")
CHUNK = 25_000_000
_radiate_cpu = corner_core.radiate_plus


def radiate_plus_gpu(Ro, P, N, w, nxE, nxH, nxH2, chunk_elems=None):
    Ro, P, w, nxE, nxH, nxH2 = (cp.asarray(x) for x in (Ro, P, w, nxE, nxH, nxH2))
    m, Ns = Ro.shape[0], P.shape[0]
    Esc = cp.zeros((m, 3), cp.complex128); Edb = cp.zeros_like(Esc); Es2 = cp.zeros_like(Esc)
    step = max(1, CHUNK // Ns)
    wJ, wJ2, wE = w[:, None] * nxH, w[:, None] * nxH2, w[:, None] * nxE
    for i0 in range(0, m, step):
        r = Ro[i0:i0 + step]
        d = r[:, None, :] - P[None, :, :]
        t = cp.sqrt(cp.einsum("ijk,ijk->ij", d, d))
        kt = K * t
        e = cp.exp(1j * kt)
        EM = ((1 - 1j * kt) * e / (4 * np.pi * t**3))[..., None] * cp.cross(wE[None, :, :], d)
        g = e / (4 * np.pi * t)
        A = 1 + 1j / kt - 1 / kt**2
        Bc = -1 - 3j / kt + 3 / kt**2
        dJ = cp.einsum("ijk,jk->ij", d, wJ) / t
        EJ = (1j * K * g)[..., None] * (A[..., None] * wJ[None] + (Bc * dJ / t)[..., None] * d)
        dJ2 = cp.einsum("ijk,jk->ij", d, wJ2) / t
        EJ2 = (1j * K * g)[..., None] * (A[..., None] * wJ2[None] + (Bc * dJ2 / t)[..., None] * d)
        sM = EM.sum(axis=1)
        Esc[i0:i0 + step] = sM + EJ.sum(axis=1)
        Es2[i0:i0 + step] = sM + EJ2.sum(axis=1)
        Edb[i0:i0 + step] = 2 * sM
    return tuple(cp.asnumpy(x) for x in (Esc, Edb, Es2))


def compute(a, s):
    global CHUNK
    for _ in range(3):
        try:
            return corner_core.run_point_plus(a, s, nobs=CFG["nobs"], tol=CFG["tol"], floor=CFG["floor"])
        except cp.cuda.memory.OutOfMemoryError:
            cp.get_default_memory_pool().free_all_blocks(); CHUNK //= 2
            print(f"GPU out of memory at a={a:.2f} s={s:.3f}: chunk -> {CHUNK:.1e}", flush=True)
    raise RuntimeError(f"GPU out of memory 3x at a={a} s={s}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chunk", type=float, default=CHUNK)
    p.add_argument("--slot", type=int, default=0, help="GPU processes sharing the card: this one takes every nslots-th")
    p.add_argument("--nslots", type=int, default=1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--check", type=int, default=0)
    args = p.parse_args()
    CHUNK = int(args.chunk)
    corner_core.radiate_plus = radiate_plus_gpu          # run_point_plus looks radiate_plus up at call time
    tasks = sorted(cell_tasks(args.stride), key=cost)
    if args.check:
        worst = 0.0
        for i, j, a, s in tasks[:args.check]:
            g = compute(a, s)
            corner_core.radiate_plus = _radiate_cpu; c = corner_core.run_point_plus(a, s, nobs=CFG["nobs"], tol=CFG["tol"], floor=CFG["floor"])
            corner_core.radiate_plus = radiate_plus_gpu
            keys = [k for k in c if k.startswith("err_") and c[k] > 1e-12]
            dev = max(abs(g[k] - c[k]) / abs(c[k]) for k in keys); worst = max(worst, dev)
            print(f"check a={a:.2f} s={s:.3f}: ppw {c['ppw_used']}/{g['ppw_used']} max rel diff {dev:.1e}", flush=True)
        print(f"GPU CHECK worst rel diff {worst:.1e} -> {'OK' if worst < 1e-4 else 'FAIL'}"); sys.exit(0 if worst < 1e-4 else 1)
    os.makedirs(OUT, exist_ok=True)
    mine = tasks[args.slot::args.nslots]
    print(f"GPU slot {args.slot}/{args.nslots}: {len(mine)} cells, chunk {CHUNK:.1e}", flush=True)
    t0, n = time.time(), 0
    for i, j, a, s in mine:
        f = fname(OUT, i, j)
        if os.path.exists(f):
            continue
        t1 = time.time(); r = compute(a, s); r.update(i=i, j=j, seconds=time.time() - t1, device="gpu")
        with open(f + ".gpu.tmp", "w") as fh:
            json.dump(r, fh)
        os.replace(f + ".gpu.tmp", f)
        n += 1
        if n % 50 == 1 or r["seconds"] > 60:
            print(f"[gpu {n}] a={a:5.2f} s={s:.3f} ppw={r['ppw_used']:2d} dbl={r['err_dbl_sc']:.2e} lpwa={r['err_sclpwa_sc']:.2e} "
                  f"{r['seconds']:.1f}s elapsed {(time.time()-t0)/3600:.2f} h", flush=True)
    print(f"GPU slot done: {n} cells", flush=True)
