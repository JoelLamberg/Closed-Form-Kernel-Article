"""GPU runner (CuPy, complex128), map subset. Same cells and JSON files as the CPU runner, SMALLEST cells first, so the
two meet in the middle; both skip finished cells.
usage:  python run_map_gpu.py --check 5            (GPU vs CPU on the 5 cheapest cells; must print OK; writes nothing)
        python run_map_gpu.py --set coarse         (two slots on one card: --slot 0 --nslots 2 and --slot 1 --nslots 2)"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, sys, time
import numpy as np
import cupy as cp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import las_map_core
from cbim_sc_core import K
from grid_map import CFG, cell_tasks, cost, fname
OUT = os.path.join(HERE, "..", "results_map")
CHUNK = 25_000_000
_radiate_cpu = las_map_core.radiate_las


def radiate_las_gpu(Ro, P, N, w, nxE, nxH, nxH2, nxH3, chunk_elems=None):
    Ro, P, w, nxE, nxH, nxH2, nxH3 = (cp.asarray(x) for x in (Ro, P, w, nxE, nxH, nxH2, nxH3))
    m, Ns = Ro.shape[0], P.shape[0]
    Esc = cp.zeros((m, 3), cp.complex128); Edb = cp.zeros_like(Esc); Es2 = cp.zeros_like(Esc); Es3 = cp.zeros_like(Esc)
    step = max(1, CHUNK // Ns)
    wE = w[:, None] * nxE; wJs = [w[:, None] * x for x in (nxH, nxH2, nxH3)]
    for i0 in range(0, m, step):
        r = Ro[i0:i0 + step]
        d = r[:, None, :] - P[None, :, :]
        t = cp.sqrt(cp.einsum("ijk,ijk->ij", d, d)); kt = K * t; e = cp.exp(1j * kt)
        sM = (((1 - 1j * kt) * e / (4 * np.pi * t**3))[..., None] * cp.cross(wE[None, :, :], d)).sum(axis=1)
        g = e / (4 * np.pi * t); A = 1 + 1j / kt - 1 / kt**2; Bc = -1 - 3j / kt + 3 / kt**2
        out = []
        for wJ in wJs:
            dJ = cp.einsum("ijk,jk->ij", d, wJ) / t
            out.append(((1j * K * g)[..., None] * (A[..., None] * wJ[None] + (Bc * dJ / t)[..., None] * d)).sum(axis=1))
        Esc[i0:i0 + step] = sM + out[0]; Es2[i0:i0 + step] = sM + out[1]; Es3[i0:i0 + step] = sM + out[2]
        Edb[i0:i0 + step] = 2 * sM
    return tuple(cp.asnumpy(x) for x in (Esc, Edb, Es2, Es3))


def compute(a, s):
    global CHUNK
    for _ in range(3):
        try:
            return las_map_core.run_point_las(a, s, nobs=CFG["nobs"], tol=CFG["tol"], floor=CFG["floor"])
        except cp.cuda.memory.OutOfMemoryError:
            cp.get_default_memory_pool().free_all_blocks(); CHUNK //= 2
            print(f"GPU out of memory at a={a:.2f} s={s:.3f}: chunk -> {CHUNK:.1e}", flush=True)
    raise RuntimeError(f"GPU out of memory 3x at a={a} s={s}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="coarse", choices=("coarse", "fine"))
    p.add_argument("--chunk", type=float, default=CHUNK)
    p.add_argument("--slot", type=int, default=0); p.add_argument("--nslots", type=int, default=1)
    p.add_argument("--check", type=int, default=0)
    args = p.parse_args()
    CHUNK = int(args.chunk)
    las_map_core.radiate_las = radiate_las_gpu
    tasks = sorted(cell_tasks(args.set), key=cost)
    if args.check:
        worst = 0.0
        for i, j, a, s in [t for t in tasks if t[3] > 0][:args.check]:
            g = compute(a, s)
            las_map_core.radiate_las = _radiate_cpu
            c = las_map_core.run_point_las(a, s, nobs=CFG["nobs"], tol=CFG["tol"], floor=CFG["floor"])
            las_map_core.radiate_las = radiate_las_gpu
            keys = [k for k in c if k.startswith("err_") and c[k] > 1e-12]
            dev = max(abs(g[k] - c[k]) / abs(c[k]) for k in keys); worst = max(worst, dev)
            print(f"check a={a:.2f} s={s:.3f}: ppw {c['ppw_used']}/{g['ppw_used']} max rel diff {dev:.1e}", flush=True)
        print(f"GPU CHECK worst rel diff {worst:.1e} -> {'OK' if worst < 1e-4 else 'FAIL'}"); sys.exit(0 if worst < 1e-4 else 1)
    os.makedirs(OUT, exist_ok=True)
    mine = tasks[args.slot::args.nslots]
    print(f"GPU slot {args.slot}/{args.nslots}: {len(mine)} cells (set {args.set}), chunk {CHUNK:.1e}", flush=True)
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
                  f"las={r['err_sclas_sc']:.2e} {r['seconds']:.1f}s elapsed {(time.time()-t0)/3600:.2f} h", flush=True)
    print(f"GPU slot done: {n} cells", flush=True)
