"""GPU worker for the same map (Joel 2026-11-05: 'use CPU and GPU'). Runs cbim_sc_core.run_point_adaptive UNCHANGED,
with radiate() swapped for a line-for-line CuPy port (complex128; agrees with the CPU to ~1e-14). Writes the same
pt_iii_jjj.json files (+ "device": "gpu") into the same results/ dir. Order: SMALLEST cost first, i.e. the opposite end
from run_part.py (largest first), so CPU and GPU only meet in the middle; each side skips finished points, so a
collision just computes one point twice (identical result, atomic os.replace). Exits when every point exists.

usage:  python run_gpu.py                      (full run)
        python run_gpu.py --verify N           (recompute N CPU-finished points into results_gpuverify/ and compare)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, datetime, json, msvcrt, sys, time
import numpy as np
import cupy as cp
import cbim_sc_core
from cbim_sc_core import K
from run_map import grid, n_src, fname, HERE

CHUNK = 25_000_000            # pair elements per GPU chunk: 4e7 peaked at 38 GB of the 48 GB card


def _frac_positive(f0, ga, gb):
    a = cp.maximum(cp.maximum(ga, gb), 1e-300); b = cp.maximum(cp.minimum(ga, gb), 1e-300)
    v = -f0
    lo, mid = -(a + b) / 2, -(a - b) / 2
    cdf = cp.where(v <= lo, 0.0,
          cp.where(v <= mid, (v - lo) ** 2 / (2 * a * b),
          cp.where(v <= -mid, (v + a / 2) / a,
          cp.where(v < -lo, 1 - (-lo - v) ** 2 / (2 * a * b), 1.0))))
    return 1.0 - cdf


def radiate_gpu(Ro, P, N, w, nxE, nxH, hTx=None, hTy=None, kappa=0.0, chunk_elems=None, nxH2=None):
    """cbim_sc_core.radiate() on the GPU; same signature and return values (numpy)."""
    Ro, P, N, w, nxE, nxH = (cp.asarray(x) for x in (Ro, P, N, w, nxE, nxH))
    m, Ns = Ro.shape[0], P.shape[0]
    Esc = cp.zeros((m, 3), cp.complex128); Ecb = cp.zeros_like(Esc); Ecp = cp.zeros_like(Esc); Es2 = cp.zeros_like(Esc)
    step = max(1, CHUNK // Ns)
    wJ, wE = w[:, None] * nxH, w[:, None] * nxE
    wJ2 = None if nxH2 is None else w[:, None] * cp.asarray(nxH2)
    curved = hTx is not None and kappa != 0.0
    if curved:
        hTx, hTy = cp.asarray(hTx), cp.asarray(hTy)
        NhTx, NhTy = cp.einsum("jk,jk->j", N, hTx)[None], cp.einsum("jk,jk->j", N, hTy)[None]
    for i0 in range(0, m, step):
        r = Ro[i0:i0 + step]
        d = r[:, None, :] - P[None, :, :]
        t = cp.sqrt(cp.einsum("ijk,ijk->ij", d, d))
        kt = K * t
        e = cp.exp(1j * kt)
        cM = (1 - 1j * kt) * e / (4 * np.pi * t**3)
        EM = cM[..., None] * cp.cross(wE[None, :, :], d)
        g = e / (4 * np.pi * t)
        A = 1 + 1j / kt - 1 / kt**2
        Bc = -1 - 3j / kt + 3 / kt**2
        dJ = cp.einsum("ijk,jk->ij", d, wJ) / t
        EJ = (1j * K * g)[..., None] * (A[..., None] * wJ[None] + (Bc * dJ / t)[..., None] * d)
        f0 = cp.einsum("ijk,jk->ij", d, N)
        if curved:
            ga = cp.abs(kappa * cp.einsum("ijk,jk->ij", d, hTx) - NhTx)
            gb = cp.abs(kappa * cp.einsum("ijk,jk->ij", d, hTy) - NhTy)
            fp = _frac_positive(f0, ga, gb)
        else:
            fp = (f0 > 0).astype(float)
        Esc[i0:i0 + step] = (EM + EJ).sum(axis=1)
        if wJ2 is not None:
            dJ2 = cp.einsum("ijk,jk->ij", d, wJ2) / t
            EJ2 = (1j * K * g)[..., None] * (A[..., None] * wJ2[None] + (Bc * dJ2 / t)[..., None] * d)
            Es2[i0:i0 + step] = (EM + EJ2).sum(axis=1)
        Ecb[i0:i0 + step] = (2 * (2 * fp - 1)[..., None] * EM).sum(axis=1)
        Ecp[i0:i0 + step] = (2 * fp[..., None] * EM).sum(axis=1)
    out = [cp.asnumpy(x) for x in (Esc, Ecb, Ecp, Es2)]
    return out if nxH2 is not None else out[:3]


cbim_sc_core.radiate = radiate_gpu        # run_point_adaptive looks radiate up in the module at call time


def log(msg):
    line = "%s  %s" % (datetime.datetime.now().strftime("%m-%d %H:%M:%S"), msg)
    print(line, flush=True)


SLOT = 0


def status(phase, busy=True):
    st = r"C:\Temp\PIPELINE_STATUS\cbim_sc_gpu_slot%d.json" % SLOT
    d = dict(pipeline="cbim_sc_gpu", purpose="GPU worker for 2026-11-02_CBIM_vs_SC_heatmap (CuPy)",
             state="running" if busy else "done", phase=phase,
             heartbeat_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             stale_after_s=3600, resources_held=dict(fdtd_gpu=busy, approx_cpu_threads=1 if busy else 0, mode_licenses=0),
             safe_for_heavy_cpu=True, safe_for_gpu=not busy, log=os.path.join(HERE, "job_gpu%d.log" % SLOT))
    try:
        json.dump(d, open(st + ".tmp", "w"), indent=2); os.replace(st + ".tmp", st)
    except OSError:
        pass


def compute(i, j, a, s, cfg, out):
    global CHUNK
    t0 = time.time()
    for attempt in range(3):                       # out of GPU memory -> drop the cache, halve the chunk, retry
        try:
            r = cbim_sc_core.run_point_adaptive(a, s, nobs=cfg["nobs"], tol=cfg["tol"], floor=cfg["floor"])
            break
        except cp.cuda.memory.OutOfMemoryError:
            cp.get_default_memory_pool().free_all_blocks()
            CHUNK //= 2
            log(f"GPU out of memory at a={a:.2f} s={s:.2e}: chunk -> {CHUNK:.1e}, retrying")
    else:
        raise RuntimeError(f"GPU out of memory 3x at a={a} s={s}")
    r.update(i=i, j=j, seconds=time.time() - t0, device="gpu")
    f = fname(out, i, j)
    with open(f + ".gpu.tmp", "w") as fh:
        json.dump(r, fh)
    os.replace(f + ".gpu.tmp", f)
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", type=int, default=0)
    p.add_argument("--slot", type=int, default=0, help="this GPU process takes every nslots-th task, offset slot")
    p.add_argument("--nslots", type=int, default=1, help="GPU processes sharing the card (small points are host-bound)")
    p.add_argument("--chunk", type=float, default=CHUNK, help="pair elements per GPU chunk (lower it when nslots > 1)")
    p.add_argument("--pool-gb", type=float, default=0, help="cap this process's CuPy memory pool (0 = no cap)")
    args = p.parse_args()
    CHUNK, SLOT = int(args.chunk), args.slot           # module globals read by radiate_gpu() / status()
    if args.pool_gb:
        cp.get_default_memory_pool().set_limit(size=int(args.pool_gb * 2**30))
    res = os.path.join(HERE, "results")
    cfg = json.load(open(os.path.join(res, "config.json")))
    a_vals, s_vals = grid(cfg)
    tasks = [(i, j, float(a), float(s)) for i, a in enumerate(a_vals) for j, s in enumerate(s_vals)]

    if args.verify:                        # recompute the smallest CPU-finished points and compare every metric
        out = os.path.join(HERE, "results_gpuverify"); os.makedirs(out, exist_ok=True)
        done = sorted((t for t in tasks if os.path.exists(fname(res, t[0], t[1]))), key=lambda t: n_src(t[2], t[3], 7))
        pick = done[:args.verify // 2] + done[len(done) // 2:len(done) // 2 + args.verify - args.verify // 2]
        worst = 0.0
        for i, j, a, s in pick:
            g = compute(i, j, a, s, cfg, out)
            c = json.load(open(fname(res, i, j)))
            keys = [k for k in c if k.startswith("err_") and isinstance(c[k], float) and c[k] > 1e-12]
            dev = max(abs(g[k] - c[k]) / abs(c[k]) for k in keys)
            worst = max(worst, dev)
            log(f"verify a={a:6.2f} s={s:.2e}: ppw cpu/gpu {c['ppw_used']}/{g['ppw_used']}  levels {c['levels']}/{g['levels']}  "
                f"max rel diff over {len(keys)} err_* metrics {dev:.1e}  time cpu {c['seconds']:.0f} s / gpu {g['seconds']:.1f} s")
        log(f"VERIFY worst rel diff {worst:.1e}")
        sys.exit(0 if worst < 1e-4 else 1)        # CPU/GPU libm rounding: fields agree ~1e-14|E|, tiny metrics ~1e-5 rel

    lock = open(os.path.join(HERE, "run_gpu_slot%d.lock" % args.slot), "w")
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        sys.exit("another run_gpu.py is running")
    tasks.sort(key=lambda t: n_src(t[2], t[3], cfg["ppw"]))         # smallest first (CPU parts go largest first)
    mine = tasks[args.slot::args.nslots]
    log(f"GPU slot {args.slot}/{args.nslots} pid {os.getpid()} on {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}, chunk {CHUNK:.1e}")
    done_here, t_start, last_hb = 0, time.time(), 0.0
    for i, j, a, s in mine:
        if os.path.exists(fname(res, i, j)):
            continue
        r = compute(i, j, a, s, cfg, res)
        done_here += 1
        if done_here % 50 == 1 or r["seconds"] > 60:
            n_all = sum(os.path.exists(fname(res, t[0], t[1])) for t in tasks) if done_here % 500 == 1 else -1
            log(f"[gpu {done_here}] a={a:6.2f} s={s:.2e} ppw={r['ppw_used']:2d} lv={r['levels']} CBIM-SC={r['err_cbim_sc']:.3e} "
                f"SC+LPWA={r['err_sclpwa_sc']:.2e} SC-ex={r['err_sc_exact']:.0e} {r['seconds']:.1f}s  "
                f"elapsed {(time.time() - t_start) / 3600:.2f} h" + (f"  map {n_all}/{len(tasks)}" if n_all >= 0 else ""))
        if time.time() - last_hb > 60:
            status(f"{done_here} points done on GPU, now a={a:.2f} s={s:.2e}"); last_hb = time.time()
    status(f"finished: {done_here} points on GPU", busy=False)
    log(f"GPU worker done: {done_here} points; no unfinished points left")
