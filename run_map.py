"""Heatmap run: CBIM vs Stratton-Chu error over (curvature a/RoC, size a/lambda).

Resumable: one JSON per map point in <out>/. Re-running skips finished points.
Works on Windows / Linux / macOS.

usage:
  python run_map.py --ns 201 --na 201 --nobs 64 --workers 24
  python run_map.py --ns 9 --na 9 --nobs 16 --out results_smoke      (quick smoke test)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, sys, time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))


def grid(cfg):
    """s = a/RoC. 'log': planar column s=0 plus ns-1 log-spaced values in [smin, smax]."""
    if cfg.get("sgrid", "log") == "log":
        s_vals = np.concatenate([[0.0], np.geomspace(cfg["smin"], cfg["smax"], cfg["ns"] - 1)])
    else:
        s_vals = np.linspace(0.0, cfg["smax"], cfg["ns"])
    a_vals = np.geomspace(cfg["amin"], cfg["amax"], cfg["na"])
    return a_vals, s_vals


def n_src(a, s, ppw):
    h = np.sqrt(1 - s**2) / ppw
    return np.pi * a**2 / h**2


def fname(out, i, j):
    return os.path.join(out, f"pt_{i:03d}_{j:03d}.json")


def work(task):
    from cbim_sc_core import run_point_adaptive
    i, j, a, s, cfg = task
    f = fname(cfg["out"], i, j)
    if os.path.exists(f):
        return None
    t0 = time.time()
    r = run_point_adaptive(a, s, nobs=cfg["nobs"], tol=cfg["tol"], floor=cfg["floor"])
    r.update(i=i, j=j, seconds=time.time() - t0)
    tmp = f + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(r, fh)
    os.replace(tmp, f)
    return r


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", type=int, default=301, help="number of curvature values a/RoC")
    p.add_argument("--na", type=int, default=301, help="number of size values a/lambda")
    p.add_argument("--smax", type=float, default=0.95)
    p.add_argument("--smin", type=float, default=1e-4, help="smallest non-zero a/RoC (log grid)")
    p.add_argument("--sgrid", choices=["log", "lin"], default="log")
    p.add_argument("--amin", type=float, default=2.8)
    p.add_argument("--amax", type=float, default=60.0)
    p.add_argument("--ppw", type=float, default=7.0,
                   help="cost-model sampling only (actual sampling is adaptive: 5,7,10,14,20,28)")
    p.add_argument("--nobs", type=int, default=49, help="observation grid nobs x nobs (odd -> Simpson)")
    p.add_argument("--tol", type=float, default=1e-3, help="adaptive tolerance, relative to the CBIM-SC error")
    p.add_argument("--floor", type=float, default=1e-8, help="errors below floor*|E| are not refined")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--out", default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse()
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    cfg = dict(ns=args.ns, na=args.na, smax=args.smax, smin=args.smin, sgrid=args.sgrid,
               amin=args.amin, amax=args.amax,
               ppw=args.ppw, nobs=args.nobs, tol=args.tol, floor=args.floor, out=out)
    os.makedirs(out, exist_ok=True)
    man = os.path.join(out, "config.json")
    if os.path.exists(man):
        old = json.load(open(man))
        same = {k: old.get(k) for k in cfg if k != "out"} == {k: cfg[k] for k in cfg if k != "out"}
        if not same:
            sys.exit(f"ERROR: {out} holds results for a different grid {old}. Use another --out.")
    else:
        json.dump({k: v for k, v in cfg.items() if k != "out"}, open(man, "w"), indent=1)

    a_vals, s_vals = grid(cfg)
    tasks = [(i, j, float(a), float(s), cfg) for i, a in enumerate(a_vals) for j, s in enumerate(s_vals)]
    tasks.sort(key=lambda t: -n_src(t[2], t[3], cfg["ppw"]))     # largest first
    todo = [t for t in tasks if not os.path.exists(fname(out, t[0], t[1]))]
    total_w = sum(n_src(t[2], t[3], cfg["ppw"]) for t in todo)
    print(f"{len(todo)} of {len(tasks)} points to do, {args.workers} workers, out={out}", flush=True)

    t0, done_w, done = time.time(), 0.0, 0
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(work, todo, chunksize=1):
            if r is None:
                continue
            done += 1
            done_w += n_src(r["a"], r["s"], cfg["ppw"])
            el = time.time() - t0
            eta = el * (total_w / done_w - 1) / 3600
            print(f"[{done}/{len(todo)}] a={r['a']:6.2f} s={r['s']:.2e} ppw={r['ppw_used']:2d} "
                  f"CBIM-SC={r['err_cbim_sc']:.3e} SC+LPWA={r['err_sclpwa_sc']:.2e} unc={r.get('est_rel_uncertainty', 0):.0e} "
                  f"SC-ex={r['err_sc_exact']:.0e} {r['seconds']:.0f}s "
                  f"elapsed {el/3600:5.2f} h  ETA {eta:5.1f} h", flush=True)
    print("done", flush=True)
