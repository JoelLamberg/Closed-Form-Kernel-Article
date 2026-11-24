"""Windows wrapper around run_map.py (Joel's box, 2026-11-02): mp.Pool on Windows caps at ~61 workers, so the grid
is split into --nparts disjoint interleaved slices of the largest-first task list (balanced), each run by its own
instance into the SAME results/ dir. Identical numerics and file format as run_map.py (it reuses grid/n_src/work).
Resumable exactly like run_map.py: finished pt_iii_jjj.json files are skipped.

usage:  python run_part.py --part 0 --nparts 2 --workers 46   [all run_map.py grid options]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, sys, time
from multiprocessing import Pool
from run_map import grid, n_src, fname, work, HERE


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--part", type=int, required=True)
    p.add_argument("--nparts", type=int, required=True)
    p.add_argument("--ns", type=int, default=301)
    p.add_argument("--na", type=int, default=301)
    p.add_argument("--smax", type=float, default=0.95)
    p.add_argument("--smin", type=float, default=1e-4)
    p.add_argument("--sgrid", choices=["log", "lin"], default="log")
    p.add_argument("--amin", type=float, default=2.8)
    p.add_argument("--amax", type=float, default=60.0)
    p.add_argument("--ppw", type=float, default=7.0)
    p.add_argument("--nobs", type=int, default=49)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--floor", type=float, default=1e-8)
    p.add_argument("--workers", type=int, default=46)
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
    want = {k: v for k, v in cfg.items() if k != "out"}
    if not os.path.exists(man):                       # atomic: the parts start at the same moment
        tmp = man + ".%d.tmp" % os.getpid()
        json.dump(want, open(tmp, "w"), indent=1)
        os.replace(tmp, man)
    old = json.load(open(man))
    if {k: old.get(k) for k in want} != want:
        sys.exit(f"ERROR: {out} holds results for a different grid {old}. Use another --out.")

    a_vals, s_vals = grid(cfg)
    tasks = [(i, j, float(a), float(s), cfg) for i, a in enumerate(a_vals) for j, s in enumerate(s_vals)]
    tasks.sort(key=lambda t: -n_src(t[2], t[3], cfg["ppw"]))     # largest first (same order as run_map.py)
    mine = tasks[args.part::args.nparts]
    todo = [t for t in mine if not os.path.exists(fname(out, t[0], t[1]))]
    total_w = sum(n_src(t[2], t[3], cfg["ppw"]) for t in todo)
    print(f"part {args.part}/{args.nparts}: {len(todo)} of {len(mine)} points to do "
          f"({len(tasks)} in grid), {args.workers} workers, out={out}", flush=True)

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
