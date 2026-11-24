"""CPU runner, map subset (resumable; one JSON per cell in ../results_map/; largest cells first).
usage:  python run_map_cpu.py --set coarse --workers 46 --slice 0 --nslices 2      (second window: --slice 1)
        python run_map_cpu.py --set fine   ...                                     (afterwards; skips finished cells)"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, json, sys, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from grid_map import CFG, cell_tasks, cost, fname
OUT = os.path.join(HERE, "..", "results_map")


def work(t):
    from las_map_core import run_point_las
    i, j, a, s = t
    f = fname(OUT, i, j)
    if os.path.exists(f):
        return None
    t0 = time.time()
    r = run_point_las(a, s, nobs=CFG["nobs"], tol=CFG["tol"], floor=CFG["floor"])
    r.update(i=i, j=j, seconds=time.time() - t0, device="cpu")
    with open(f + ".cpu.tmp", "w") as fh:
        json.dump(r, fh)
    os.replace(f + ".cpu.tmp", f)
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="coarse", choices=("coarse", "fine"))
    p.add_argument("--workers", type=int, default=min(60, max(1, (os.cpu_count() or 2) - 2)))
    p.add_argument("--slice", type=int, default=0); p.add_argument("--nslices", type=int, default=1)
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tasks = sorted(cell_tasks(args.set), key=lambda t: -cost(t))[args.slice::args.nslices]
    todo = [t for t in tasks if not os.path.exists(fname(OUT, t[0], t[1]))]
    tot = sum(cost(t) for t in todo) or 1.0
    print(f"{len(todo)} of {len(tasks)} cells to do (set {args.set}, slice {args.slice}/{args.nslices}), {args.workers} workers", flush=True)
    t0, done_c, n = time.time(), 0.0, 0
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(work, todo, chunksize=1):
            if r is None:
                continue
            n += 1; done_c += cost((0, 0, r["a"], r["s"])); el = time.time() - t0
            print(f"[{n}/{len(todo)}] a={r['a']:5.2f} s={r['s']:.3f} ppw={r['ppw_used']:2d} dbl={r['err_dbl_sc']:.2e} "
                  f"lpwa={r['err_sclpwa_sc']:.2e} las={r['err_sclas_sc']:.2e} SC-ex={r['err_sc_exact']:.0e} {r['seconds']:.0f}s "
                  f"elapsed {el/3600:.2f} h ETA {el*(tot/done_c-1)/3600:.1f} h", flush=True)
    print("done", flush=True)
