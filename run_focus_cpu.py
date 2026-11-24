"""CPU runner, focused-beam sweep with LAS2c (resumable; one JSON per point in ../results_focus/; largest first).
usage:  python run_focus_cpu.py --subset a10 --workers 46 --slice 0 --nslices 2     (second window: --slice 1)
        python run_focus_cpu.py --subset all ...                                     (adds the a = 20 / 40 points)"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, json, sys, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from grid_focus import tasks, fpath, cost
from settings import CFG
OUT = os.path.join(HERE, "..", "results_focus")
CHOICE = json.load(open(os.path.join(HERE, "..", "reference", "focus_choice.json")))


def key(t):
    return f"{t[0]}_{t[1]}_{t[2]}_{int(t[3])}"


def work(t):
    from focus_las_core import run_point_focus_las
    from focus_core import Z_RIM
    f = fpath(OUT, t)
    if os.path.exists(f):
        return None
    st, F = CHOICE[key(t)]; t0 = time.time()
    r = (dict(na=t[0], a=t[1], s=t[2], concave=t[3], status=st) if st != "ok"
         else run_point_focus_las(t[1], t[2], t[0], t[3], zf=Z_RIM + F, **CFG))
    r.update(seconds=time.time() - t0, device="cpu")
    with open(f + ".cpu.tmp", "w") as fh:
        json.dump(r, fh)
    os.replace(f + ".cpu.tmp", f)
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="a10", choices=("a10", "all"))
    p.add_argument("--workers", type=int, default=min(60, max(1, (os.cpu_count() or 2) - 2)))
    p.add_argument("--slice", type=int, default=0); p.add_argument("--nslices", type=int, default=1)
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    T = [t for t in tasks() if args.subset == "all" or t[1] == 10.0]
    T = sorted(T, key=lambda t: -cost(t))[args.slice::args.nslices]
    todo = [t for t in T if not os.path.exists(fpath(OUT, t))]
    print(f"{len(todo)} of {len(T)} points to do (subset {args.subset}), {args.workers} workers", flush=True)
    t0, n = time.time(), 0
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(work, todo, chunksize=1):
            if r is None:
                continue
            n += 1
            msg = (f"dbl={r['err_dbl_sc']:.2e} lpwa={r['err_lpwa_sc']:.2e} las={r['err_las_sc']:.2e} SC-ex={r['err_sc_exact']:.0e} "
                   f"ppw={r['ppw_used']}" if r.get("status") == "ok" else r.get("status"))
            print(f"[{n}/{len(todo)}] NA={r['na']} a={r['a']} s={r['s']} {'cc' if r['concave'] else 'cv'}: {msg} "
                  f"{r['seconds']:.0f}s elapsed {(time.time()-t0)/3600:.2f} h", flush=True)
    print("done", flush=True)
