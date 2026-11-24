"""Pre-select the focal distance of every grid point (writes ../reference/focus_choice.json). Already done; rerun only
if focus_core.choose_focus or the grid changes.  usage: python precompute_focus.py --workers 12"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, json, sys, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from grid_focus import tasks
from settings import CFG


def one(t):
    from focus_core import choose_focus, Z_RIM
    na, a, s, c = t; zf, st = choose_focus(a, s, c, na, CFG["rim_target"])
    return f"{na}_{a}_{s}_{int(c)}", (st, zf - Z_RIM)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=12); args = p.parse_args()
    t0 = time.time()
    with Pool(args.workers) as pool:
        out = dict(pool.map(one, tasks(), chunksize=1))
    json.dump(out, open(os.path.join(HERE, "..", "reference", "focus_choice.json"), "w"), indent=0)
    from collections import Counter
    print(Counter(v[0] for v in out.values()), f"{time.time()-t0:.0f}s")
    for na in sorted({float(k.split('_')[0]) for k in out}):
        As = sorted({float(k.split('_')[1]) for k in out if float(k.split('_')[0]) == na})
        print(f"NA {na}: " + " | ".join(f"a={a:.0f} ok {sum(v[0]=='ok' for k, v in out.items() if k.startswith(f'{na}_{a}_'))}/13"
                                          for a in As))
