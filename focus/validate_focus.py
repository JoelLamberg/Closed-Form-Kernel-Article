"""Run first on the big machine (CPU, a few minutes). Checks: (1) the focused beam is an exact Maxwell field and its
analytic gradient is right; (2) on a plane, Smythe (one-sided CBIM) equals Stratton-Chu up to the rim-truncation floor and
Stratton-Chu reproduces the exact beam; (3) one curved point runs end to end. Writes nothing into ../results/."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from focus_core import FocusedBeam, K, run_point_focus, Z_RIM
from settings import CFG
CHOICE = json.load(open(os.path.join(HERE, "..", "reference", "focus_choice.json")))
ok = True
rng = np.random.default_rng(3)
for na in (0.2, 0.5, 0.7):
    beam = FocusedBeam(na, 10.0, 15.0); P = rng.uniform([-3, -3, 2], [3, 3, 9], (6, 3)); h = 1e-5
    E, H, G = beam.fields(P, want_grad=True); curl = np.zeros_like(E); div = np.zeros(len(P), complex); Gfd = np.zeros_like(G)
    for a in range(3):
        e = np.zeros(3); e[a] = h; dE = (beam.fields(P + e)[0] - beam.fields(P - e)[0]) / (2 * h); Gfd[:, :, a] = dE
        div += dE[:, a]; b, c = (a + 1) % 3, (a + 2) % 3; curl[:, c] += dE[:, b]; curl[:, b] -= dE[:, c]
    ec = np.linalg.norm(curl - 1j * K * H) / np.linalg.norm(K * H); ed = np.linalg.norm(div) / np.linalg.norm(K * E)
    eg = np.linalg.norm(G - Gfd) / np.linalg.norm(G); good = max(ec, ed, eg) < 1e-7; ok &= good
    print(f"beam NA {na}: curl {ec:.1e} div {ed:.1e} grad {eg:.1e} -> {'OK' if good else 'FAIL'}", flush=True)
cands = [k for k, v in CHOICE.items() if v[0] == "ok"]
def pick(cond):
    c = [k for k in cands if cond(*k.split("_"))]
    return min(c, key=lambda k: float(k.split("_")[1])) if c else None
for k in [pick(lambda na, a, s, c: float(s) == 0.0), pick(lambda na, a, s, c: float(s) >= 0.3 and c == "1")]:
    if k is None: continue
    na, a, s, c = k.split("_"); F = CHOICE[k][1]
    r = run_point_focus(float(a), float(s), float(na), c == "1", zf=Z_RIM + F, **CFG)
    print(f"point NA={na} a={a} s={s} {'cc' if c=='1' else 'cv'}: ppw {r['ppw_used']} levels {r['levels']} | SC vs exact {r['err_sc_exact']:.1e} | "
          f"dbl {r['err_dbl_sc']:.2e} lpwa {r['err_lpwa_sc']:.2e} two {r['err_two_sc']:.2e} | rim {r['rim_ratio']:.1e}", flush=True)
    if float(s) == 0.0:
        good = r["err_dbl_sc"] < 10 * max(r["err_sc_exact"], r["rim_ratio"]) and r["err_sc_exact"] < 1e-2
        ok &= good; print(f"  planar: Smythe = SC within the truncation floor -> {'OK' if good else 'FAIL'}")
print("VALIDATE", "OK" if ok else "FAIL (stop and report)"); sys.exit(0 if ok else 1)
