"""Sweep grid for the focusing-beam test."""
import os
NAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)      # 1/e^2 intensity half-angle, sin(theta_e)
AS = (10.0, 20.0)                                    # aperture radius / lambda (a = 5 cannot reach the rim level)
A_LOWNA = {0.1: (40.0,)}                             # NA 0.1 needs a larger aperture
SS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)             # curvature a / Rc
ORIENTS = (False, True)                              # concave?


def tasks():
    return [(na, a, s, c) for na in NAS for a in A_LOWNA.get(na, AS) for s in SS for c in ORIENTS if not (s == 0 and c)]


def name(t):
    na, a, s, c = t
    return f"pt_na{na:.2f}_a{a:05.1f}_s{s:.2f}_{'cc' if c else 'cv'}.json"


def fpath(out, t):
    return os.path.join(out, name(t))


def cost(t):
    na, a, s, c = t
    return a**2 / max(1e-3, 1 - s**2)
