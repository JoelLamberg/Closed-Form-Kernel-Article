"""Export the data of Fig. 4 (fig_setup_v2.py): |E|^2 in the plane y = 0 for (a) the map test (CSP Gaussian beam,
convex cap a = 8, a/Rc = 0.5) and (b) the focusing test (NA 0.5, concave cap a = 10, a/Rc = 0.5), plus geometry.
"""
import sys, json
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'focus'))
import numpy as np, scipy.io as sio
import cbim_sc_core as c
import focus_core as fc
out = {"Z_RIM": c.Z_RIM}
# (a) map test
a, s = 8.0, 0.5; w0 = a / 5.5; b = c.K * w0**2 / 2
x = np.linspace(-1.6 * a, 1.6 * a, 420); z = np.linspace(0.0, 2.6 * a, 520); X, Z = np.meshgrid(x, z, indexing="ij")
E, _ = c.csp_fields(np.stack([X, np.zeros_like(X), Z], -1).reshape(-1, 3), b)
I = np.sum(np.abs(E)**2, -1).reshape(X.shape); I /= I.max()
zv = c.Z_RIM - np.sqrt((a / s)**2 - a**2) + a / s; zo = zv + a
out.update(a_x=x, a_z=z, a_I=I, a_a=a, a_s=s, a_zv=zv, a_zo=zo, a_L=2.5 * c.beam_w(zo, w0))
# (b) focusing test
na, a2, s2, cc = 0.5, 10.0, 0.5, True
F = json.load(open(os.path.join(REPO, 'focus', 'reference', 'focus_choice.json')))[f"{na}_{a2}_{s2}_1"][1]
zf = c.Z_RIM + F; beam = fc.beam_for(a2, s2, cc, na, zf); zv2 = fc.z_vertex(a2, s2, cc)
x = np.linspace(-1.3 * a2, 1.3 * a2, 360); z = np.linspace(zv2 - 3, zf + 4, 300); X, Z = np.meshgrid(x, z, indexing="ij")
E, _, _ = beam.fields(np.stack([X, np.zeros_like(X), Z], -1).reshape(-1, 3))
I = np.sum(np.abs(E)**2, -1).reshape(X.shape); I /= I.max()
out.update(b_x=x, b_z=z, b_I=I, b_a=a2, b_s=s2, b_na=na, b_zf=zf, b_zv=zv2)
sio.savemat(os.path.join(REPO, 'data', 'figures', 'fig_setup_data.mat'), out, do_compression=True); print("saved fig_setup_data.mat")
