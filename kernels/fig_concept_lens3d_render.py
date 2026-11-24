"""Rendered 3D concept illustration for Fig. 1 (PyVista / VTK).
Plano-convex glass lens with the prescribed tangential electric field (blue) on the curved
surface and radiation leaving along the local surface normals (orange).
Requires pyvista and Pillow:  python fig_concept_lens3d_render.py
Writes ../figures/output/fig_concept_lens3d.png (transparent background, auto-cropped)."""
import os
import numpy as np
import pyvista as pv
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures", "output", "fig_concept_lens3d.png"); os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ---------------------------------------------------------------- geometry
a, R, t_edge = 1.0, 1.45, 0.14                 # aperture radius, RoC of front, edge thickness
z0 = t_edge - np.sqrt(R**2 - a**2)             # centre of the front sphere (rim at z = t_edge)

# dome: fine sphere with its poles on the x-axis (no singular apex), clipped at the rim plane
dome = pv.Sphere(radius=R, center=(0, 0, z0), direction=(1, 0, 0),
                 theta_resolution=400, phi_resolution=400)
dome = dome.clip(normal=(0, 0, -1), origin=(0, 0, t_edge), invert=True)
side = pv.Cylinder(center=(0, 0, t_edge / 2), direction=(0, 0, 1), radius=a, height=t_edge,
                   resolution=400, capping=False)
back = pv.Disc(center=(0, 0, 0), inner=0, outer=a, normal=(0, 0, -1), r_res=1, c_res=400)
def ring(z):                                   # closed polyline (pv.Circle is a FILLED disk)
    t = np.linspace(0, 2 * np.pi, 401)
    return pv.lines_from_points(np.c_[a * np.cos(t), a * np.sin(t), np.full_like(t, z)])

def front(x, y):
    p = np.array([x, y, z0 + np.sqrt(R**2 - x * x - y * y)])
    return p, (p - np.array([0, 0, z0])) / R

seeds = [(0.0, 0.0)]
for r, n in [(0.42, 6), (0.80, 11)]:
    seeds += [(r * np.cos(t), r * np.sin(t)) for t in np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.2]

rad_arrows, e_arrows = [], []
ex = np.array([1.0, 0.0, 0.0])
for x, y in seeds:
    p, n = front(x, y)
    rad_arrows.append(pv.Arrow(start=p + 0.01 * n, direction=n, scale=0.55, tip_length=0.22,
                               tip_radius=0.065, shaft_radius=0.024, tip_resolution=48,
                               shaft_resolution=48))
    et = ex - (ex @ n) * n; et /= np.linalg.norm(et)
    e_arrows.append(pv.Arrow(start=p + 0.018 * n - 0.11 * et, direction=et, scale=0.22,
                             tip_length=0.36, tip_radius=0.1, shaft_radius=0.035, tip_resolution=36,
                             shaft_resolution=36))
rad, eas = pv.merge(rad_arrows), pv.merge(e_arrows)

# ---------------------------------------------------------------- scene
pl = pv.Plotter(off_screen=True, window_size=(2600, 1900), lighting="none")
pl.enable_depth_peeling(number_of_peels=16, occlusion_ratio=0.0)
glass = dict(color="#bcdcf4", opacity=0.36, smooth_shading=True, ambient=0.25, diffuse=0.6,
             specular=1.0, specular_power=120)
for m in (dome, side, back):
    pl.add_mesh(m, **glass)
pl.add_mesh(ring(t_edge), color="#2f6db3", line_width=3, opacity=0.9)
pl.add_mesh(ring(0.0), color="#2f6db3", line_width=2, opacity=0.45)
arrow_look = dict(smooth_shading=True, ambient=0.25, diffuse=0.75, specular=0.55, specular_power=40)
pl.add_mesh(rad, color="#ee8a24", **arrow_look)
pl.add_mesh(eas, color="#1f5aa6", **arrow_look)

# three-point lighting
pl.add_light(pv.Light(position=(-3.0, -2.6, 4.2), focal_point=(0, 0, 0.3), intensity=0.95,
                      color="#fff4e6"))                                  # key (warm)
pl.add_light(pv.Light(position=(3.6, -1.2, 1.6), focal_point=(0, 0, 0.3), intensity=0.45,
                      color="#e6efff"))                                  # fill (cool)
pl.add_light(pv.Light(position=(0.4, 4.2, 2.8), focal_point=(0, 0, 0.3), intensity=0.6))   # rim
pl.add_light(pv.Light(light_type="headlight", intensity=0.2))

pl.camera_position = [(2.5, -3.7, 2.9), (0.0, 0.0, 0.36), (0, 0, 1)]
pl.camera.view_angle = 24
pl.enable_anti_aliasing("ssaa")
img = pl.screenshot(transparent_background=True, return_img=True)
pl.close()

# ---------------------------------------------------------------- crop and save
ys, xs = np.where(img[..., 3] > 2)
pad = 30
img = img[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
Image.fromarray(img).rotate(-90, expand=True).save(OUT)          # 90 deg clockwise for Fig. 1
print("wrote", os.path.abspath(OUT), img.shape)
