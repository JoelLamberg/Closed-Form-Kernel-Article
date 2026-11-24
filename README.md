# Closed-form kernels of the curved boundary integral method and their reduction to Smythe's formula

Code and data for the article by J. Lamberg, L. Lamberg and Z. Taylor (manuscript in preparation).
Every figure and number of the main text and of Supplement 1 can be regenerated from this repository.
All code is Python. Lengths are in wavelengths and k = 2π unless stated otherwise.

## Quick start

```bash
pip install -r requirements.txt
bash data/unpack.sh                      # unzips the raw results next to their archives

python kernels/verify_kernels.py         # Fig. 2, kernel identities (closed form vs spectral quadrature, Prop. 1, Prop. 2)
python kernels/bench_speed.py            # Fig. 3, closed form vs spectral quadrature, cost per pair
python kernels/planar_asm_check.py       # Sec. 3.1, planar limit vs FFT angular spectrum method (7e-12)
python kernels/gradient_test.py          # Sec. 3.1 and Supplement Sec. 4, Adam gradient test

python figures/fig_setup_v2.py           # Fig. 4
python figures/fig_eonly_map_v4.py       # Fig. 5
python figures/plot_focal_fields.py data/focus figures/output/fig6                    # Fig. 6
python figures/plot_focus_sweep_v3.py data/focus/unz/results figures/output/fig7      # Fig. 7
python figures/fig_las_crossover.py      # Fig. 8
python figures/plot_focal_fields.py data/focus figures/output/figS1 --comp y          # Supplement Fig. S1
```

Figures 4, 5 and 8 read compact `.mat` files in `data/figures/`. These are produced from the raw
results by `figures/export_setup_data.py`, `figures/export_eonly_data.py` and
`figures/export_las_data.py`. Each figure script is self-contained, with its own style block.
Figure 1 is rendered by `kernels/fig_concept_lens3d_render.py` (needs `pyvista` and `Pillow`).

## Layout

| Folder | Content |
|---|---|
| `kernels/` | closed-form kernels (`cbim_core.py`), verification, speed benchmark, planar check, gradient test, Fig. 1 render |
| `map/` | the 301 × 301 map over curvature and aperture size (Sec. 4.1); `cbim_sc_core.py` holds the beam, caps, CBIM, Stratton–Chu, LPWA and the adaptive ladder; `run_map.py` is the CPU runner; `pc_runner/` the GPU runner used on the workstation |
| `map/corner/`, `map/backface/` | direct one-sided runs at the 6,261 map points where an observation point lies behind the tangent plane of a source point (4,750 + 2,723 points) |
| `focus/` | focused-beam sweep (Sec. 4.2), exact plane-wave-sum beam, runners, focal-plane fields for Fig. 6 |
| `stronger_H/` | second-order local estimate of the magnetic field, LAS2c (Sec. 4.3, Supplement Sec. 5), map and focus runners |
| `bench/` | cost per pair of all methods on the same pairs (Table 1) |
| `checks/` | verification of every run against its reference values; `merge_and_claims.py` builds `data/map/merged_map.npz` and prints the map statistics of Sec. 4.1 |
| `figures/` | one script per figure, plus the export scripts |
| `data/` | raw results (zip), merged arrays (npz), figure data (mat), benchmark logs |

## Data

| File | Content |
|---|---|
| `data/map/results_301x301.zip` | one JSON per map point (90,601), errors of CBIM, Stratton–Chu with exact and with LPWA magnetic field, sampling, Richardson order and uncertainty |
| `data/map/map_cache.npz` | the same as arrays |
| `data/map/merged_map.npz` | map with the one-sided values of the corner and back-face runs merged in (used by Fig. 5) |
| `data/corner/results_corner.zip`, `data/backface/results_backface.zip` | direct one-sided runs, logs, workstation benchmark |
| `data/focus/results_focus.zip` | focused-beam sweep, 169 grid points (132 computable) |
| `data/focus/fields_*.npz` | focal-plane fields of the three cases of Fig. 6 at 20 and 28 points per wavelength |
| `data/stronger_H/results_stronger_H.zip` | LAS2c runs, 961 map points and 46 focused beams, logs, GPU runner added on the workstation |
| `data/stronger_H/map_las.npz` | LAS2c map results as arrays (written by `checks/verify_hlas.py`) |
| `data/bench/` | timings of Fig. 3 and Table 1 |
| `data/figures/*.mat` | data of Figs. 4, 5 and 8 |

The 301 × 301 map run evaluated CBIM with the factor −1 of the original definition behind the
tangent plane of a source point. Where no observation point lies behind any tangent plane
(84,340 of the 90,601 points) this equals the one-sided form used in the article. At the other
6,261 points the one-sided form was computed directly (`map/corner/`, `map/backface/`), and
`checks/merge_and_claims.py` merges the two.

## Reproducing the large runs

The map, corner, back-face, focus and LAS2c runs are resumable, write one JSON per point and
use the adaptive sampling ladder with Richardson extrapolation described in Supplement 1,
Section 4. Each folder has a `validate*.py` that compares a few points against the stored
reference values in its `reference/` folder before a full run. On the CPUs of a 96-core
workstation alone the full map would have taken about 220 hours, so it was split between CPU
and GPU (`pc_runner/`, CuPy). A single point takes from seconds for small caps to about an hour
for the largest ones on a laptop, so any point or row of the map can be recomputed and
compared with the stored results.

## Requirements

Python 3.10 or newer with NumPy, SciPy and Matplotlib. CuPy is needed only for the GPU runners,
PyVista and Pillow only for Fig. 1.

## Citation

To be added on publication.
