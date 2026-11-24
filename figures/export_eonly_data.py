"""Export the 301x301 map arrays used by Fig. 5 (fig_eonly_map_v4.py) to .mat.
Source: data/map/merged_map.npz (from checks/merge_and_claims.py) (one-sided computed directly at all 6,261 back-facing points)."""
import numpy as np, scipy.io as sio
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(REPO, 'data', 'map', 'merged_map.npz'))
sio.savemat(os.path.join(REPO, 'data', 'figures', 'fig_eonly_data.mat'), {k: d[k] for k in ("A", "S", "one", "lp", "fl", "two", "src")}, do_compression=True)
print("saved fig_eonly_data.mat")
