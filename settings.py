"""Accuracy settings for every runner (edit here only). 'High accuracy' defaults: tol 1e-4 relative to each method's
difference from Stratton-Chu, floor 1e-10, ladder ppw 5...40, 49x49 Simpson focal plane +-3 spot radii."""
CFG = dict(nobs=49, half_width=3.0, tol=1e-4, floor=1e-10, nobs_mid=25, rim_target=1e-4)
