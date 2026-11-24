"""Planar limit: closed-form CBIM (direct sum) vs FFT angular spectrum method, vector field."""
import numpy as np
K = 2 * np.pi
w0, z, L, dx = 3.0, 5.0, 40.0, 0.2          # waist, propagation distance, half-window, sampling (lambda)
x = np.arange(-L, L, dx); X, Y = np.meshgrid(x, x, indexing="ij")
Ex0 = np.exp(-(X**2 + Y**2) / w0**2)          # x-polarised Gaussian on the plane z=0

# --- ASM (FFT): Ex propagates with e^{ikz z}, Ez from transversality
kx = 2 * np.pi * np.fft.fftfreq(x.size, dx); KX, KY = np.meshgrid(kx, kx, indexing="ij")
kz = np.sqrt((K**2 - KX**2 - KY**2).astype(complex))
A = np.fft.fft2(Ex0)
Ex_asm = np.fft.ifft2(A * np.exp(1j * kz * z))
kzs = np.where(np.abs(kz) < 1e-12, 1.0, kz)
Ez_asm = np.fft.ifft2(np.where(np.abs(kz) < 1e-12, 0.0, -KX / kzs) * A * np.exp(1j * kz * z))
assert np.all(np.isfinite(Ez_asm)) and np.all(np.isfinite(Ex_asm))

# --- closed-form CBIM, direct sum, at a subset of observation points
idx = [(i, j) for i in range(x.size // 2 - 40, x.size // 2 + 41, 8) for j in range(x.size // 2 - 40, x.size // 2 + 41, 8)]
m = np.abs(Ex0) > 1e-16
Xs, Ys, E0 = X[m], Y[m], Ex0[m]
errs = []
for i, j in idx:
    xb, yb = x[i] - Xs, x[j] - Ys
    t = np.sqrt(xb**2 + yb**2 + z**2)
    c = (1 - 1j * K * t) * np.exp(1j * K * t) / t**3 * (2 * np.pi) / (4 * np.pi**2) * dx * dx
    Ex_c = np.sum(E0 * c * z)                  # E1 term
    Ez_c = np.sum(E0 * c * (-xb))              # E3 term
    errs.append(max(abs(Ex_c - Ex_asm[i, j]), abs(Ez_c - Ez_asm[i, j])) / np.abs(Ex_asm).max())
assert np.all(np.isfinite(errs))
print(f"{len(idx)} observation points at z = {z} lambda: max |CBIM closed form - ASM| / max|E| = {max(errs):.2e}")
