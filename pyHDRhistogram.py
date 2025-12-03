# -*- coding: utf-8 -*-
"""
HDR histogram (Adobe-style) from a 16-bit PNG tagged P3 PQ
See README.md for overview & usage.
"""
from __future__ import annotations
import sys
from typing import Tuple

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ---------- Layout constants ----------
WHITE_NITS   = 100.0      # SDR/HDR boundary (x-axis vertical bar)
HDR_MAX_NITS = 3200.0     # rightmost HDR limit (5 stops above 100)
X_AT_100     = 0.375      # position of 100 nit on x-axis (left compressed SDR)
HDR_STOPS    = float(np.log2(HDR_MAX_NITS / WHITE_NITS))  # == 5

# ---------- Transfer functions ----------
def pq_eotf(v: np.ndarray) -> np.ndarray:
    """ST-2084 (PQ) code → linear relative luminance in [0..1] where 1 = 10k nit."""
    m1 = 2610/16384.0; m2 = 2523/32.0
    c1 = 3424/4096.0;  c2 = 2413/128.0; c3 = 2392/128.0
    v  = np.clip(v, 0.0, 1.0).astype(np.float32)
    vp = np.power(v, 1.0/m2)
    num = np.maximum(vp - c1, 0.0)
    den = c2 - c3 * vp
    return np.power(num/den, 1.0/m1)

def srgb_encode(x: np.ndarray) -> np.ndarray:
    """Linear → sRGB code (OETF)."""
    x = np.clip(x, 0.0, 1.0).astype(np.float32); a = 0.055
    y = np.empty_like(x); low = x <= 0.0031308
    y[low]  = 12.92 * x[low]
    y[~low] = (1 + a) * np.power(x[~low], 1/2.4) - a
    return y

def srgb_decode(y: np.ndarray) -> np.ndarray:
    """sRGB code → linear (inverse OETF)."""
    y = np.clip(y, 0.0, 1.0).astype(np.float32); a = 0.055
    x = np.empty_like(y); low = y <= 0.04045
    x[low]  = y[low] / 12.92
    x[~low] = np.power((y[~low] + a) / (1 + a), 2.4)
    return x

# ---------- Mapping: absolute nits → x in [0..1] (two-segment axis) ----------
def nits_to_x(nits: np.ndarray, white: float = WHITE_NITS) -> np.ndarray:
    n = np.asarray(nits, dtype=np.float32)
    x = np.empty_like(n, dtype=np.float32)

    # SDR: 0..white nit (sRGB-shaped) → [0, X_AT_100]
    sdr = (n <= white)
    if np.any(sdr):
        y = n[sdr] / white
        x[sdr] = X_AT_100 * srgb_encode(y)

    # HDR: white..HDR_MAX_NITS (log2) → [X_AT_100, 1]
    hdr = ~sdr
    if np.any(hdr):
        t = np.clip(np.log2(n[hdr] / white) / HDR_STOPS, 0.0, 1.0)
        x[hdr] = X_AT_100 + (1.0 - X_AT_100) * t
    return x

# ---------- Bin edges with matched hinge in u = Y/WHITE ----------
def _choose_bins_hdr_for_joint(bins_sdr: int, u_max: float,
                               n_min: int = 64, n_max: int = 4096) -> Tuple[int, np.ndarray, float]:
    # SDR last-bin width in u
    y_edges = np.linspace(0.0, 1.0, bins_sdr + 1, dtype=np.float32)
    u_sdr   = srgb_decode(y_edges)   # 0..1 in u
    u_sdr[-1] = 1.0
    w_s = float(u_sdr[-1] - u_sdr[-2])

    # Choose HDR bin count so first HDR bin width ≈ w_s
    L = np.log2(u_max); denom = np.log2(1.0 + w_s)
    N_star = int(round(L / denom)) if denom > 0 else n_min
    N = int(np.clip(N_star, n_min, n_max))
    return N, u_sdr, w_s

def adobe_edges_matched_joint(bins_sdr: int = 256, u_max: float = HDR_MAX_NITS/WHITE_NITS) -> np.ndarray:
    n_hdr, u_sdr, _ = _choose_bins_hdr_for_joint(bins_sdr, u_max)
    hdr_edges = 2.0 ** np.linspace(0.0, np.log2(u_max), n_hdr + 1, dtype=np.float64)
    return np.concatenate([u_sdr, hdr_edges[1:]])  # avoid duplicated 1.0

# ---------- IO & histogram ----------
def read_rgb01(path: str) -> np.ndarray:
    """Open PNG with OpenCV, return float RGB in [0,1] (preserving bit depth)."""
    im = cv.imread(path, cv.IMREAD_UNCHANGED)
    if im is None: raise SystemExit(f"Cannot read {path}")
    if im.ndim == 2: im = np.stack([im, im, im], axis=-1)
    if im.dtype == np.uint16: arr = im.astype(np.float32)/65535.0
    elif im.dtype == np.uint8: arr = im.astype(np.float32)/255.0
    else: arr = im.astype(np.float32)
    return arr[..., ::-1]  # BGR → RGB

def luma_from_linear_rgb(rgb_linear: np.ndarray) -> np.ndarray:
    """Luma from linear RGB (Rec.709 coefficients)."""
    r = rgb_linear[..., 0]; g = rgb_linear[..., 1]; b = rgb_linear[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def moving_average(y: np.ndarray, win: int = 11) -> np.ndarray:
    """Symmetric moving average (odd window)."""
    win = max(1, int(win))
    if win % 2 == 0: win += 1
    if win == 1: return y.astype(np.float64)
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(y.astype(np.float64), kernel, mode="same")

def compute_histograms(rgb_codes_pq: np.ndarray, edges_u: np.ndarray, smooth_win: int = 11):
    """Return x-centers and smoothed histograms (R,G,B,Y)."""
    lin = pq_eotf(rgb_codes_pq)     # 0..1 (1 == 10k nit)
    scale = 10000.0 / WHITE_NITS
    u_rgb = lin * scale
    y_lin = luma_from_linear_rgb(lin); u_y = y_lin * scale

    R = u_rgb[...,0].ravel(); G = u_rgb[...,1].ravel(); B = u_rgb[...,2].ravel(); Y = u_y.ravel()
    hR,_ = np.histogram(R, bins=edges_u); hG,_ = np.histogram(G, bins=edges_u)
    hB,_ = np.histogram(B, bins=edges_u); hY,_ = np.histogram(Y, bins=edges_u)

    centers_u = 0.5*(edges_u[:-1]+edges_u[1:]); centers_nit = centers_u * WHITE_NITS
    x_centers = nits_to_x(centers_nit, white=WHITE_NITS)

    hR = moving_average(hR, smooth_win); hG = moving_average(hG, smooth_win)
    hB = moving_average(hB, smooth_win); hY = moving_average(hY, smooth_win)
    return x_centers, hR, hG, hB, hY

# ---------- Plot ----------
def plot_adobe_style(png_path: str, smooth_win: int = 11, figsize=(12,4)) -> None:
    img_codes = read_rgb01(png_path)
    edges_u   = adobe_edges_matched_joint(bins_sdr=256, u_max=HDR_MAX_NITS/WHITE_NITS)
    x, hR, hG, hB, hY = compute_histograms(img_codes, edges_u, smooth_win)

    fig, ax = plt.subplots(figsize=figsize, facecolor="#111314"); ax.set_facecolor("#1d2324")
    # Backgrounds
    ax.axvspan(0, X_AT_100, facecolor="#1d2324", alpha=1.0)     # teal-ish SDR
    ax.axvspan(X_AT_100, 1, facecolor="#241a1b", alpha=1.0)     # maroon HDR

    # Curves
    ax.plot(x, hR, color=(1,0,0,0.95), lw=2.0)
    ax.plot(x, hG, color=(0,1,0,0.95), lw=2.0)
    ax.plot(x, hB, color=(0,0,1,0.95), lw=2.0)
    ax.plot(x, hY, color=(1,1,1,0.95), lw=2.2, zorder=5)

    # Baseline
    ax.axhline(0, color="white", lw=2)

    # Vertical bars
    ax.axvline(nits_to_x([WHITE_NITS])[0], color="#C0C6CC", lw=2.0)
    for n in (200, 400, 800, 1600, 3200):
        ax.axvline(nits_to_x([n])[0], color="#C0C6CC", ls=(0,(12,10)), lw=2.0, alpha=0.75)

    # X ticks (absolute nit, mapped)
    powers = np.arange(-10, int(np.log2(HDR_MAX_NITS/WHITE_NITS)) + 1 + 10)
    ticks_n = WHITE_NITS * (2.0**powers)
    ticks_n = ticks_n[(ticks_n >= 0.1) & (ticks_n <= HDR_MAX_NITS)]
    ticks_x = nits_to_x(ticks_n, white=WHITE_NITS)
    labels = [f"{ticks_n[0]:.1f}"] + [
        (f"{int(n)}" if n >= 10 else f"{n:.1f}" if n >= 1 else "")
        for n in ticks_n[1:]
    ]
    ax.set_xticks(ticks_x, labels, color="white")
    ax.set_xlim(0, 1)

    # Minimal chrome
    ax.set_yticks([]); ax.set_ylabel(""); ax.set_title("")
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("white"); ax.tick_params(axis="x", colors="white")

    # Labels
    y_top = ax.get_ylim()[1]
    ax.text(X_AT_100-0.01, 0.92*y_top, "SDR", ha="right", va="top",
            color="white", fontsize=11, weight="semibold")
    ax.text(X_AT_100+0.01, 0.92*y_top, "HDR", ha="left", va="top",
            color="white", fontsize=11, weight="semibold")

    plt.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.16)
    plt.show()

def main(argv):
    if len(argv) < 2:
        print("Usage: python pyHDRhistogram.py path/to/image.png")
        sys.exit(2)
    plot_adobe_style(argv[1])

if __name__ == "__main__":
    main(sys.argv)
