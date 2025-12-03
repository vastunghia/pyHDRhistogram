


# pyHDRhistogram
This repo provides a compact Python script that plots **RGB + luma histograms** for a
PQ-coded, Display P3 PNG (16-bit or 8-bit), using an **Adobe-style layout**:

- The **SDR region** (0 → **100 nit**) is on the **left**.
- The **HDR region** (**100 → 3200 nit**) is on the **right** with a **fixed 5-stop span**.
- The vertical split at **100 nit** is constant; dashed ticks at 200, 400, 800, 1600, 3200 nit.
- Dark background (cooler tone for SDR, warmer for HDR) matching Adobe’s feel.
- Curves are **lightly smoothed**; a **white curve** overlays **luma Y with Rec.709 coefficients**
  computed in the PQ-decoded, linear domain.

The script reads raw PNG codes, applies **PQ EOTF (ST 2084)** to obtain linear relative luminance,
builds histograms with **matched binning at the 100-nit joint** (to avoid the hinge), then maps the X-axis
as **sRGB code** on the SDR half and **log₂ luminance** on the HDR half (fixed 5 stops).

> **Why 100 nit for SDR?**  
> 100 nit is the conventional SDR reference white. Displays can be brighter, but 100 nit is a stable reference
> for analysis and tone-mapping discussions. You may find that in some cases 200 nit is used instead of 100
> (possibly because 203 nit is the standard paper white in HDR pipelines)

---

## Requirements

- Python 3.9+
- Packages: `opencv-python`, `numpy`, `matplotlib`

Install:

```bash
pip install opencv-python numpy matplotlib
```

---

## How to run:

```bash
python pyHDRhistogram.py myHDRimage.png
```

A window opens with the Adobe-style histogram (RGB + luma).

---

## How it works (high level)

1. **Read PNG** with OpenCV (keeps 16-bit depth if present). Convert BGR→RGB and normalize to [0,1].
2. **PQ decode** (ST-2084 EOTF) per channel → linear relative luminance (0..1 @ 10,000 nit).
3. Convert to “relative to white” units `u = Y / WHITE`, with `WHITE = 100 nit` (configurable).
4. **Binning (joint-matched):**
- SDR: edges are **sRGB-code spaced** (equispaced in sRGB code).
- HDR: edges are **log2-spaced** from 100 to 3200 nit, with the **first HDR bin width** matched to the **last SDR bin width** (in linear luminance units) to avoid a visible discontinuity at 100 nit.
5. **Histograms**: counts for R, G, B; **luma** is computed after PQ decode and histogrammed with the same edges.
6. **Smoothing**: a short moving-average window is applied to all curves (adjustable).
7. **X mapping**:
- SDR bins → sRGB code mapped to `[0 .. X_AT_100]`, with `X_AT_100 = 0.375` (because HDR occupies 5 stops and we want x = 200 nit exactly at the center as in Adobe's histograms → 0.625 of the axis).
- HDR bins → log2 luminance mapped to `[X_AT_100 .. 1]` covering exactly **5 stops** (100, 200, 400, 800, 1600, 3200 nit).
8. **Styling**: dark background; SDR label on the left of the 100-nit split; HDR label on the right; baseline (y=0) shown as a white line; vertical markers at HDR stops.

---

## Customization

Key constants inside the script:

- `WHITE = 100.0` — SDR reference white (nit).
- `HDR_MAX_NITS = 3200.0` — top of the HDR range (exactly 5 stops above 100).
- `X_AT_100 = 0.375` — position of the 100-nit split on the X axis.
- `BINS_SDR = 256` — number of SDR bins (sRGB-code spaced).
- `SMOOTH_WIN = 7` — moving average window for curve smoothing (odd integer).

---

## Notes & assumptions

- Input should be a PNG with **P3 PQ** tagging (e.g., Lightroom “P3 PQ” ICC). The script treats raw codes as PQ and decodes them before analysis.
- The HDR histogram is always shown over a **fixed 5-stop span** (100→3200 nit) regardless of actual image peak.
- The joint-matched binning minimizes, but does not mathematically eliminate, tiny slope changes at the SDR/HDR split.
- The plot intentionally omits Y-axis ticks/labels; it is intended for **shape/position** comparison rather than absolute counts.

---

## License

MIT (attribution appreciated).