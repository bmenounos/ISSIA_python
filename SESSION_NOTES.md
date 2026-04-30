# ISSIA Python – Session Notes (2026-04-30)

## Status (updated end of session)
- All three products (grain size, albedo, RF) now agree: **128,335 valid pixels** for
  `24_4012_02_GLOF_2024-08-31_19-52-46-rect_img`.
- Remaining gap to MATLAB: ~16,471 pixels (144,806 − 128,335). Cause TBD (see Next Steps).
- Changes are **uncommitted** — verify before committing.

## Problem
Python output produced only ~48,267 valid pixels vs MATLAB's ~144,806 expected pixels for
flight line `24_4012_02_GLOF_2024-08-31_19-52-46-rect_img`. Root cause: five algorithmic
regressions introduced during the dask-port period (after January 2026).

---

## Changes Made (uncommitted – test before committing)

### `run_issia.py`
| # | What changed | Before → After | Why |
|---|---|---|---|
| 1 | NDSI threshold default (`_CONFIG_DEFAULTS` + signature) | 0.4 → **0.87** | MATLAB hardcodes `NDSIval=0.87` |
| 2 | Band depth window left edge (`continuum_removal_830_1130` + `parallel_band_depth`) | 900 nm → **830 nm** | MATLAB uses 830–1130 nm ("MATLAB standard") |
| 3 | Numba continuum removal (`_continuum_removal_single`) | Simplified local-maxima hull → **monotone-chain upper convex hull** | Old algorithm returned NaN for ~66% of valid snow/ice pixels |
| 4 | Scipy fallback vertex filtering (`continuum_removal_830_1130`) | `np.delete(K,[0,1])` → **`K[(K>0) & (K<=n)]`** | Matches reference package exactly |
| 5 | Smoothing scope | savgol inside band-depth window only → **savgol on full 451-band strip before NDSI** | MATLAB does `smoothdata(cube,3,'sgolay',10)` before NDSI and band depth |
| 6 | Removed savgol from `_process_bd_row` (shared-mem fallback) | Had `savgol_filter(spectrum,11,3)` → **removed** | Was applied in wrong scope |
| 7 | CLI help string | "default 0.87" was changed to 0.4 → **back to 0.87** | Consistency |
| 8 | Removed unused `from scipy.signal import savgol_filter` import, then re-added it | — | Import needed for full-strip smoothing |

### `issia_processor.py`
| # | What changed | Before → After | Why |
|---|---|---|---|
| 9 | `compute_albedo_rf_chunked` NaN handling | `np.trapezoid(sa * fx)` with NaN from zero-masked bands | Replace NaN with 0 before integration (`sa_int`, `flux_int`); gate broadband on `np.isfinite(gs_slice)` |

**Effect**: Albedo valid pixel count went from ~4,490 → 128,335, now matching grain size and RF exactly.  
**Root cause**: `refl = np.where(refl == 0, np.nan, refl)` in `run_issia.py` converts SWIR absorption-window bands (1400nm, 1900nm) to NaN. `np.trapezoid` with any NaN in the integrand returns NaN for the entire pixel. MATLAB keeps zeros in the integration (no NaN masking).

### `issia_config.json`
- `ndsi_threshold`: 0.4 → **0.87**
- Updated note to say "0.87=MATLAB default; use 0.4 for dirty/glacial ice"

---

## Diagnostic Findings for 19-52-46 Flight Line

| Stage | Pixel count |
|---|---|
| Total image pixels | 11,455,913 |
| ATM nodata (raw=15000) | 7,297,203 |
| Valid spectral (r600>0, r1500>0, not nodata) | 969,410 |
| + NDSI ≥ 0.87 | 167,775 |
| + shadow r380/r560 ≤ 1.0 | 145,446 |
| + theta_eff ≤ 85° | **144,806** (expected MATLAB output) |
| + NDSI ≥ 0.4 (alternative threshold) | 211,093 → 168,769 → 167,444 |
| Old Python output (broken hull) | 48,267 |

**Valid spectral data spatial extent**: rows 1801–4272 only (rows 0–1800 have zero
reflectance at 600 nm and 1500 nm — ATCOR artifact, not a code bug).

**Chunk 0 (rows 0–2047) always has 0 valid pixels**: rows 0–2047 max NDSI = 0.53.
No snow/ice at the bottom of this flight line. Valid pixels are in chunks 1–2 (rows 2048+).

**`refl[65]: nan–nan` in chunk diagnostic is NOT a bug**: that line is printed *after*
`refl[:, ~final_mask] = np.nan`, so if 0 pixels pass the mask all values are NaN.

---

## MATLAB vs Python Differences (from reading ISSIA.m on GitHub)

| Feature | MATLAB | Python (current) |
|---|---|---|
| NDSI threshold | 0.87 | 0.87 ✓ |
| Band depth window | 830–1130 nm | 830–1130 nm ✓ |
| Spectral smoothing | `smoothdata(cube,3,'sgolay',10)` on full cube before NDSI | `savgol_filter(refl,11,3,axis=0)` on strip before NDSI ✓ (approximate) |
| Shadow ratio | R(band1≈380nm) / R(560nm) ≤ 1.0 | `refl[0]/refl[idx_560]` ✓ |
| Solar angle rounding | `round()` to nearest integer | Not rounded (minor difference) |
| Nodata handling | Implicit (NDSI≈0 for nodata) | Explicit mask for `raw==15000` ✓ |

**Smoothing parameter uncertainty**: MATLAB uses `smoothdata(...,'sgolay',10)` where 10
is polynomial degree and window is auto-determined. Python uses `window=11, polyorder=3`
as an approximation. If pixel counts still don't match, try adjusting window_length.

---

## MATLAB Reference File Status
- `24_4012_02_GLOF_2024-08-31_19-52-46-rect_img_HSI_albedo.tif` — **does not exist** in Downloads.
  Only `.aux.xml` present. No MATLAB reference for this specific flight line to compare against.
- Available MATLAB outputs for same campaign (different times):
  - `19-38-32`: 54,807 valid pixels
  - `19-42-59`: 252,021 valid pixels

---

## Next Steps
1. ✅ All three products now output 128,335 valid pixels (grain size = albedo = RF)
2. Remaining 16K pixel gap to MATLAB (144,806) — possible causes to investigate:
   a. **Savgol mismatch**: MATLAB uses `smoothdata(...,'sgolay',10)` (polyorder=10, auto window).
      Python uses `window=11, polyorder=3`. Try `polyorder=5` or `window=21`.
   b. **Zero-masking before vs after smoothing**: smoothing order relative to nodata mask may differ
   c. **Shadow filter edge**: MATLAB `shadow_b1=1` → first non-zero band? Check if Python uses idx 0 vs 1
3. To commit: `git commit -am "Fix ISSIA regressions: hull, NDSI, savgol, albedo NaN propagation"`
4. Do NOT `git push` until pixel count gap is resolved or accepted
5. To use 0.4 threshold for glacial ice: `python run_issia.py ... --ndsi-threshold 0.4`

---

## January Baseline Branch
Branch `january-baseline` points to commit `67b8c58` (last January 2026 commit).
Use `git checkout january-baseline` to run the old code for comparison.
