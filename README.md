# ISSIA - Imaging Spectrometer Snow and Ice Algorithm

Python implementation of the ISSIA algorithm for retrieving snow and ice surface properties from hyperspectral imagery.

## Overview

ISSIA retrieves three primary products from ATCOR-processed hyperspectral data:

1. **Snow Grain Size** (μm) - Optical effective grain radius
2. **Broadband Albedo** - Hemispherical reflectance integrated across solar spectrum
3. **Radiative Forcing** (W/m²) - Absorption enhancement by light-absorbing particles

## Installation

### Requirements

```
numpy>=1.20
scipy>=1.7
dask>=2022.0
rasterio>=1.3
gdal>=3.0
tqdm>=4.60
numba>=0.56  # HIGHLY RECOMMENDED for 5-10x speedup
```

Install dependencies:
```bash
pip install numpy scipy dask rasterio tqdm numba
```

**Important:** Without Numba, processing will use shared-memory multiprocessing and be 5-10x slower.

## Quick Start

### 1. Generate Lookup Tables (one-time)

Before processing flight lines, generate the lookup tables:

```bash
python generate_luts.py --wvl wvl.npy --iop IOP_2008_ASCIItable.txt --output-dir luts
```

This takes 30–60 minutes and creates:
- `sbd_lut.npy` — Scaled band depth lookup table
- `albedo_lut.npy` — Clean snow albedo lookup table
- `anisotropy_lut.npz` — BRDF anisotropy correction factors

### 2. Process Single Flight Line

```bash
python run_issia.py \
    --data-dir /path/to/atcor/data \
    --flight-line 24_4012_05_2024-06-06_17-54-38-rect_img \
    --output-dir /path/to/output \
    --lut-dir luts
```

### 3. Batch Process an Acquisition

Process all flight lines in a directory, then mosaic the results:

```bash
python run_issia_batch.py \
    --data-dir /path/to/atcor/clipped \
    --output-dir /path/to/output \
    --lut-dir /path/to/luts \
    --continue-on-error \
    --mosaic
```

Flight lines whose three output files (`_gs.tif`, `_albedo.tif`, `_rf.tif`) already exist are **skipped automatically**, so re-running the command only processes new or failed lines and then rebuilds the mosaic.

### 4. Mosaic Only

If flight lines are already processed and you only need the mosaic:

```bash
# All three products from a batch output directory:
python mosaic.py --batch-dir /path/to/output --mosaic-dir /path/to/mosaics

# Single product from a wildcard:
python mosaic.py -i "output/*_albedo.tif" -o mosaic_albedo.tif
```

## Input Data Requirements

ISSIA expects ATCOR-4 output files for each flight line:

| File | Description |
|------|-------------|
| `{flight_line}_atm.dat` | Atmospherically corrected reflectance (ENVI BSQ) |
| `{flight_line}_eglo.dat` | Global irradiance flux (ENVI BSQ) |
| `{flight_line}_slp.dat` | Terrain slope (degrees) |
| `{flight_line}_asp.dat` | Terrain aspect (degrees) |
| `{flight_line}.inn` | ATCOR input file with solar geometry |

## Output Products

| File | Description | Units |
|------|-------------|-------|
| `{flight_line}_gs.tif` | Snow grain size | μm |
| `{flight_line}_albedo.tif` | Broadband albedo | 0–1 |
| `{flight_line}_rf.tif` | Radiative forcing | W/m² |

Mosaic outputs (when `--mosaic` is used):

| File | Description |
|------|-------------|
| `mosaics/mosaic_gs.tif` | Seamless grain size mosaic |
| `mosaics/mosaic_albedo.tif` | Seamless albedo mosaic |
| `mosaics/mosaic_rf.tif` | Seamless radiative forcing mosaic |

### Optional Diagnostics

Use `--diagnostics` flag to also save:
- `_slope.tif`, `_aspect.tif` — Terrain parameters
- `_theta_i_eff.tif`, `_theta_v_eff.tif` — Effective viewing angles
- `_band_depth.tif` — 1030 nm absorption band depth

## Seamless Mosaicking

`mosaic.py` uses **distance-transform edge weighting** to blend overlapping flight lines without seams:

- Each pixel's weight = its distance to the nearest nodata edge, minus `--edge-setback` pixels
- In overlap zones the weighted average naturally de-emphasises swath edges, where ATCOR correction is least accurate
- If a valid region is narrower than the setback (e.g., a thin snow patch), the setback is automatically ignored so no data is lost
- Processing is tile-based (default 2048 px) to handle large mosaics without loading everything into memory

```bash
python mosaic.py --batch-dir output/ --mosaic-dir mosaics/ --edge-setback 50
```

## Processing Parameters

### Snow Detection Mask

Pixels are processed if they meet all criteria:
- NDSI ≥ 0.87 (snow/ice)
- Local illumination angle ≤ 85°
- Not in shadow (first band / 560 nm band ≤ 1.0)

### LUT Dimensions

| Parameter | Range | Step | Values |
|-----------|-------|------|--------|
| Grain radius | 30–5000 μm | 30 | 167 |
| Illumination angle | 0–85° | 5 | 18 |
| Viewing angle | 0–85° | 5 | 18 |
| Relative azimuth | 0–360° | 10 | 37 |

## Package Structure

```
ISSIA_python/
├── issia_core.py            # Base processor class
├── issia_processor.py       # Extended processor (Numba pixel-wise methods,
│                            #   chunked albedo+RF computation)
├── run_issia.py             # Single flight line processing with tqdm bars
├── run_issia_batch.py       # Batch processing (skip existing, --mosaic flag)
├── mosaic.py                # Seamless distance-weighted mosaic
├── generate_luts.py         # Lookup table generator
├── generate_snicar_luts.py  # SNICAR-based LUT generator
├── compare_optical_properties.py  # Validation against MATLAB outputs
├── wvl.npy                  # Wavelength array (nm)
└── IOP_2008_ASCIItable.txt  # Ice optical properties
```

## Command Line Reference

### run_issia_batch.py

```
--data-dir          Directory containing ATCOR files (required)
--output-dir        Output directory (required)
--lut-dir           LUT directory (default: luts)
--wvl-path          Wavelength file (default: wvl.npy)
--flight-lines      Specific flight lines to process
--flight-list       Text file with flight line IDs (one per line)
--diagnostics       Save diagnostic outputs
--continue-on-error Continue if a flight line fails
--workers           Number of worker threads
--mosaic            Mosaic all products after processing
--mosaic-dir        Output directory for mosaics (default: <output-dir>/mosaics)
--edge-setback      Pixels to suppress at each swath edge in mosaic (default: 50)
```

### run_issia.py

```
--data-dir      Directory containing ATCOR files (required)
--flight-line   Flight line identifier (required)
--output-dir    Output directory (required)
--lut-dir       LUT directory (default: luts)
--wvl-path      Wavelength file (default: wvl.npy)
--subset        Spatial subset as "ymin,ymax,xmin,xmax"
--diagnostics   Save diagnostic outputs
--workers       Number of worker threads
```

### mosaic.py

```
--batch-dir     Batch output dir — mosaics all three products
--mosaic-dir    Output directory for mosaics (default: mosaics)
-i / --input    Input wildcard for single-product mosaic
-o / --output   Output path (required with --input)
--tile-size     Processing tile size in pixels (default: 2048)
--edge-setback  Pixels to suppress at each swath edge (default: 50)
```

## Performance

Processing uses **Numba JIT compilation** for all pixel-wise operations:
- Band depth: chunked by rows with per-chunk progress bar
- Grain size / anisotropy / RF: Numba `prange` across all CPU cores
- Albedo + RF: computed in a single column-chunked pass to avoid materialising
  the full spectral albedo array (~13 GB for a typical flight line)

**Measured performance** (10-core Mac, 451-band FENIX data, ~2700×2600 pixels):

| Step | Time |
|------|------|
| LUT loading | ~11s |
| Data loading | ~125s (disk I/O bound) |
| Masking | ~10s |
| Band depth (Numba) | ~11s |
| Grain size (Numba) | <1s |
| Anisotropy (Numba) | ~14s |
| Albedo + RF (chunked) | ~30–40s |
| **Total per flight line** | **~90s** |

## Algorithm Reference

- **ART Model**: Kokhanovsky & Zege (2004) "Scattering optics of snow"
- **Terrain Correction**: Dumont et al. (2011) local viewing geometry
- **Radiative Forcing**: Painter et al. (2013) integration method

## License

[Add your license here]

## Citation

[Add citation information here]
