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
tqdm>=4.60
numba>=0.56  # HIGHLY RECOMMENDED for 5-10x speedup
```

Install dependencies:
```bash
pip install numpy scipy dask rasterio tqdm numba
```

**Important:** Without Numba, processing will use pure Python loops and be 5-10x slower.

## Quick Start

### 1. Generate Lookup Tables (one-time)

Before processing flight lines, generate the lookup tables:

```bash
python generate_luts.py --wvl wvl.npy --iop IOP_2008_ASCIItable.txt --output-dir luts
```

This takes 30-60 minutes and creates:
- `sbd_lut.npy` - Scaled band depth lookup table
- `albedo_lut.npy` - Clean snow albedo lookup table  
- `anisotropy_lut.npz` - BRDF anisotropy correction factors

### 2. Process Single Flight Line

```bash
python run_issia.py \
    --data-dir /path/to/atcor/data \
    --flight-line 24_4012_05_2024-06-06_17-54-38-rect_img \
    --output-dir /path/to/output \
    --lut-dir luts
```

### 3. Batch Process Multiple Flight Lines

Process all flight lines in a directory:
```bash
python run_issia_batch.py \
    --data-dir /path/to/atcor/data \
    --output-dir /path/to/output \
    --lut-dir luts \
    --continue-on-error
```

Process specific flight lines:
```bash
python run_issia_batch.py \
    --data-dir /path/to/atcor/data \
    --output-dir /path/to/output \
    --flight-lines flight1 flight2 flight3
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
| `{flight_line}_albedo.tif` | Broadband albedo | 0-1 |
| `{flight_line}_rf.tif` | Radiative forcing | W/m² |

### Optional Diagnostics

Use `--diagnostics` flag to also save:
- `_slope.tif`, `_aspect.tif` - Terrain parameters
- `_theta_i_eff.tif`, `_theta_v_eff.tif`, `_raa_eff.tif` - Effective viewing angles
- `_band_depth.tif` - 1030nm absorption band depth

## Processing Parameters

### Snow Detection Mask

Pixels are processed if they meet all criteria:
- NDSI ≥ 0.87 (snow/ice)
- Local illumination angle ≤ 85°
- Not in shadow (first band / 560nm band ≤ 1.0)

### LUT Dimensions

| Parameter | Range | Step | Values |
|-----------|-------|------|--------|
| Grain radius | 30-10000 μm | 30 | 333 |
| Illumination angle | 0-85° | 5 | 18 |
| Viewing angle | 0-85° | 5 | 18 |
| Relative azimuth | 0-360° | 10 | 37 |

## Package Structure

```
issia_package/
├── issia_core.py          # Base processor class
├── issia_processor.py     # Extended processor with pixel-wise methods
├── run_issia.py           # Single flight line processing
├── run_issia_batch.py     # Batch processing
├── generate_luts.py       # Lookup table generator
├── wvl.npy               # Wavelength array
├── IOP_2008_ASCIItable.txt # Ice optical properties
└── README.md
```

## Algorithm Reference

The algorithm is based on:

- **ART Model**: Kokhanovsky & Zege (2004) "Scattering optics of snow"
- **Terrain Correction**: Dumont et al. (2011) local viewing geometry
- **Radiative Forcing**: Painter et al. (2013) integration method

## Command Line Options

### run_issia.py

```
--data-dir      Directory containing ATCOR files (required)
--flight-line   Flight line identifier (required)
--output-dir    Output directory (required)
--lut-dir       LUT directory (default: luts)
--wvl-path      Wavelength file (default: wvl.npy)
--subset        Spatial subset as "ymin,ymax,xmin,xmax"
--diagnostics   Save diagnostic outputs
```

### run_issia_batch.py

```
--data-dir          Directory containing ATCOR files (required)
--output-dir        Output directory (required)
--lut-dir           LUT directory (default: luts)
--wvl-path          Wavelength file (default: wvl.npy)
--flight-lines      Specific flight lines to process
--flight-list       Text file with flight line IDs
--diagnostics       Save diagnostic outputs
--continue-on-error Continue if a flight line fails
```

## Performance

### Parallel Processing

The package uses **Numba JIT compilation** for parallel pixel processing. This provides:
- 5-10x speedup over pure Python
- Automatic multi-core utilization via `prange`
- No Dask overhead for pixel-wise operations

**With Numba (recommended):**
- Grain size retrieval: ~5-10 seconds for 2000×2000 pixels
- Total processing: ~30-60 seconds per flight line

**Without Numba:**
- Same operations: 3-10 minutes
- Pure Python nested loops are the bottleneck

### Why Not Dask Parallelization?

Dask's `map_blocks` with Python functions hits the Global Interpreter Lock (GIL). The inner pixel loops in the original code were pure Python, defeating Dask's parallelism. Numba's `prange` releases the GIL and enables true parallel execution.

### Typical Processing Times

| Operation | With Numba | Without Numba |
|-----------|------------|---------------|
| Band depth | 5-10s | 60-120s |
| Grain size retrieval | 3-8s | 45-90s |
| Anisotropy lookup | 5-15s | 60-180s |
| **Total per flight line** | **30-60s** | **5-10 min** |

*Times for 2000×2000 pixel image on 8-core CPU*

## License

[Add your license here]

## Citation

[Add citation information here]
