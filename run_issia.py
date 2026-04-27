#!/usr/bin/env python
"""
ISSIA - Optimized Single Flight Line Processing

Uses Numba JIT compilation for 5-10x speedup over pure Python.
All pixel-wise operations are parallelized across CPU cores.

Usage:
    python run_issia_optimized.py --data-dir /path/to/data --flight-line flight_id --output-dir /path/to/output
"""

import numpy as np
import argparse
import time
from pathlib import Path
import warnings
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter
import dask.array as da
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Try to import numba for band depth calculation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from issia_processor import ISSIAProcessorOptimized


# ============================================================================
# BAND DEPTH CALCULATION - NUMBA ACCELERATED
# ============================================================================

if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def _continuum_removal_single(spec_subset, wl_subset):
        """Single spectrum continuum removal - Numba compatible"""
        n = len(spec_subset)
        
        # Check for invalid data
        for i in range(n):
            if np.isnan(spec_subset[i]) or spec_subset[i] <= 0:
                return np.nan
        
        # Extended arrays for convex hull
        v_ext = np.empty(n + 2)
        v_ext[0] = 0
        v_ext[1:n+1] = spec_subset
        v_ext[n+1] = 0
        
        x_ext = np.empty(n + 2)
        x_ext[0] = wl_subset[0] - 1e-10
        x_ext[1:n+1] = wl_subset
        x_ext[n+1] = wl_subset[n-1] + 1e-10
        
        # Simple upper hull (for reflectance spectra)
        # Find local maxima as hull vertices
        hull_idx = []
        hull_idx.append(0)  # Start point
        
        for i in range(1, n):
            # Check if this is a local maximum or on upper envelope
            if spec_subset[i] >= spec_subset[i-1]:
                if i == n-1 or spec_subset[i] >= spec_subset[i+1]:
                    hull_idx.append(i)
        
        hull_idx.append(n-1)  # End point
        
        if len(hull_idx) < 2:
            return np.nan
        
        # Linear interpolation for continuum
        continuum = np.empty(n)
        hi = 0
        for i in range(n):
            while hi < len(hull_idx) - 1 and hull_idx[hi + 1] < i:
                hi += 1
            if hi >= len(hull_idx) - 1:
                hi = len(hull_idx) - 2
            
            i0, i1 = hull_idx[hi], hull_idx[hi + 1]
            if i1 == i0:
                continuum[i] = spec_subset[i0]
            else:
                t = (i - i0) / (i1 - i0)
                continuum[i] = spec_subset[i0] + t * (spec_subset[i1] - spec_subset[i0])
        
        # Ensure positive continuum
        for i in range(n):
            if continuum[i] < 1e-10:
                continuum[i] = 1e-10
        
        # Find minimum of ratio
        min_ratio = spec_subset[0] / continuum[0]
        for i in range(1, n):
            ratio = spec_subset[i] / continuum[i]
            if ratio < min_ratio:
                min_ratio = ratio
        
        return 1.0 - min_ratio

    @jit(nopython=True, parallel=True, cache=True)
    def _parallel_band_depth_numba(spec_block, wavelengths, left_idx, right_idx):
        """Numba-parallelized band depth calculation"""
        n_bands, rows, cols = spec_block.shape
        result = np.full((rows, cols), np.nan, dtype=np.float32)
        
        wl_subset = wavelengths[left_idx:right_idx+1].astype(np.float32)
        n_subset = right_idx - left_idx + 1
        
        for i in prange(rows):
            for j in range(cols):
                # Extract and smooth spectrum subset
                spec_subset = np.empty(n_subset, dtype=np.float32)
                for k in range(n_subset):
                    val = spec_block[left_idx + k, i, j]
                    if np.isnan(val):
                        val = 0.0
                    spec_subset[k] = val
                
                # Simple smoothing (moving average instead of Savgol for Numba)
                smoothed = np.empty(n_subset, dtype=np.float32)
                window = 5  # Half window
                for k in range(n_subset):
                    start = max(0, k - window)
                    end = min(n_subset, k + window + 1)
                    total = 0.0
                    count = 0
                    for m in range(start, end):
                        total += spec_subset[m]
                        count += 1
                    smoothed[k] = total / count
                
                result[i, j] = _continuum_removal_single(smoothed, wl_subset)
        
        return result


def continuum_removal_830_1130(spectrum, wavelengths):
    """Calculate continuum-removed band depth (fallback)"""
    left_wl, right_wl = 900.0, 1130.0
    left_idx = np.argmin(np.abs(wavelengths - left_wl))
    right_idx = np.argmin(np.abs(wavelengths - right_wl))
    spec_subset = spectrum[left_idx:right_idx+1]
    wl_subset = wavelengths[left_idx:right_idx+1]
    
    if np.any(np.isnan(spec_subset)) or np.any(spec_subset <= 0):
        return np.nan
    
    n = len(spec_subset)
    v_ext = np.concatenate([[0], spec_subset, [0]])
    x_ext = np.concatenate([[wl_subset[0]-1e-10], wl_subset, [wl_subset[-1]+1e-10]])
    points = np.column_stack([x_ext, v_ext])
    
    try:
        hull = ConvexHull(points)
        K = hull.vertices
        K = np.delete(K, [0, 1])
        K = np.delete(K, -1)
        K = np.sort(K) - 1
        
        if len(K) < 2 or np.max(K) >= n:
            return np.nan
        
        continuum = np.interp(np.arange(n), K, spec_subset[K])
        return 1.0 - np.min(spec_subset / np.maximum(continuum, 1e-10))
    except:
        return np.nan


# Shared memory globals for band depth workers
_shm = None
_spec = None
_out_shm = None
_output = None
_wvl = None

def _init_bd_worker(shm_name, out_name, shape, wvl):
    global _shm, _spec, _out_shm, _output, _wvl
    from multiprocessing import shared_memory
    _shm = shared_memory.SharedMemory(name=shm_name)
    _spec = np.ndarray(shape, dtype=np.float32, buffer=_shm.buf)
    _out_shm = shared_memory.SharedMemory(name=out_name)
    _output = np.ndarray((shape[1], shape[2]), dtype=np.float32, buffer=_out_shm.buf)
    _wvl = wvl

def _process_bd_row(i):
    for j in range(_spec.shape[2]):
        spectrum = np.nan_to_num(_spec[:, i, j], nan=0.0)
        smoothed = savgol_filter(spectrum, 11, 3, mode='nearest')
        _output[i, j] = continuum_removal_830_1130(smoothed, _wvl)
    return i

def parallel_band_depth(spec_block, wavelengths, chunk_rows=256):
    """Calculate band depth - uses Numba JIT if available, else shared memory fallback"""
    if HAS_NUMBA:
        left_idx = np.argmin(np.abs(wavelengths - 900.0))
        right_idx = np.argmin(np.abs(wavelengths - 1130.0))
        spec_f32 = spec_block.astype(np.float32)
        wl_f32 = wavelengths.astype(np.float32)
        rows = spec_f32.shape[1]
        result = np.full((rows, spec_f32.shape[2]), np.nan, dtype=np.float32)
        with tqdm(total=rows, desc="    band depth", unit="row", leave=False) as pbar:
            for r0 in range(0, rows, chunk_rows):
                r1 = min(r0 + chunk_rows, rows)
                result[r0:r1] = _parallel_band_depth_numba(
                    spec_f32[:, r0:r1, :], wl_f32, left_idx, right_idx
                )
                pbar.update(r1 - r0)
        return result

    from multiprocessing import Pool, shared_memory

    spec_block = spec_block.astype(np.float32)
    rows, cols = spec_block.shape[1], spec_block.shape[2]
    n_workers = os.cpu_count()
    print(f"    Using {n_workers} processes with shared memory...")

    shm = shared_memory.SharedMemory(create=True, size=spec_block.nbytes)
    shared_arr = np.ndarray(spec_block.shape, dtype=spec_block.dtype, buffer=shm.buf)
    shared_arr[:] = spec_block[:]

    out_shm = shared_memory.SharedMemory(create=True, size=rows * cols * 4)
    output = np.ndarray((rows, cols), dtype=np.float32, buffer=out_shm.buf)
    output[:] = np.nan

    try:
        with Pool(n_workers, initializer=_init_bd_worker,
                  initargs=(shm.name, out_shm.name, spec_block.shape, wavelengths)) as pool:
            list(pool.imap_unordered(_process_bd_row, range(rows)))
        result = output.copy()
    finally:
        shm.close(); shm.unlink()
        out_shm.close(); out_shm.unlink()

    return result


def process_flight_line(data_dir, flight_line, output_dir, lut_dir, wvl_path,
                       subset=None, save_diagnostics=False, n_workers=None):
    """
    Process a single flight line with optimized parallelization
    """
    print("="*70)
    print(f"ISSIA OPTIMIZED PROCESSING")
    print(f"Flight line: {flight_line}")
    print(f"Numba JIT: {'ENABLED' if HAS_NUMBA else 'DISABLED (install numba for 5-10x speedup)'}")
    print(f"CPU cores: {os.cpu_count()}")
    print("="*70)
    
    t0_total = time.time()
    
    # Load wavelengths
    wavelengths = np.load(wvl_path)
    
    # Initialize optimized processor
    processor = ISSIAProcessorOptimized(
        wavelengths=wavelengths,
        #grain_radii=np.arange(30, 10001, 30),
        grain_radii=np.arange(30, 5001, 30), 
        illumination_angles=np.arange(0, 86, 5),
        viewing_angles=np.arange(0, 86, 5),
        relative_azimuths=np.arange(0, 361, 10),
        chunk_size=(1024, 1024),
        n_workers=n_workers
    )
    
    steps = tqdm(total=9, desc="  pipeline", unit="step", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    # [0] Load lookup tables
    steps.set_description("[0] LUT loading")
    t0 = time.time()
    processor.load_lookup_tables(
        sbd_path=lut_dir / "sbd_lut.npy",
        aniso_path=lut_dir / "anisotropy_lut.npz",
        alb_path=lut_dir / "albedo_lut.npy"
    )
    steps.write(f"[0] LUT loading: {time.time()-t0:.1f}s")
    steps.update(1)

    # [1] Read ATCOR files
    steps.set_description("[1] Data loading")
    t0 = time.time()
    data = processor.read_atcor_files(data_dir, flight_line, subset=subset)
    reflectance = data['reflectance'].compute()
    global_flux = data['global_flux'].compute()
    slope = data['slope'].compute() if isinstance(data['slope'], da.Array) else data['slope']
    aspect = data['aspect'].compute() if isinstance(data['aspect'], da.Array) else data['aspect']
    steps.write(f"[1] Data loading: {time.time()-t0:.1f}s | Shape: {reflectance.shape}")
    steps.update(1)

    # [2] Masking
    steps.set_description("[2] Masking")
    t0 = time.time()
    local_illum = processor.calculate_local_illumination_angle(
        data['solar_zenith'], data['solar_azimuth'], slope, aspect
    )
    if isinstance(local_illum, da.Array):
        local_illum = local_illum.compute()

    idx_600 = np.argmin(np.abs(wavelengths - 600))
    idx_1500 = np.argmin(np.abs(wavelengths - 1500))
    ndsi = (reflectance[idx_600] - reflectance[idx_1500]) / \
           (reflectance[idx_600] + reflectance[idx_1500] + 1e-10)

    geometric_mask = (local_illum <= 85)
    snow_mask = (ndsi >= 0.87)

    idx_560 = np.argmin(np.abs(wavelengths - 560))
    shadow_ratio = reflectance[0] / (reflectance[idx_560] + 1e-10)
    shadow_mask = (shadow_ratio <= 1.0)

    final_mask = geometric_mask & snow_mask & shadow_mask
    reflectance[:, ~final_mask] = np.nan
    refl_masked = reflectance

    n_valid = np.sum(final_mask)
    steps.write(f"[2] Masking: {time.time()-t0:.1f}s | Valid pixels: {n_valid:,} ({100*n_valid/final_mask.size:.1f}%)")
    steps.update(1)

    # [3] Band depth
    steps.set_description("[3] Band depth")
    t0 = time.time()
    band_depths = parallel_band_depth(refl_masked, wavelengths)
    steps.write(f"[3] Band depth: {time.time()-t0:.1f}s")
    steps.update(1)

    # [4] Viewing geometry
    steps.set_description("[4] Viewing geometry")
    t0 = time.time()
    theta_i_eff, theta_v_eff, raa_eff = processor.calculate_local_viewing_geometry(
        data['solar_zenith'], data['solar_azimuth'],
        slope, aspect,
        viewing_zenith=0.0, viewing_azimuth=0.0
    )
    steps.write(f"[4] Viewing geometry: {time.time()-t0:.1f}s")
    steps.update(1)

    # [5] Grain size
    steps.set_description("[5] Grain size")
    t0 = time.time()
    grain_size = processor.retrieve_grain_size_pixelwise(
        band_depths, theta_i_eff, theta_v_eff, raa_eff
    )
    if isinstance(grain_size, da.Array):
        grain_size = grain_size.compute()
    steps.write(f"[5] Grain size: {time.time()-t0:.1f}s")
    steps.update(1)

    # [6] Anisotropy
    steps.set_description("[6] Anisotropy")
    t0 = time.time()
    anisotropy = processor.calculate_anisotropy_factor_pixelwise(
        grain_size, theta_i_eff, theta_v_eff, raa_eff
    )
    if isinstance(anisotropy, da.Array):
        anisotropy = anisotropy.compute()
    if anisotropy.ndim == 3:
        anisotropy_2d = anisotropy[0]
    else:
        anisotropy_2d = anisotropy
    steps.write(f"[6] Anisotropy: {time.time()-t0:.1f}s")
    steps.update(1)

    # [7+8] Albedo + Radiative forcing (single chunked pass — avoids 13 GB spectral albedo array)
    steps.set_description("[7] Albedo+RF")
    t0 = time.time()
    broadband_albedo, rf_lap = processor.compute_albedo_rf_chunked(
        refl_masked, anisotropy, global_flux, grain_size
    )
    steps.write(f"[7] Albedo: {time.time()-t0:.1f}s")
    steps.update(1)

    steps.set_description("[8] Radiative forcing")
    steps.write(f"[8] Radiative forcing: (computed with albedo above)")
    steps.update(1)
    steps.close()
    
    # Print summary
    total_time = time.time() - t0_total
    print(f"\n{'='*70}")
    print(f"COMPLETED in {total_time:.1f}s")
    print(f"Mean GS: {np.nanmean(grain_size):.1f} μm")
    print(f"Mean Albedo: {np.nanmean(broadband_albedo):.3f}")
    print(f"Mean RF: {np.nanmean(rf_lap):.2f} W/m²")
    print(f"{'='*70}")
    
    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    processor._save_geotiff(grain_size, output_dir / f"{flight_line}_gs.tif", 
                           data['transform'], data['crs'])
    processor._save_geotiff(broadband_albedo, output_dir / f"{flight_line}_albedo.tif", 
                           data['transform'], data['crs'])
    processor._save_geotiff(rf_lap, output_dir / f"{flight_line}_rf.tif", 
                           data['transform'], data['crs'])
    
    if save_diagnostics:
        print("\nSaving diagnostics...")
        processor._save_geotiff(slope, output_dir / f"{flight_line}_slope.tif", 
                               data['transform'], data['crs'])
        processor._save_geotiff(theta_i_eff, output_dir / f"{flight_line}_theta_i_eff.tif", 
                               data['transform'], data['crs'])
        processor._save_geotiff(theta_v_eff, output_dir / f"{flight_line}_theta_v_eff.tif", 
                               data['transform'], data['crs'])
        processor._save_geotiff(band_depths, output_dir / f"{flight_line}_band_depth.tif", 
                               data['transform'], data['crs'])
    
    print(f"\nResults saved to {output_dir}/")
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description='ISSIA Optimized Processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--flight-line', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--lut-dir', type=str, default='luts')
    parser.add_argument('--wvl-path', type=str, default='wvl.npy')
    parser.add_argument('--subset', type=str, default=None,
                       help='Spatial subset as "ymin,ymax,xmin,xmax"')
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker threads')
    
    args = parser.parse_args()
    
    subset = None
    if args.subset:
        subset = tuple(map(int, args.subset.split(',')))
    
    process_flight_line(
        data_dir=Path(args.data_dir),
        flight_line=args.flight_line,
        output_dir=Path(args.output_dir),
        lut_dir=Path(args.lut_dir),
        wvl_path=Path(args.wvl_path),
        subset=subset,
        save_diagnostics=args.diagnostics,
        n_workers=args.workers
    )


if __name__ == "__main__":
    main()
