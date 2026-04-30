#!/usr/bin/env python
"""
ISSIA - Single Flight Line Processing (chunked, constant-memory)

All data is read and processed in spatial row-strips so that peak RAM stays
constant regardless of flight-line size.  Only one strip (~chunk_rows rows) of
reflectance and flux is live at a time; results are written incrementally.

Usage:
    python run_issia.py --data-dir /path/to/data --flight-line flight_id --output-dir /path/to/output
    python run_issia.py ... --chunk-rows 128   # reduce for very tight RAM
"""

import contextlib
import json
import numpy as np
import rasterio
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


def _read_hdr_scale_factor(dat_path):
    """Return the scale factor from an ENVI .hdr, defaulting to 1.0."""
    hdr = str(dat_path).replace('.dat', '.hdr')
    try:
        with open(hdr, 'r') as f:
            for line in f:
                if 'reflectance scale factor' in line.lower():
                    return float(line.split('=')[1].strip())
    except (FileNotFoundError, OSError, ValueError):
        pass
    return 1.0


def _create_output_tiff(path, height, width, transform, crs):
    """Open a single-band float32 GeoTIFF for incremental windowed writing."""
    return rasterio.open(
        str(path), 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='float32',
        crs=crs,
        transform=transform,
        compress='lzw',
        nodata=float('nan'),
    )


_CONFIG_DEFAULTS = {
    "ndsi_threshold":         0.87,
    "shadow_ratio_threshold": 1.0,
    "solar_zenith_max":       85.0,
    "chunk_rows":             256,
}

def load_config(path=None):
    """Load issia_config.json, returning merged defaults + file values.

    Search order: explicit path → ./issia_config.json → defaults only.
    """
    cfg = dict(_CONFIG_DEFAULTS)
    candidates = [Path(path)] if path else [Path("issia_config.json")]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                cfg.update(json.load(f))
            print(f"Config loaded from {p}")
            break
    return cfg


def process_flight_line(data_dir, flight_line, output_dir, lut_dir, wvl_path,
                        subset=None, save_diagnostics=False, n_workers=None,
                        chunk_rows=256, ndsi_threshold=0.87,
                        shadow_ratio_threshold=1.0, solar_zenith_max=85.0):
    """
    Process a single flight line with constant peak-memory chunked I/O.

    Each iteration reads `chunk_rows` rows from every source file, runs the
    full retrieval pipeline on that strip, writes the results, then discards
    the strip.  Peak RAM ≈ chunk_rows × n_cols × n_bands × ~3 arrays, so a
    25 GB flight line with chunk_rows=256 typically needs < 3 GB at peak.
    """
    t0_total = time.time()
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    lut_dir    = Path(lut_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"ISSIA CHUNKED PROCESSING  |  flight line: {flight_line}")
    if HAS_NUMBA:
        import numba
        numba_threads = numba.get_num_threads()
        try:
            from numba import threading_layer
            tl = threading_layer()
        except Exception:
            tl = "unknown"
    else:
        numba_threads, tl = 0, "n/a"
    print(f"Numba JIT: {'ENABLED' if HAS_NUMBA else 'DISABLED'}  |  "
          f"cores: {os.cpu_count()}  |  numba_threads: {numba_threads}  |  "
          f"threading_layer: {tl}  |  chunk_rows: {chunk_rows}")
    print(f"Thresholds  |  NDSI >= {ndsi_threshold}  |  "
          f"shadow_ratio <= {shadow_ratio_threshold}  |  "
          f"solar_zenith_max: {solar_zenith_max}")
    print("=" * 70)

    wavelengths = np.load(wvl_path)

    # [0] Processor + LUTs — loaded once, reused across all chunks
    print("[0] Loading LUTs...")
    t0 = time.time()
    processor = ISSIAProcessorOptimized(
        wavelengths=wavelengths,
        grain_radii=np.arange(30, 5001, 30),
        illumination_angles=np.arange(0, 86, 5),
        viewing_angles=np.arange(0, 86, 5),
        relative_azimuths=np.arange(0, 361, 10),
        chunk_size=(chunk_rows, chunk_rows),
        n_workers=n_workers,
    )
    processor.load_lookup_tables(
        sbd_path=lut_dir / "sbd_lut.npy",
        aniso_path=lut_dir / "anisotropy_lut.npz",
        alb_path=lut_dir / "albedo_lut.npy",
    )
    print(f"[0] LUTs loaded in {time.time()-t0:.1f}s")

    # Source file paths
    atm_path = data_dir / f"{flight_line}_atm.dat"
    eglo_path = data_dir / f"{flight_line}_eglo.dat"
    slp_path  = data_dir / f"{flight_line}_slp.dat"
    asp_path  = data_dir / f"{flight_line}_asp.dat"
    inn_path  = data_dir / f"{flight_line}.inn"

    atm_scale  = _read_hdr_scale_factor(atm_path)
    eglo_scale = _read_hdr_scale_factor(eglo_path)
    solar_zenith, solar_azimuth = processor._read_solar_zenith(inn_path)

    # Band indices computed once
    idx_600  = int(np.argmin(np.abs(wavelengths - 600)))
    idx_1500 = int(np.argmin(np.abs(wavelengths - 1500)))
    idx_560  = int(np.argmin(np.abs(wavelengths - 560)))

    # Raster geometry
    with rasterio.open(str(atm_path)) as src:
        src_height, src_width = src.height, src.width
        src_transform = src.transform
        crs = src.crs
        if subset:
            ymin, ymax, xmin, xmax = subset
            col_off, row_off = xmin, ymin
            height, width = ymax - ymin, xmax - xmin
            out_transform = src.window_transform(
                rasterio.windows.Window(col_off, row_off, width, height))
        else:
            col_off, row_off = 0, 0
            height, width = src_height, src_width
            out_transform = src_transform

    n_chunks = (height + chunk_rows - 1) // chunk_rows
    print(f"[1] Raster: {height} rows × {width} cols | {n_chunks} chunks")

    # Output file paths
    gs_path  = output_dir / f"{flight_line}_gs.tif"
    alb_path = output_dir / f"{flight_line}_albedo.tif"
    rf_path  = output_dir / f"{flight_line}_rf.tif"
    diag_keys = ['slope', 'theta_i', 'theta_v', 'bd'] if save_diagnostics else []
    diag_paths = {
        'slope':  output_dir / f"{flight_line}_slope.tif",
        'theta_i': output_dir / f"{flight_line}_theta_i_eff.tif",
        'theta_v': output_dir / f"{flight_line}_theta_v_eff.tif",
        'bd':     output_dir / f"{flight_line}_band_depth.tif",
    }

    # Accumulators for end-of-run summary (weighted by valid pixel count)
    sum_gs = sum_alb = sum_rf = 0.0
    n_valid_total = 0

    mk = lambda p: _create_output_tiff(p, height, width, out_transform, crs)

    with contextlib.ExitStack() as stack:
        # Source rasters (read-only, kept open for windowed reads)
        src_atm  = stack.enter_context(rasterio.open(str(atm_path)))
        src_eglo = stack.enter_context(rasterio.open(str(eglo_path)))
        src_slp  = stack.enter_context(rasterio.open(str(slp_path)))
        src_asp  = stack.enter_context(rasterio.open(str(asp_path)))

        # Output rasters (write-only, supports windowed writes)
        dst_gs   = stack.enter_context(mk(gs_path))
        dst_alb  = stack.enter_context(mk(alb_path))
        dst_rf   = stack.enter_context(mk(rf_path))
        if save_diagnostics:
            dst_diag = {k: stack.enter_context(mk(diag_paths[k])) for k in diag_keys}

        for r0 in tqdm(range(0, height, chunk_rows), desc="chunks", unit="chunk"):
            r1      = min(r0 + chunk_rows, height)
            chunk_h = r1 - r0

            # Windows: src_win maps into the original file, out_win into the output
            src_win = rasterio.windows.Window(col_off, row_off + r0, width, chunk_h)
            out_win = rasterio.windows.Window(0, r0, width, chunk_h)

            # --- Read one strip ---
            refl   = src_atm.read(window=src_win).astype(np.float32)  / atm_scale
            flux   = src_eglo.read(window=src_win).astype(np.float32) / eglo_scale
            slope  = src_slp.read(1, window=src_win).astype(np.float32)
            aspect = src_asp.read(1, window=src_win).astype(np.float32)

            # Zero is the ENVI no-data sentinel
            refl = np.where(refl == 0, np.nan, refl)
            flux = np.where(flux == 0, np.nan, flux)

            # --- Viewing geometry (pure numpy, fast) ---
            theta_i_eff, theta_v_eff, raa_eff = \
                processor.calculate_local_viewing_geometry(
                    solar_zenith, solar_azimuth, slope, aspect,
                    viewing_zenith=0.0, viewing_azimuth=0.0)

            # --- Masking ---
            ndsi = ((refl[idx_600] - refl[idx_1500]) /
                    (refl[idx_600] + refl[idx_1500] + 1e-10))
            shadow_ratio = refl[0] / (refl[idx_560] + 1e-10)
            final_mask = (theta_i_eff <= solar_zenith_max) & (ndsi >= ndsi_threshold) & (shadow_ratio <= shadow_ratio_threshold)

            refl[:, ~final_mask] = np.nan
            n_valid = int(np.sum(final_mask))

            # NaN tile used when the entire strip is invalid
            nan_tile = np.full((1, chunk_h, width), np.nan, dtype=np.float32)

            if n_valid == 0:
                dst_gs.write(nan_tile,  window=out_win)
                dst_alb.write(nan_tile, window=out_win)
                dst_rf.write(nan_tile,  window=out_win)
                if save_diagnostics:
                    dst_diag['slope'].write(slope[np.newaxis],          window=out_win)
                    dst_diag['theta_i'].write(theta_i_eff[np.newaxis],  window=out_win)
                    dst_diag['theta_v'].write(theta_v_eff[np.newaxis],  window=out_win)
                    dst_diag['bd'].write(nan_tile,                       window=out_win)
                continue

            # --- Band depth (Numba-accelerated, internally row-chunked) ---
            band_depths = parallel_band_depth(refl, wavelengths)

            # --- Grain size ---
            grain_size = processor.retrieve_grain_size_pixelwise(
                band_depths, theta_i_eff, theta_v_eff, raa_eff)
            if isinstance(grain_size, da.Array):
                grain_size = grain_size.compute()

            # --- Anisotropy (n_wvl × chunk_h × width) ---
            anisotropy = processor.calculate_anisotropy_factor_pixelwise(
                grain_size, theta_i_eff, theta_v_eff, raa_eff)
            if isinstance(anisotropy, da.Array):
                anisotropy = anisotropy.compute()

            # --- Broadband albedo + RF (inner column-chunked loop, O(chunk) RAM) ---
            broadband_albedo, rf_lap = processor.compute_albedo_rf_chunked(
                refl, anisotropy, flux, grain_size)

            # --- Write this strip's results immediately ---
            dst_gs.write(grain_size[np.newaxis],      window=out_win)
            dst_alb.write(broadband_albedo[np.newaxis], window=out_win)
            dst_rf.write(rf_lap[np.newaxis],            window=out_win)

            if save_diagnostics:
                dst_diag['slope'].write(slope[np.newaxis],         window=out_win)
                dst_diag['theta_i'].write(theta_i_eff[np.newaxis], window=out_win)
                dst_diag['theta_v'].write(theta_v_eff[np.newaxis], window=out_win)
                dst_diag['bd'].write(band_depths[np.newaxis],      window=out_win)

            # Accumulate stats (nansum avoids counting NaN pixels twice)
            sum_gs  += float(np.nansum(grain_size))
            sum_alb += float(np.nansum(broadband_albedo))
            sum_rf  += float(np.nansum(rf_lap))
            n_valid_total += n_valid

            # Explicitly release the strip — Python GC may not be fast enough
            del refl, flux, slope, aspect, band_depths, grain_size, anisotropy
            del broadband_albedo, rf_lap

    # All rasterio files are now closed via ExitStack
    total_time = time.time() - t0_total
    print(f"\n{'='*70}")
    print(f"COMPLETED in {total_time:.1f}s")
    if n_valid_total > 0:
        print(f"Mean GS:     {sum_gs  / n_valid_total:.1f} μm")
        print(f"Mean Albedo: {sum_alb / n_valid_total:.3f}")
        print(f"Mean RF:     {sum_rf  / n_valid_total:.2f} W/m²")
    else:
        print("No valid (snow) pixels found in this flight line.")
    print(f"{'='*70}")

    output_files = {
        'gs':     str(gs_path),
        'albedo': str(alb_path),
        'rf':     str(rf_path),
    }
    if save_diagnostics:
        output_files.update({k: str(diag_paths[k]) for k in diag_keys})

    print(f"Results saved to {output_dir}/")
    return output_files


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description='ISSIA Optimized Processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--flight-line', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--lut-dir', type=str, default='luts')
    parser.add_argument('--wvl-path', type=str, default='wvl.npy')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to issia_config.json (default: ./issia_config.json)')
    parser.add_argument('--subset', type=str, default=None,
                       help='Spatial subset as "ymin,ymax,xmin,xmax"')
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker threads')
    parser.add_argument('--chunk-rows', type=int, default=None,
                       help='Rows per processing chunk (overrides config, default 256)')
    parser.add_argument('--ndsi-threshold', type=float, default=None,
                       help='Minimum NDSI to include pixel (overrides config, default 0.87)')
    parser.add_argument('--shadow-ratio-threshold', type=float, default=None,
                       help='Maximum shadow ratio to include pixel (overrides config, default 1.0)')
    parser.add_argument('--solar-zenith-max', type=float, default=None,
                       help='Maximum solar zenith angle in degrees (overrides config, default 85)')

    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)

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
        n_workers=args.workers,
        chunk_rows=args.chunk_rows             if args.chunk_rows             is not None else cfg['chunk_rows'],
        ndsi_threshold=args.ndsi_threshold     if args.ndsi_threshold         is not None else cfg['ndsi_threshold'],
        shadow_ratio_threshold=args.shadow_ratio_threshold if args.shadow_ratio_threshold is not None else cfg['shadow_ratio_threshold'],
        solar_zenith_max=args.solar_zenith_max if args.solar_zenith_max       is not None else cfg['solar_zenith_max'],
    )


if __name__ == "__main__":
    main()
