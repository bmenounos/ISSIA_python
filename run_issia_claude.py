#!/usr/bin/env python
import numpy as np
import os
from pathlib import Path
import warnings
import time
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter
import dask.array as da
from issia_notebook import ISSIAProcessorNotebook

warnings.filterwarnings('ignore')

def continuum_removal_830_1130(spectrum, wavelengths):
    smoothed = savgol_filter(spectrum, 11, 3, mode='nearest')
    left_wl, right_wl = 830.0, 1130.0
    left_idx = np.argmin(np.abs(wavelengths - left_wl))
    right_idx = np.argmin(np.abs(wavelengths - right_wl))
    spec_subset = smoothed[left_idx:right_idx+1]
    wl_subset = wavelengths[left_idx:right_idx+1]
    
    if np.any(np.isnan(spec_subset)) or np.any(spec_subset <= 0):
        return np.nan
    
    n = len(spec_subset)
    points = np.column_stack([
        np.concatenate([[wl_subset[0] - 1e-10], wl_subset, [wl_subset[-1] + 1e-10]]),
        np.concatenate([[0], spec_subset, [0]])
    ])
    
    try:
        hull = ConvexHull(points)
        vertices = np.sort(hull.vertices[(hull.vertices > 0) & (hull.vertices <= n)]) - 1
        continuum = np.interp(np.arange(n), vertices, spec_subset[vertices])
        return 1.0 - np.min(spec_subset / np.maximum(continuum, 1e-10))
    except:
        return np.nan

def parallel_band_depth(spec_block, wavelengths):
    rows, cols = spec_block.shape[1], spec_block.shape[2]
    result = np.zeros((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            spectrum = np.nan_to_num(spec_block[:, i, j], nan=0.0)
            smoothed = savgol_filter(spectrum, 7, 2, mode='nearest')
            result[i, j] = continuum_removal_830_1130(smoothed, wavelengths)
    return result

if __name__ == "__main__":
    print("="*70)
    print("FIXED PIPELINE")
    print("="*70)

    wavelengths = np.load('wvl.npy')
    processor = ISSIAProcessorNotebook(
        wavelengths=wavelengths,
        grain_radii=np.arange(30, 5001, 30),
        illumination_angles=np.arange(0, 86, 5),
        viewing_angles=np.arange(0, 86, 5),
        relative_azimuths=np.arange(0, 361, 10),
        chunk_size=(1024, 1024)
    )

    processor.load_lookup_tables(
        sbd_path=Path("luts/sbd_lut.npy"),
        aniso_path=Path("luts/anisotropy_lut.npy"),
        alb_path=Path("luts/albedo_lut.npy")
    )

    data_dir = Path('data')
    flight_line = '24_4012_05_2024-06-06_17-54-38-rect_img'
    
    t0_total = time.time()
    
    t0 = time.time()
    data = processor.read_atcor_files(data_dir, flight_line, subset=(1500, 1700, 500, 700))
    local_illum = processor.calculate_local_illumination_angle(
        data['solar_zenith'], data['solar_azimuth'], data['slope'], data['aspect']
    )
    idx_600 = np.argmin(np.abs(wavelengths - 600))
    idx_1500 = np.argmin(np.abs(wavelengths - 1500))
    ndsi = (data['reflectance'][idx_600] - data['reflectance'][idx_1500]) / \
           (data['reflectance'][idx_600] + data['reflectance'][idx_1500])
    refl_masked = da.where(ndsi >= 0.4, data['reflectance'], np.nan)
    print(f"[1] Setup: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    band_depths = da.map_blocks(
        parallel_band_depth, refl_masked, wavelengths,
        dtype=float, drop_axis=0, chunks=data['reflectance'].chunks[1:]
    ).persist()
    print(f"[2] Band depth: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    grain_size = processor.retrieve_grain_size(band_depths, local_illum, 0.0, 0.0)
    anisotropy = processor.calculate_anisotropy_factor(grain_size, local_illum, 0.0, 0.0)
    
    # Fix anisotropy shape bug - take first band only
    if isinstance(anisotropy, da.Array) and anisotropy.ndim == 3:
        anisotropy = anisotropy[0]
    elif isinstance(anisotropy, np.ndarray) and anisotropy.ndim == 3:
        anisotropy = da.from_array(anisotropy[0], chunks=anisotropy.shape[1:])
    
    print(f"[3] Grain/Aniso: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    anisotropy_3d = anisotropy[np.newaxis, :, :]
    
    # Don't pre-multiply by anisotropy - calculate_spectral_albedo does it internally
    spectral_albedo = processor.calculate_spectral_albedo(grain_size, refl_masked, anisotropy)
    mean_flux = da.nanmean(data['global_flux'], axis=(1, 2)).compute()
    broadband_albedo = processor.calculate_broadband_albedo(spectral_albedo, mean_flux)
    
    # Use pixel-wise flux RF calculation (matches MATLAB implementation)
    rf_lap = processor.calculate_radiative_forcing(grain_size, spectral_albedo, data['global_flux'])
    
    # Compute final results
    if isinstance(grain_size, da.Array):
        grain_size = grain_size.compute()
    if isinstance(broadband_albedo, da.Array):
        broadband_albedo = broadband_albedo.compute()
    if isinstance(rf_lap, da.Array):
        rf_lap = rf_lap.compute()
    
    print(f"[4] Albedo/RF: {time.time()-t0:.1f}s")
    
    print(f"\n{'='*70}")
    print(f"✅ TOTAL: {time.time()-t0_total:.1f}s")
    print(f"✅ Mean GS: {np.nanmean(grain_size):.1f} μm")
    print(f"✅ Mean Albedo: {np.nanmean(broadband_albedo):.3f}")
    print(f"✅ Mean RF: {np.nanmean(rf_lap):.2f} W/m²")
    print(f"✅ Albedo range: [{np.nanmin(broadband_albedo):.3f}, {np.nanmax(broadband_albedo):.3f}]")
    print(f"{'='*70}")
    
    out = Path("issia_results_optimized")
    out.mkdir(parents=True, exist_ok=True)
    
    processor._save_geotiff(grain_size, out / f"{flight_line}_gs.tif", data['transform'], data['crs'])
    processor._save_geotiff(broadband_albedo, out / f"{flight_line}_albedo.tif", data['transform'], data['crs'])
    processor._save_geotiff(rf_lap, out / f"{flight_line}_rf.tif", data['transform'], data['crs'])
    
    print(f"\n✅ Results saved to {out}/")
