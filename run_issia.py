#!/usr/bin/env python
# coding: utf-8

"""
ISSIA Processing - FINAL VERSION WITH 830-1130 nm

- Explicit 830-1130 nm wavelength range (MATLAB standard)
- NDSI threshold: 0.4
- Uses scipy ConvexHull (proven to work)
- Fast, no validation overhead
"""

import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from issia_notebook import ISSIAProcessorNotebook
import dask
import dask.array as da
import time
from scipy.spatial import ConvexHull

# ============================================================================
# CONTINUUM REMOVAL: 830-1130 nm (MATLAB standard)
# ============================================================================

def continuum_removal_830_1130(spectrum, wavelengths):
    """
    Continuum removal using 830-1130 nm range (MATLAB standard)
    Uses scipy ConvexHull for reliability
    """
    # Find wavelength indices for 830-1130 nm
    left_wl = 830.0
    right_wl = 1130.0
    
    left_idx = np.argmin(np.abs(wavelengths - left_wl))
    right_idx = np.argmin(np.abs(wavelengths - right_wl))
    
    # Extract spectral subset
    spec_subset = spectrum[left_idx:right_idx+1]
    wl_subset = wavelengths[left_idx:right_idx+1]
    
    # Check for invalid data
    if np.any(np.isnan(spec_subset)) or np.any(spec_subset <= 0):
        return np.nan
    
    # Create points for convex hull (extended with zeros at edges)
    n = len(spec_subset)
    points = np.column_stack([
        np.concatenate([[wl_subset[0] - 1e-10], wl_subset, [wl_subset[-1] + 1e-10]]),
        np.concatenate([[0], spec_subset, [0]])
    ])
    
    try:
        # Compute convex hull
        hull = ConvexHull(points)
        
        # Get upper hull vertices (exclude the zero endpoints)
        hull_vertices = hull.vertices
        hull_vertices = hull_vertices[(hull_vertices > 0) & (hull_vertices <= n)]
        hull_vertices = np.sort(hull_vertices) - 1  # Adjust for zero point
        
        if len(hull_vertices) < 2:
            return np.nan
        
        # Interpolate continuum
        continuum = np.interp(
            np.arange(n),
            hull_vertices,
            spec_subset[hull_vertices]
        )
        
        # Avoid division by zero
        continuum = np.maximum(continuum, 1e-10)
        
        # Continuum removal
        continuum_removed = spec_subset / continuum
        
        # Band depth = 1 - minimum of continuum-removed spectrum
        band_depth = 1.0 - np.min(continuum_removed)
        band_depth = max(0.0, band_depth)
        
        return band_depth
        
    except Exception as e:
        # If convex hull fails, return NaN
        return np.nan


# ============================================================================
# Setup
# ============================================================================

cpu_count = os.cpu_count()
num_workers = max(1, int(cpu_count * 0.75))

print("="*70)
print("🎯 ISSIA - FINAL VERSION")
print("="*70)
print(f"\n💻 Hardware: {cpu_count} cores, {num_workers} workers")

dask.config.set({
    'scheduler': 'threads',
    'num_workers': num_workers,
    'array.chunk-cache-size': '2GB',
})

print("\n✅ Configuration complete!")
print("="*70)

# ============================================================================
# Initialize Processor
# ============================================================================

wavelengths = np.load('wvl.npy')
grain_radii = np.arange(30, 5001, 30)
illumination_angles = np.arange(0, 86, 5)
viewing_angles = np.arange(0, 86, 5)
relative_azimuths = np.arange(0, 361, 10)
chunk_size = (1024, 1024)

processor = ISSIAProcessorNotebook(
    wavelengths=wavelengths,
    grain_radii=grain_radii,
    illumination_angles=illumination_angles,
    viewing_angles=viewing_angles,
    relative_azimuths=relative_azimuths,
    coord_ref_sys_code=32610,
    chunk_size=chunk_size,
    verbose=False
)

print("\n✅ Processor initialized")

# ============================================================================
# Load LUTs
# ============================================================================

lut_dir = Path("lookup_tables_fixed")
if not lut_dir.exists():
    lut_dir = Path("lookup_tables_fixed")

processor.load_lookup_tables(
    sbd_lut_path=lut_dir / "sbd_lut.npy",
    anisotropy_lut_path=lut_dir / "anisotropy_lut.npy",
    albedo_lut_path=lut_dir / "albedo_lut.npy"
)

print(f"✅ Lookup tables loaded from: {lut_dir}")

# ============================================================================
# Process Data
# ============================================================================

data_dir = Path('/Users/menounos/Desktop/issia_local')
flight_line = '24_4012_05_2024-06-06_17-54-38-rect_img'
output_dir = Path("issia_results_final")

print("\n" + "#"*70)
print("# PROCESSING")
print("#"*70)
print("\n🎯 Wavelength range: 830-1130 nm (MATLAB standard - for finer grains)")
print("🎯 NDSI threshold: 0.4")
print("🚀 Using scipy ConvexHull (proven reliable)\n")

overall_start = time.time()

# Load data
print("Loading data (full image)...")
data = processor.read_atcor_files(data_dir, flight_line, subset=None)

reflectance = data['reflectance']
global_flux = data['global_flux']
slope = data['slope']
aspect = data['aspect']
solar_zenith = data['solar_zenith']
solar_azimuth = data['solar_azimuth']
transform = data['transform']
crs = data['crs']

n_bands, n_rows, n_cols = reflectance.shape
print(f"✓ Shape: {n_rows}×{n_cols} pixels, {n_bands} bands")

# Calculate local illumination
print("\nStep 1/6: Calculating local illumination angles...")
local_illum = processor.calculate_local_illumination_angle(
    solar_zenith, solar_azimuth, slope, aspect
)
print("  ✓ Completed")

# Apply NDSI mask
print("\nApplying NDSI snow/ice mask (threshold = 0.4)...")
idx_600 = np.argmin(np.abs(processor.wavelengths - 600))
idx_1500 = np.argmin(np.abs(processor.wavelengths - 1500))
r_600 = reflectance[idx_600, :, :]
r_1500 = reflectance[idx_1500, :, :]
ndsi = (r_600 - r_1500) / (r_600 + r_1500)

snow_mask = ndsi >= 0.4
reflectance = da.where(snow_mask, reflectance, np.nan)
print("  ✓ Completed")

# Band depth calculation with explicit 830-1130 nm
print("\nStep 2/6: Calculating scaled band depths (830-1130 nm)...")

def compute_band_depths_830_1130(spec_block, wavelengths_arr):
    """Compute band depths using 830-1130 nm range"""
    rows, cols = spec_block.shape[1], spec_block.shape[2]
    result = np.zeros((rows, cols), dtype=float)
    
    for i in range(rows):
        for j in range(cols):
            spectrum = spec_block[:, i, j]
            bd = continuum_removal_830_1130(spectrum, wavelengths_arr)
            result[i, j] = bd
    
    return result

band_depths = da.map_blocks(
    lambda spec_block: compute_band_depths_830_1130(spec_block, processor.wavelengths),
    reflectance,
    dtype=float,
    drop_axis=0,
    chunks=(processor.chunk_size[0], processor.chunk_size[1])
)

print("  ✓ Completed")

# Grain size retrieval
print("\nStep 3/6: Retrieving grain sizes...")
grain_size = processor.retrieve_grain_size(band_depths, local_illum, 0.0, 0.0)
print("  ✓ Completed")

# Anisotropy
print("\nStep 4/6: Calculating anisotropy factors...")
anisotropy = processor.calculate_anisotropy_factor(grain_size, local_illum, 0.0, 0.0)

if reflectance.shape != anisotropy.shape:
    _, n_rows, n_cols = anisotropy.shape
    reflectance = reflectance[:, :n_rows, :n_cols]
    global_flux = global_flux[:, :n_rows, :n_cols]

reflectance_hdrf = reflectance * anisotropy
print("  ✓ Completed")

# Spectral albedo
print("\nStep 5/6: Calculating spectral albedo...")
spectral_albedo = processor.calculate_spectral_albedo(grain_size, reflectance_hdrf, anisotropy)
mean_flux = da.nanmean(global_flux, axis=(1, 2)).compute()
broadband_albedo = processor.calculate_broadband_albedo(spectral_albedo, mean_flux)
print("  ✓ Completed")

# Radiative forcing
print("\nStep 6/6: Calculating radiative forcing...")
clean_albedo = processor.calculate_spectral_albedo(
    grain_size, reflectance_hdrf * 0 + anisotropy, anisotropy
)
rf_lap = processor.calculate_radiative_forcing(clean_albedo, spectral_albedo, mean_flux)
print("  ✓ Completed")

# Compute results
print("\n" + "="*70)
print("COMPUTING RESULTS")
print("="*70)

from dask.diagnostics import ProgressBar

print("\n[1/3] Computing grain size...")
with ProgressBar():
    grain_size_result = grain_size.compute()

print("[2/3] Computing broadband albedo...")
with ProgressBar():
    broadband_albedo_result = broadband_albedo.compute()

print("[3/3] Computing radiative forcing...")
with ProgressBar():
    rf_lap_result = rf_lap.compute()

# Statistics
print("\n" + "="*70)
print("FINAL STATISTICS")
print("="*70)

valid_gs = np.sum(~np.isnan(grain_size_result))
total_pixels = grain_size_result.size

print(f"\nValid pixels: {valid_gs}/{total_pixels} ({100*valid_gs/total_pixels:.1f}%)")

print(f"\n📊 Grain Size:")
gs_mean = np.nanmean(grain_size_result)
print(f"  Mean: {gs_mean:.1f} μm")
print(f"  Median: {np.nanmedian(grain_size_result):.1f} μm")
print(f"  Std: {np.nanstd(grain_size_result):.1f} μm")
print(f"  Range: {np.nanmin(grain_size_result):.0f} - {np.nanmax(grain_size_result):.0f} μm")

print(f"\n💡 Target: MATLAB ~368-410 μm")
print(f"   This version (830-1130 nm): {gs_mean:.1f} μm")

print(f"\n📊 Broadband Albedo:")
print(f"  Mean: {np.nanmean(broadband_albedo_result):.3f}")
print(f"  Range: {np.nanmin(broadband_albedo_result):.3f} - {np.nanmax(broadband_albedo_result):.3f}")

print(f"\n📊 Radiative Forcing:")
print(f"  Mean: {np.nanmean(rf_lap_result):.1f} W/m²")
print(f"  Range: {np.nanmin(rf_lap_result):.1f} - {np.nanmax(rf_lap_result):.1f} W/m²")

# Save outputs
print("\n" + "="*70)
print("SAVING OUTPUTS")
print("="*70)

output_dir.mkdir(parents=True, exist_ok=True)

grain_size_file = output_dir / f"{flight_line}_grain_size.tif"
processor._save_geotiff(grain_size_result, grain_size_file, transform, crs)

albedo_file = output_dir / f"{flight_line}_broadband_albedo.tif"
processor._save_geotiff(broadband_albedo_result, albedo_file, transform, crs)

rf_file = output_dir / f"{flight_line}_radiative_forcing.tif"
processor._save_geotiff(rf_lap_result, rf_file, transform, crs)

overall_time = time.time() - overall_start

print(f"\n✓ Files saved to: {output_dir}/")
print(f"✅ PROCESSING COMPLETE in {overall_time/60:.1f} minutes")

print("\n" + "="*70)

print("="*70)