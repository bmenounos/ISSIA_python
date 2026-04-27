#!/usr/bin/env python
"""
ISSIA SNICAR Lookup Table Generator

Generates lookup tables using SNICAR radiative transfer model for clean snow:
1. SBD (Scaled Band Depth) LUT - maps band depth to grain size
2. Albedo LUT - clean snow spectral albedo vs grain size
3. Anisotropy LUT - BRDF correction factors

Requires: PySNICAR (https://github.com/maximlamare/pySNICAR)

Usage:
    python generate_snicar_luts.py --wvl wvl.npy --output-dir luts_snicar

Notes:
    - Generation takes longer than ART (~2-4 hours depending on CPU count)
    - Uses SNICAR for spectral albedo computation
    - BRDF approximation: SNICAR albedo + empirical angular correction
"""

import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull
from tqdm import tqdm
import multiprocessing as mp
import itertools
import argparse
import os
import sys

# Check for BioSNICAR
try:
    # Try importing BioSNICAR
    import sys
    biosnicar_path = Path.home() / 'biosnicar-py'
    if biosnicar_path.exists():
        sys.path.insert(0, str(biosnicar_path))
    from biosnicar import snicar_feeder, setup_snicar
    SNICAR_AVAILABLE = True
    print("✓ BioSNICAR found")
except ImportError:
    SNICAR_AVAILABLE = False
    print("⚠ BioSNICAR not found")
    print("  Install: git clone https://github.com/jmcook1186/biosnicar-py.git ~/biosnicar-py")
    print("  Then: pip install -e ~/biosnicar-py")
    print("  Falling back to parameterized approximation...")


def calculate_band_depth_matlab(spectrum, wavelengths):
    """
    Calculate continuum-removed band depth - MATLAB matched
    
    Same algorithm as ART version for consistency.
    """
    idx_start = np.argmin(np.abs(wavelengths - 900))
    idx_end = np.argmin(np.abs(wavelengths - 1130))
    spec_sub = spectrum[idx_start:idx_end+1]
    wl_sub = wavelengths[idx_start:idx_end+1]
    
    n = len(spec_sub)
    v_ext = np.concatenate([[0], spec_sub, [0]])
    x_ext = np.concatenate([[wl_sub[0]-1e-10], wl_sub, [wl_sub[-1]+1e-10]])
    points = np.column_stack([x_ext, v_ext])
    
    try:
        hull = ConvexHull(points)
        K = hull.vertices
        K = np.delete(K, [0, 1])
        K = np.delete(K, -1)
        K = np.sort(K) - 1
        
        if len(K) < 2 or np.max(K) >= n:
            return np.nan
        
        continuum = np.interp(np.arange(n), K, spec_sub[K])
        return 1.0 - np.min(spec_sub / np.maximum(continuum, 1e-10))
    except:
        return np.nan


def snicar_directional_albedo(grain, illum, wvl, layer_depth=10.0, density=300.0):
    """
    Calculate directional-hemispherical albedo using SNICAR
    
    Parameters
    ----------
    grain : float
        Grain radius in micrometers
    illum : float
        Solar zenith angle in degrees
    wvl : np.ndarray
        Wavelength array in nm
    layer_depth : float
        Snow layer depth in meters (default 10m = semi-infinite)
    density : float
        Snow density in kg/m³
        
    Returns
    -------
    np.ndarray
        Spectral albedo at each wavelength
    """
    if SNICAR_AVAILABLE:
        # Use BioSNICAR
        try:
            # BioSNICAR configuration for clean snow
            config = {
                'grain_rds': grain,  # µm
                'layer_type': [1],  # 1 = snow
                'dz': [layer_depth],  # layer thickness in m
                'rho_layers': [density],  # kg/m³
                'grain_shp': [1],  # 1 = sphere
                
                # Illumination
                'solzen': illum,  # degrees
                'incoming_i': 1,  # direct irradiance
                
                # Clean snow: no impurities
                'mss_cnc_soot1': [0],
                'mss_cnc_soot2': [0],
                'mss_cnc_dust1': [0],
                'mss_cnc_dust2': [0],
                'mss_cnc_dust3': [0],
                'mss_cnc_dust4': [0],
                'mss_cnc_ash1': [0],
                'mss_cnc_GRISdust1': [0],
                'mss_cnc_GRISdust2': [0],
                'mss_cnc_GRISdust3': [0],
                'mss_cnc_GRISdustP1': [0],
                'mss_cnc_GRISdustP2': [0],
                'mss_cnc_GRISdustP3': [0],
                'mss_cnc_snw_alg': [0],
                'mss_cnc_glacier_algae1': [0],
                'mss_cnc_glacier_algae2': [0],
            }
            
            # Run SNICAR
            result = setup_snicar(**config)
            albedo_spectral = result['albedo']
            
            # Interpolate to match requested wavelengths
            snicar_wvl = result['wavelengths']
            return np.interp(wvl, snicar_wvl, albedo_spectral)
            
        except Exception as e:
            print(f"BioSNICAR error: {e}, falling back to parameterization")
            return snicar_parameterized_albedo(grain, illum, wvl)
    else:
        # Fallback: Use parameterization
        return snicar_parameterized_albedo(grain, illum, wvl)


def snicar_parameterized_albedo(grain, illum, wvl):
    """
    SNICAR-equivalent parameterization for clean snow albedo
    
    Based on Wiscombe & Warren (1980) + Kokhanovsky (2004)
    This provides reasonable approximation when PySNICAR unavailable.
    
    Parameters
    ----------
    grain : float
        Grain radius in micrometers
    illum : float
        Solar zenith angle in degrees
    wvl : np.ndarray
        Wavelength array in nm
        
    Returns
    -------
    np.ndarray
        Spectral albedo
    """
    # Ice refractive index (imaginary part) - Warren & Brandt 2008
    # Simplified parameterization for 350-2500 nm
    wvl_um = wvl / 1000.0
    
    # Piecewise linear approximation of ice absorption
    n_imag = np.zeros_like(wvl_um)
    
    # Visible (350-700 nm): very low absorption
    mask_vis = wvl_um < 0.7
    n_imag[mask_vis] = 1e-9 * np.exp((wvl_um[mask_vis] - 0.4) / 0.15)
    
    # NIR (700-1400 nm): moderate absorption, ice absorption bands
    mask_nir = (wvl_um >= 0.7) & (wvl_um < 1.4)
    n_imag[mask_nir] = 1e-7 * np.exp((wvl_um[mask_nir] - 0.8) / 0.3)
    
    # SWIR (1400-2500 nm): strong absorption
    mask_swir = wvl_um >= 1.4
    n_imag[mask_swir] = 1e-5 * np.exp((wvl_um[mask_swir] - 1.5) / 0.4)
    
    # Effective path length (depends on grain size and zenith angle)
    mu0 = np.cos(np.deg2rad(illum))
    path_scale = 1.0 / np.maximum(mu0, 0.1)  # Avoid division by zero at horizon
    
    # Absorption coefficient (m^-1)
    # Scales with grain size: larger grains = longer path = more absorption
    alpha = (4 * np.pi * n_imag / (wvl_um * 1e-6)) * (grain * 1e-6)
    
    # Single scattering albedo
    ssa = 1.0 - alpha / (alpha + 1e6)  # Normalize to prevent >1
    
    # Asymmetry parameter (typical for snow)
    g = 0.89
    
    # Two-stream approximation for albedo
    gamma1 = (np.sqrt(3) / 2) * (1 - ssa * (1 + g) / 2)
    gamma2 = (np.sqrt(3) / 2) * ssa * (1 - g) / 2
    
    # Diffuse albedo (simplified)
    k = np.sqrt(gamma1**2 - gamma2**2)
    
    # Apply path length scaling
    albedo = gamma2 / (gamma1 + k / path_scale)
    
    # Clamp to physical range
    albedo = np.clip(albedo, 0.0, 1.0)
    
    return albedo


def snicar_brdf_approximation(albedo, grain, illum, view, azim):
    """
    Approximate BRDF from SNICAR hemispherical albedo
    
    Uses empirical angular correction based on Ross-Li kernel approach
    This maintains the spectral shape from SNICAR while adding angular dependence.
    
    Parameters
    ----------
    albedo : np.ndarray
        Spectral albedo from SNICAR (wavelength dimension)
    grain : float
        Grain radius (affects roughness)
    illum : float
        Illumination zenith angle in degrees
    view : float
        Viewing zenith angle in degrees
    azim : float
        Relative azimuth angle in degrees
        
    Returns
    -------
    np.ndarray
        Spectral BRDF
    """
    ti, tv, phi = np.deg2rad(illum), np.deg2rad(view), np.deg2rad(azim)
    
    # Scattering angle
    theta = np.arccos(np.cos(ti)*np.cos(tv) + np.sin(ti)*np.sin(tv)*np.cos(phi))
    
    # Ross-Thick kernel (volume scattering)
    xi = np.arccos(np.cos(ti)*np.cos(tv) + np.sin(ti)*np.sin(tv)*np.cos(phi))
    ross = (np.pi/2 - xi) * np.cos(xi) + np.sin(xi)
    ross = ross / (np.cos(ti) + np.cos(tv)) - np.pi/4
    
    # Li-Sparse kernel (geometric-optical)
    tan_ti, tan_tv = np.tan(ti), np.tan(tv)
    cos_phi = np.cos(phi)
    D = np.sqrt(tan_ti**2 + tan_tv**2 - 2*tan_ti*tan_tv*cos_phi)
    
    cos_t = 2 / np.pi * ((np.pi/2 - xi)*np.cos(xi) + np.sin(xi)) / (np.cos(ti) + np.cos(tv))
    t = np.arccos(np.clip(cos_t, -1, 1))
    
    O = (t - np.sin(t)*np.cos(t)) / np.pi
    li = O - (np.cos(ti) + np.cos(tv)) + 0.5*(1 + np.cos(xi))*(np.cos(ti)*np.cos(tv))
    
    # Combine kernels with grain-size dependent weights
    # Smaller grains = more volume scattering, larger grains = more geometric
    w_ross = 0.3 * (1 - np.exp(-grain/500))  # Increases with grain size
    w_li = 0.1 * np.exp(-grain/1000)         # Decreases with grain size
    
    # BRDF normalization factor
    brdf_factor = 1.0 + w_ross * ross + w_li * li
    brdf_factor = np.maximum(brdf_factor, 0.3)  # Prevent unrealistic darkening
    
    # Apply to each wavelength
    return albedo * brdf_factor


# Global variables for multiprocessing
_WVL = None
_LAYER_DEPTH = None
_DENSITY = None


def init_worker(wvl, layer_depth, density):
    """Initialize worker process with shared data"""
    global _WVL, _LAYER_DEPTH, _DENSITY
    _WVL, _LAYER_DEPTH, _DENSITY = wvl, layer_depth, density


def worker_sbd(params):
    """Worker function for SBD LUT calculation using SNICAR"""
    illum, view, azim, grain = params
    
    # Get SNICAR albedo for this grain size and illumination
    albedo = snicar_directional_albedo(grain, illum, _WVL, _LAYER_DEPTH, _DENSITY)
    
    # Convert to BRDF
    brdf = snicar_brdf_approximation(albedo, grain, illum, view, azim)
    
    # Calculate band depth
    return calculate_band_depth_matlab(brdf, _WVL)


def worker_anisotropy(params):
    """Worker function for anisotropy LUT calculation"""
    illum, view, azim, grain = params
    
    # Hemispherical albedo (average over viewing angles)
    albedo = snicar_directional_albedo(grain, illum, _WVL, _LAYER_DEPTH, _DENSITY)
    
    # BRDF at specific viewing geometry
    brdf = snicar_brdf_approximation(albedo, grain, illum, view, azim)
    
    # Anisotropy factor = albedo / BRDF
    aniso = albedo / (brdf + 1e-10)
    aniso = np.where((albedo < 0.001) | (brdf < 0.01), 1.0, aniso)
    
    return aniso.astype(np.float32)


def generate_luts(wvl_path, output_dir, layer_depth=10.0, density=300.0, num_workers=None):
    """
    Generate SNICAR-based lookup tables for clean snow
    
    Parameters
    ----------
    wvl_path : Path
        Path to wavelength file (.npy)
    output_dir : Path
        Output directory for LUTs
    layer_depth : float
        Snow layer depth in meters (default 10m = semi-infinite)
    density : float
        Snow density in kg/m³ (default 300)
    num_workers : int, optional
        Number of worker processes (default: CPU count)
    """
    print("="*70)
    print("ISSIA SNICAR Lookup Table Generator (Clean Snow)")
    print("="*70)
    
    # Load wavelengths
    wvl = np.load(wvl_path)
    print(f"Wavelengths: {len(wvl)} bands ({wvl.min():.0f}-{wvl.max():.0f} nm)")
    
    # Define LUT dimensions (match ART)
    grain_radii = np.arange(30, 5001, 30)
    illum_angles = np.arange(0, 86, 5)
    view_angles = np.arange(0, 86, 5)
    azim_angles = np.arange(0, 361, 10)
    
    print(f"\nLUT dimensions:")
    print(f"  Grain radii: {len(grain_radii)} values ({grain_radii.min()}-{grain_radii.max()} µm)")
    print(f"  Illumination angles: {len(illum_angles)} values (0-85°)")
    print(f"  Viewing angles: {len(view_angles)} values (0-85°)")
    print(f"  Azimuth angles: {len(azim_angles)} values (0-360°)")
    print(f"  Layer depth: {layer_depth} m")
    print(f"  Snow density: {density} kg/m³")
    
    total_combinations = len(illum_angles) * len(view_angles) * len(azim_angles) * len(grain_radii)
    print(f"  Total combinations: {total_combinations:,}")
    
    if not SNICAR_AVAILABLE:
        print("\n⚠ WARNING: Using parameterized albedo (PySNICAR not installed)")
        print("  Results will be approximate. For production use, install PySNICAR.")
    
    # Setup multiprocessing
    if num_workers is None:
        num_workers = os.cpu_count()
    print(f"\nUsing {num_workers} worker processes")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate parameter combinations
    params_list = list(itertools.product(illum_angles, view_angles, azim_angles, grain_radii))
    
    # --- TABLE 1: SBD (Scaled Band Depth) ---
    print(f"\n[1/3] Generating SBD LUT (SNICAR)...")
    with mp.Pool(processes=num_workers, initializer=init_worker, 
                 initargs=(wvl, layer_depth, density)) as pool:
        results = list(tqdm(pool.imap(worker_sbd, params_list, chunksize=1000), 
                           total=len(params_list), desc="SBD"))
    
    sbd_lut = np.array(results).reshape(
        len(illum_angles), len(view_angles), len(azim_angles), len(grain_radii)
    )
    np.save(output_dir / "sbd_lut_snicar.npy", sbd_lut.astype(np.float32))
    print(f"  Saved: {output_dir / 'sbd_lut_snicar.npy'} ({sbd_lut.nbytes / 1e6:.1f} MB)")
    
    # --- TABLE 2: ALBEDO ---
    print(f"\n[2/3] Generating Albedo LUT (SNICAR)...")
    albedo_results = []
    for g in tqdm(grain_radii, desc="Albedo"):
        # Use nadir illumination (30°) as representative
        albedo = snicar_directional_albedo(g, 30.0, wvl, layer_depth, density)
        albedo_results.append(albedo)
    
    albedo_lut = np.array(albedo_results)
    np.save(output_dir / "albedo_lut_snicar.npy", albedo_lut)
    print(f"  Saved: {output_dir / 'albedo_lut_snicar.npy'} ({albedo_lut.nbytes / 1e6:.1f} MB)")
    
    # --- TABLE 3: ANISOTROPY ---
    print(f"\n[3/3] Generating Anisotropy LUT (SNICAR)...")
    with mp.Pool(processes=num_workers, initializer=init_worker,
                 initargs=(wvl, layer_depth, density)) as pool:
        results = list(tqdm(pool.imap(worker_anisotropy, params_list, chunksize=500), 
                           total=len(params_list), desc="Anisotropy"))
    
    ani_lut = np.array(results).reshape(
        len(illum_angles), len(view_angles), len(azim_angles), len(grain_radii), len(wvl)
    )
    np.savez_compressed(output_dir / "anisotropy_lut_snicar.npz", data=ani_lut.astype(np.float32))
    print(f"  Saved: {output_dir / 'anisotropy_lut_snicar.npz'} (compressed)")
    
    print(f"\n{'='*70}")
    print(f"All SNICAR LUTs generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate SNICAR-based ISSIA lookup tables for clean snow',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--wvl', type=str, default='wvl.npy',
                       help='Path to wavelength file')
    parser.add_argument('--output-dir', type=str, default='luts_snicar',
                       help='Output directory for lookup tables')
    parser.add_argument('--layer-depth', type=float, default=10.0,
                       help='Snow layer depth in meters (10m = semi-infinite)')
    parser.add_argument('--density', type=float, default=300.0,
                       help='Snow density in kg/m³')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count)')
    
    args = parser.parse_args()
    
    generate_luts(
        wvl_path=Path(args.wvl),
        output_dir=Path(args.output_dir),
        layer_depth=args.layer_depth,
        density=args.density,
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()
