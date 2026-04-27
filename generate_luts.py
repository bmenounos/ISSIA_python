#!/usr/bin/env python
"""
ISSIA Lookup Table Generator

Generates the three lookup tables required for ISSIA processing:
1. SBD (Scaled Band Depth) LUT - maps band depth to grain size
2. Albedo LUT - clean snow spectral albedo vs grain size
3. Anisotropy LUT - BRDF correction factors

Based on Asymptotic Radiative Transfer (ART) theory.

Usage:
    python generate_luts.py --wvl wvl.npy --iop IOP_2008_ASCIItable.txt --output-dir luts

Notes:
    - Generation takes ~30-60 minutes depending on CPU count
    - LUTs match MATLAB implementation (no spectral smoothing)
"""

import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull
from tqdm import tqdm
import multiprocessing as mp
import itertools
import argparse
import os


def calculate_band_depth_matlab(spectrum, wavelengths):
    """
    Calculate continuum-removed band depth - MATLAB matched
    
    No Savitzky-Golay smoothing in LUT generation (applied at runtime instead).
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


def art_model_reflectance(grain, illum, view, azim, wvl, n_imag):
    """
    ART model for snow bidirectional reflectance
    
    Parameters
    ----------
    grain : float
        Grain radius in micrometers
    illum : float
        Illumination zenith angle in degrees
    view : float
        Viewing zenith angle in degrees
    azim : float
        Relative azimuth angle in degrees
    wvl : np.ndarray
        Wavelength array in nm
    n_imag : np.ndarray
        Imaginary refractive index of ice
        
    Returns
    -------
    np.ndarray
        Spectral reflectance
    """
    ti, tv, phi = np.deg2rad(illum), np.deg2rad(view), np.deg2rad(azim)
    new_phi = np.pi - phi
    
    # Scattering angle
    theta = np.arccos(-np.cos(ti)*np.cos(tv) + np.sin(ti)*np.sin(tv)*np.cos(new_phi))
    
    # Phase function
    phase = 11.1*np.exp(-0.087*np.rad2deg(theta)) + 1.1*np.exp(-0.014*np.rad2deg(theta))
    
    # Reflectance at surface
    r0 = (1.247 + 1.186*(np.cos(ti)+np.cos(tv)) + 5.157*(np.cos(ti)*np.cos(tv)) + phase) / (4*(np.cos(ti)+np.cos(tv)))
    
    # Escape functions
    k0v, k0i = (3/7)*(1+2*np.cos(tv)), (3/7)*(1+2*np.cos(ti))
    
    # Absorption
    alpha = np.sqrt((4*np.pi*n_imag/(wvl*1e-9)) * 13 * 2 * grain * 1e-6)
    
    return r0 * np.exp(-alpha * k0i * k0v / r0)


def art_albedo(grain, wvl, n_imag):
    """
    ART model for snow hemispherical albedo
    
    Parameters
    ----------
    grain : float
        Grain radius in micrometers
    wvl : np.ndarray
        Wavelength array in nm
    n_imag : np.ndarray
        Imaginary refractive index of ice
        
    Returns
    -------
    np.ndarray
        Spectral albedo
    """
    k0v, k0i = (3/7)*3, (3/7)*3
    alpha = np.sqrt((4*np.pi*n_imag/(wvl*1e-9)) * 13 * 2 * grain * 1e-6)
    return np.exp(-1/((k0i*k0v)/1.0) * np.log(1.0/np.exp(-alpha * k0i * k0v / 1.0)))


# Global variables for multiprocessing
_WVL = None
_N_IMAG = None


def init_worker(wvl, n_imag):
    """Initialize worker process with shared data"""
    global _WVL, _N_IMAG
    _WVL, _N_IMAG = wvl, n_imag


def worker_sbd(params):
    """Worker function for SBD LUT calculation"""
    refl = art_model_reflectance(params[3], params[0], params[1], params[2], _WVL, _N_IMAG)
    return calculate_band_depth_matlab(refl, _WVL)


def worker_anisotropy(params):
    """Worker function for anisotropy LUT calculation"""
    refl = art_model_reflectance(params[3], params[0], params[1], params[2], _WVL, _N_IMAG)
    albedo = art_albedo(params[3], _WVL, _N_IMAG)
    
    # Anisotropy factor = albedo / BRF
    aniso = albedo / (refl + 1e-10)
    aniso = np.where((albedo < 0.001) | (refl < 0.01), 1.0, aniso)
    
    return aniso.astype(np.float32)


def generate_luts(wvl_path, iop_path, output_dir, num_workers=None):
    """
    Generate all three lookup tables
    
    Parameters
    ----------
    wvl_path : Path
        Path to wavelength file (.npy)
    iop_path : Path
        Path to ice optical properties file
    output_dir : Path
        Output directory for LUTs
    num_workers : int, optional
        Number of worker processes (default: CPU count)
    """
    print("="*70)
    print("ISSIA Lookup Table Generator")
    print("="*70)
    
    # Load wavelengths
    wvl = np.load(wvl_path)
    print(f"Wavelengths: {len(wvl)} bands ({wvl.min():.0f}-{wvl.max():.0f} nm)")
    
    # Load ice optical properties
    ri_data = np.loadtxt(iop_path)
    n_imag = np.interp(wvl, ri_data[96:265, 0]*1000, ri_data[96:265, 2])
    print(f"Ice optical properties loaded")
    
    # Define LUT dimensions
    #grain_radii = np.arange(30, 10001, 30)  # 30:30:10000 µm (333 values)
    grain_radii = np.arange(30, 5001, 30) 
    illum_angles = np.arange(0, 86, 5)       # 0:5:85° (18 values)
    view_angles = np.arange(0, 86, 5)        # 0:5:85° (18 values)
    azim_angles = np.arange(0, 361, 10)      # 0:10:360° (37 values)
    
    print(f"\nLUT dimensions:")
    print(f"  Grain radii: {len(grain_radii)} values ({grain_radii.min()}-{grain_radii.max()} µm)")
    print(f"  Illumination angles: {len(illum_angles)} values (0-85°)")
    print(f"  Viewing angles: {len(view_angles)} values (0-85°)")
    print(f"  Azimuth angles: {len(azim_angles)} values (0-360°)")
    
    total_combinations = len(illum_angles) * len(view_angles) * len(azim_angles) * len(grain_radii)
    print(f"  Total combinations: {total_combinations:,}")
    
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
    print(f"\n[1/3] Generating SBD LUT...")
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(wvl, n_imag)) as pool:
        results = list(tqdm(pool.imap(worker_sbd, params_list, chunksize=2000), 
                           total=len(params_list), desc="SBD"))
    
    sbd_lut = np.array(results).reshape(
        len(illum_angles), len(view_angles), len(azim_angles), len(grain_radii)
    )
    np.save(output_dir / "sbd_lut.npy", sbd_lut.astype(np.float32))
    print(f"  Saved: {output_dir / 'sbd_lut.npy'} ({sbd_lut.nbytes / 1e6:.1f} MB)")
    
    # --- TABLE 2: ALBEDO ---
    print(f"\n[2/3] Generating Albedo LUT...")
    albedo_lut = np.array([art_albedo(g, wvl, n_imag) for g in tqdm(grain_radii, desc="Albedo")])
    np.save(output_dir / "albedo_lut.npy", albedo_lut)
    print(f"  Saved: {output_dir / 'albedo_lut.npy'} ({albedo_lut.nbytes / 1e6:.1f} MB)")
    
    # --- TABLE 3: ANISOTROPY ---
    print(f"\n[3/3] Generating Anisotropy LUT...")
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(wvl, n_imag)) as pool:
        results = list(tqdm(pool.imap(worker_anisotropy, params_list, chunksize=1000), 
                           total=len(params_list), desc="Anisotropy"))
    
    ani_lut = np.array(results).reshape(
        len(illum_angles), len(view_angles), len(azim_angles), len(grain_radii), len(wvl)
    )
    np.savez_compressed(output_dir / "anisotropy_lut.npz", data=ani_lut.astype(np.float32))
    print(f"  Saved: {output_dir / 'anisotropy_lut.npz'} (compressed)")
    
    print(f"\n{'='*70}")
    print(f"All LUTs generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate ISSIA lookup tables',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--wvl', type=str, default='wvl.npy',
                       help='Path to wavelength file')
    parser.add_argument('--iop', type=str, default='IOP_2008_ASCIItable.txt',
                       help='Path to ice optical properties file')
    parser.add_argument('--output-dir', type=str, default='luts',
                       help='Output directory for lookup tables')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count)')
    
    args = parser.parse_args()
    
    generate_luts(
        wvl_path=Path(args.wvl),
        iop_path=Path(args.iop),
        output_dir=Path(args.output_dir),
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()
