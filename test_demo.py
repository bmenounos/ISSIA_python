"""
Test/Demo Script for ISSIA

Creates synthetic test data and demonstrates ISSIA functionality
without requiring actual ATCOR-4 files.
"""

import numpy as np
import dask.array as da
from pathlib import Path
import matplotlib.pyplot as plt
from issia import ISSIAProcessor
from lut_generator import ISSIALUTGenerator


def create_synthetic_snow_scene(rows: int = 500, cols: int = 500, 
                                n_bands: int = 451) -> dict:
    """
    Create synthetic snow/ice spectral data for testing
    
    Parameters:
    -----------
    rows, cols : int
        Scene dimensions
    n_bands : int
        Number of spectral bands
        
    Returns:
    --------
    scene : dict
        Dictionary with synthetic data
    """
    print("Creating synthetic test scene...")
    
    # Wavelengths (380-2500 nm)
    wavelengths = np.linspace(380, 2500, n_bands)
    
    # Create spatial patterns for grain size
    x = np.linspace(0, 4*np.pi, cols)
    y = np.linspace(0, 4*np.pi, rows)
    X, Y = np.meshgrid(x, y)
    
    # Grain size varies spatially (50-500 μm)
    grain_size_pattern = 200 + 150 * np.sin(X) * np.cos(Y)
    grain_size_pattern = np.clip(grain_size_pattern, 50, 500)
    
    # Terrain slope (0-30 degrees)
    slope = 10 + 10 * np.sin(X/2) * np.cos(Y/2)
    slope = np.clip(slope, 0, 30)
    
    # Terrain aspect (0-360 degrees)
    aspect = 180 + 90 * np.sin(X) + 90 * np.cos(Y)
    aspect = aspect % 360
    
    # Generate reflectance spectra
    reflectance = np.zeros((n_bands, rows, cols))
    
    for i in range(rows):
        if i % 50 == 0:
            print(f"  Generating spectra: {i}/{rows}")
        for j in range(cols):
            reflectance[:, i, j] = generate_snow_spectrum(
                wavelengths, 
                grain_size_pattern[i, j],
                impurity_level=0.01 * np.random.rand()
            )
    
    # Global flux (approximate solar spectrum)
    global_flux = np.zeros((n_bands, rows, cols))
    solar_spectrum = generate_solar_spectrum(wavelengths)
    for i in range(rows):
        for j in range(cols):
            global_flux[:, i, j] = solar_spectrum
    
    scene = {
        'reflectance': reflectance,
        'global_flux': global_flux,
        'slope': slope,
        'aspect': aspect,
        'grain_size_truth': grain_size_pattern,  # For validation
        'wavelengths': wavelengths
    }
    
    print("✓ Synthetic scene created")
    return scene


def generate_snow_spectrum(wavelengths: np.ndarray,
                           grain_size: float,
                           impurity_level: float = 0.0) -> np.ndarray:
    """
    Generate a synthetic snow reflectance spectrum
    
    Parameters:
    -----------
    wavelengths : np.ndarray
        Wavelength array (nm)
    grain_size : float
        Grain size (μm)
    impurity_level : float
        Level of impurities (0-1)
        
    Returns:
    --------
    spectrum : np.ndarray
        Reflectance spectrum
    """
    wl_um = wavelengths / 1000.0  # Convert to μm
    
    # Ice absorption (simplified)
    alpha_ice = np.zeros_like(wl_um)
    
    # Visible (minimal absorption)
    visible_mask = wl_um < 0.7
    alpha_ice[visible_mask] = 0.001
    
    # NIR (grain size dependent)
    nir_mask = (wl_um >= 0.7) & (wl_um < 1.4)
    alpha_ice[nir_mask] = 0.01 * ((wl_um[nir_mask] - 0.7) / 0.7)**2
    
    # SWIR (strong absorption)
    swir_mask = wl_um >= 1.4
    alpha_ice[swir_mask] = 0.1 * np.exp((wl_um[swir_mask] - 1.5) * 2)
    
    # Grain size effect
    grain_mm = grain_size / 1000.0
    tau = 4 * alpha_ice * grain_mm
    
    # Base reflectance
    r_clean = np.exp(-np.sqrt(tau))
    
    # Add impurities (reduce reflectance in visible)
    r_dirty = r_clean.copy()
    r_dirty[visible_mask] *= (1.0 - impurity_level)
    
    # Add some noise
    r_dirty += np.random.normal(0, 0.01, len(r_dirty))
    
    return np.clip(r_dirty, 0, 1)


def generate_solar_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """
    Generate approximate solar spectrum at ground
    
    Parameters:
    -----------
    wavelengths : np.ndarray
        Wavelength array (nm)
        
    Returns:
    --------
    spectrum : np.ndarray
        Solar irradiance (W/m²/nm)
    """
    wl_um = wavelengths / 1000.0
    
    # Simplified Planck function at 5800K with atmospheric attenuation
    T = 5800
    h = 6.626e-34
    c = 3.0e8
    k = 1.381e-23
    
    wl_m = wl_um * 1e-6
    
    # Planck function
    B = (2 * h * c**2 / wl_m**5) / (np.exp(h * c / (wl_m * k * T)) - 1)
    
    # Normalize and scale
    spectrum = B / B.max() * 1000
    
    # Add atmospheric absorption features
    # Water vapor absorption around 940, 1140, 1380, 1870 nm
    water_bands = [940, 1140, 1380, 1870]
    for band in water_bands:
        idx = np.argmin(np.abs(wavelengths - band))
        width = 20  # nm
        attenuation = np.exp(-((wavelengths - band)**2) / (2 * width**2))
        spectrum *= (1 - 0.3 * attenuation)
    
    return spectrum


def test_continuum_removal():
    """Test continuum removal function"""
    print("\n" + "="*60)
    print("TEST 1: Continuum Removal")
    print("="*60)
    
    wavelengths = np.linspace(380, 2500, 451)
    spectrum = generate_snow_spectrum(wavelengths, grain_size=200)
    
    # Create processor instance
    processor = ISSIAProcessor(
        wavelengths=wavelengths,
        grain_radii=np.array([100, 200, 300]),
        illumination_angles=np.array([0, 30, 60]),
        viewing_angles=np.array([0]),
        relative_azimuths=np.array([0])
    )
    
    # Test continuum removal
    cr_spectrum, band_depth = processor.continuum_removal(spectrum, wavelengths)
    
    print(f"Band depth at 1030 nm: {band_depth:.4f}")
    print(f"Expected range: 0.05 - 0.30 for snow")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(wavelengths, spectrum, 'b-', linewidth=2)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Reflectance')
    ax1.set_title('Original Spectrum')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([900, 1200])
    
    # Find continuum region
    left_idx = np.argmin(np.abs(wavelengths - 950))
    right_idx = np.argmin(np.abs(wavelengths - 1100))
    wl_subset = wavelengths[left_idx:right_idx+1]
    
    ax2.plot(wl_subset, cr_spectrum, 'r-', linewidth=2)
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=1030, color='g', linestyle='--', alpha=0.5, label='1030 nm')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Continuum-Removed Reflectance')
    ax2.set_title(f'Continuum Removed (Band Depth = {band_depth:.4f})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('test_continuum_removal.png', dpi=150)
    print("✓ Saved: test_continuum_removal.png")
    plt.close()


def test_lut_generation():
    """Test LUT generation with small dimensions"""
    print("\n" + "="*60)
    print("TEST 2: LUT Generation (Small Test)")
    print("="*60)
    
    # Use very small dimensions for quick test
    wavelengths = np.linspace(380, 2500, 100)  # Fewer bands
    grain_radii = np.array([50, 100, 200, 400, 800])
    illumination_angles = np.array([0, 30, 60])
    viewing_angles = np.array([0, 30])
    relative_azimuths = np.array([0, 90, 180])
    
    print(f"Test LUT dimensions:")
    print(f"  Wavelengths: {len(wavelengths)}")
    print(f"  Grain sizes: {len(grain_radii)}")
    print(f"  Illumination: {len(illumination_angles)}")
    print(f"  Viewing: {len(viewing_angles)}")
    print(f"  Azimuth: {len(relative_azimuths)}")
    
    generator = ISSIALUTGenerator(
        wavelengths=wavelengths,
        grain_radii=grain_radii,
        illumination_angles=illumination_angles,
        viewing_angles=viewing_angles,
        relative_azimuths=relative_azimuths
    )
    
    # Generate small test LUTs
    sbd_lut = generator.generate_sbd_lut()
    print(f"✓ SBD LUT shape: {sbd_lut.shape}")
    
    albedo_lut = generator.generate_albedo_lut()
    print(f"✓ Albedo LUT shape: {albedo_lut.shape}")
    
    # Skip anisotropy for quick test (too large)
    print("⊘ Skipping anisotropy LUT for quick test")


def test_synthetic_processing():
    """Test processing with synthetic data"""
    print("\n" + "="*60)
    print("TEST 3: Processing Synthetic Scene")
    print("="*60)
    
    # Create small synthetic scene
    scene = create_synthetic_snow_scene(rows=200, cols=200, n_bands=100)
    
    # Simple retrieval without full LUTs
    print("\nPerforming simple grain size retrieval...")
    
    wavelengths = scene['wavelengths']
    reflectance = scene['reflectance']
    grain_size_truth = scene['grain_size_truth']
    
    # Calculate band depth for each pixel
    band_depths = np.zeros((200, 200))
    
    for i in range(200):
        if i % 50 == 0:
            print(f"  Processing row {i}/200")
        for j in range(200):
            spectrum = reflectance[:, i, j]
            _, bd = ISSIAProcessor(
                wavelengths, np.array([100]), np.array([0]), 
                np.array([0]), np.array([0])
            ).continuum_removal(spectrum, wavelengths)
            band_depths[i, j] = bd
    
    # Simple empirical relationship: band_depth ∝ sqrt(grain_size)
    # This is approximate - real retrieval uses LUTs
    grain_size_retrieved = (band_depths * 2000)**2
    
    # Compare with truth
    error = grain_size_retrieved - grain_size_truth
    rmse = np.sqrt(np.mean(error**2))
    
    print(f"\nRetrieval Statistics:")
    print(f"  RMSE: {rmse:.2f} μm")
    print(f"  Mean error: {np.mean(error):.2f} μm")
    print(f"  Correlation: {np.corrcoef(grain_size_truth.flatten(), grain_size_retrieved.flatten())[0,1]:.3f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im1 = axes[0].imshow(grain_size_truth, cmap='YlOrRd', vmin=50, vmax=500)
    axes[0].set_title('True Grain Size (μm)')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(grain_size_retrieved, cmap='YlOrRd', vmin=50, vmax=500)
    axes[1].set_title('Retrieved Grain Size (μm)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(error, cmap='RdBu_r', vmin=-100, vmax=100)
    axes[2].set_title('Error (μm)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('test_synthetic_retrieval.png', dpi=150)
    print("✓ Saved: test_synthetic_retrieval.png")
    plt.close()


def main():
    """Run all tests"""
    print("="*60)
    print("ISSIA TEST SUITE")
    print("="*60)
    
    # Test 1: Continuum removal
    test_continuum_removal()
    
    # Test 2: LUT generation
    test_lut_generation()
    
    # Test 3: Synthetic processing
    test_synthetic_processing()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - test_continuum_removal.png")
    print("  - test_synthetic_retrieval.png")
    print("\nThese tests verify basic functionality.")
    print("For real data processing, use example_workflow.py")


if __name__ == "__main__":
    main()
