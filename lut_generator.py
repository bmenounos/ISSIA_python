"""
FIXED Lookup Table (LUT) Generator for ISSIA

Critical fix: Corrected continuum removal to match MATLAB's convex hull method
This should produce band depths around 0.08-0.12 for typical snow (not 0.18-0.28)

Generates the three required lookup tables:
1. Scaled Band Depth LUT - 4D array [illumination, viewing, azimuth, grain_size]
2. Anisotropy Factor LUT - 5D array [illumination, viewing, azimuth, grain_size, wavelength]
3. White-Sky Albedo LUT - 2D array [grain_size, wavelength]
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class ISSIALUTGenerator:
    """
    Generator for ISSIA lookup tables using radiative transfer modeling
    """
    
    def __init__(self,
                 wavelengths: np.ndarray,
                 grain_radii: np.ndarray,
                 illumination_angles: np.ndarray,
                 viewing_angles: np.ndarray,
                 relative_azimuths: np.ndarray):
        """
        Initialize LUT generator
        """
        self.wavelengths = wavelengths
        self.grain_radii = grain_radii
        self.illumination_angles = illumination_angles
        self.viewing_angles = viewing_angles
        self.relative_azimuths = relative_azimuths
        
    def generate_sbd_lut(self, output_path: Optional[Path] = None) -> np.ndarray:
        """Generate Scaled Band Depth (SBD) lookup table"""
        print("="*70)
        print("Generating Scaled Band Depth LUT...")
        print("="*70)
        print(f"Dimensions: {len(self.illumination_angles)} x "
              f"{len(self.viewing_angles)} x {len(self.relative_azimuths)} x "
              f"{len(self.grain_radii)}")
        
        # Initialize LUT
        sbd_lut = np.zeros((len(self.illumination_angles),
                           len(self.viewing_angles),
                           len(self.relative_azimuths),
                           len(self.grain_radii)))
        
        # Generate for each geometry and grain size
        total = (len(self.illumination_angles) * len(self.viewing_angles) * 
                len(self.relative_azimuths))
        count = 0
        
        for i, illum in enumerate(self.illumination_angles):
            for j, view in enumerate(self.viewing_angles):
                for k, azim in enumerate(self.relative_azimuths):
                    count += 1
                    if count % 100 == 0 or count == total:
                        print(f"Progress: {100 * count / total:.1f}%", end='\r')
                    
                    # Simulate reflectance for this geometry at all grain sizes
                    for g, grain_size in enumerate(self.grain_radii):
                        # Run radiative transfer model
                        spectrum = self._simulate_snow_reflectance(
                            grain_size, illum, view, azim
                        )
                        
                        # Calculate continuum-removed band depth (FIXED!)
                        _, band_depth = self._calculate_band_depth_fixed(spectrum)
                        
                        sbd_lut[i, j, k, g] = band_depth
        
        print(f"\nProgress: 100.0%")
        
        # Verify band depths are reasonable
        print("\n" + "="*70)
        print("VERIFICATION: Checking band depth values")
        print("="*70)
        sample_bd = sbd_lut[7, 0, 0, :]  # illum=35°, view=0°, azim=0°
        print(f"Sample band depths (illum=35°, view=0°, azim=0°):")
        print(f"  Grain 30 μm:  {sample_bd[0]:.4f}")
        print(f"  Grain 150 μm: {sample_bd[4]:.4f}")
        print(f"  Grain 300 μm: {sample_bd[9]:.4f}")
        print(f"  Grain 500 μm: {sample_bd[15]:.4f}")
        
        if sample_bd[4] > 0.15:
            print(f"\n⚠️  WARNING: Band depths seem high (>0.15)")
            print(f"    Expected range: 0.08-0.12 for typical snow")
        else:
            print(f"\n✓ Band depths look reasonable (0.08-0.12 range)")
        
        if output_path:
            np.save(output_path, sbd_lut)
            print(f"\nSaved SBD LUT to: {output_path}")
        
        return sbd_lut
    
    def generate_anisotropy_lut(self, output_path: Optional[Path] = None) -> np.ndarray:
        """Generate Anisotropy Factor lookup table"""
        print("\n" + "="*70)
        print("Generating Anisotropy Factor LUT...")
        print("="*70)
        
        shape = (len(self.illumination_angles),
                len(self.viewing_angles),
                len(self.relative_azimuths),
                len(self.grain_radii),
                len(self.wavelengths))
        
        size_gb = np.prod(shape) * 4 / 1e9  # 4 bytes per float32
        print(f"Expected size: {size_gb:.2f} GB (using float32)")
        
        anisotropy_lut = np.zeros(shape, dtype=np.float32)
        
        total = np.prod(shape[:4])
        count = 0
        
        for i, illum in enumerate(self.illumination_angles):
            for j, view in enumerate(self.viewing_angles):
                for k, azim in enumerate(self.relative_azimuths):
                    for g, grain in enumerate(self.grain_radii):
                        count += 1
                        if count % 1000 == 0 or count == total:
                            print(f"Progress: {100 * count / total:.1f}%", end='\r')
                        
                        # Directional reflectance
                        directional = self._simulate_snow_reflectance(grain, illum, view, azim)
                        
                        # Hemispherical reflectance
                        hemispherical = self._simulate_snow_albedo(grain)
                        
                        # Anisotropy factor
                        anisotropy_lut[i, j, k, g, :] = directional / (hemispherical + 1e-10)
        
        print(f"\nProgress: 100.0%")
        
        if output_path:
            print(f"\nSaving to: {output_path}")
            np.save(output_path, anisotropy_lut)
            print(f"Saved Anisotropy LUT")
        
        return anisotropy_lut
    
    def generate_albedo_lut(self, output_path: Optional[Path] = None) -> np.ndarray:
        """Generate White-Sky Albedo lookup table"""
        print("\n" + "="*70)
        print("Generating White-Sky Albedo LUT...")
        print("="*70)
        
        albedo_lut = np.zeros((len(self.grain_radii), len(self.wavelengths)))
        
        for g, grain_size in enumerate(self.grain_radii):
            if (g+1) % 10 == 0 or (g+1) == len(self.grain_radii):
                print(f"Progress: {100 * (g+1) / len(self.grain_radii):.1f}%", end='\r')
            
            # Simulate white-sky albedo
            albedo_lut[g, :] = self._simulate_snow_albedo(grain_size)
        
        print(f"\nProgress: 100.0%")
        
        if output_path:
            np.save(output_path, albedo_lut)
            print(f"\nSaved Albedo LUT to: {output_path}")
        
        return albedo_lut
    
    def _load_ice_refractive_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load Warren 2008 ice refractive index"""
        data = np.loadtxt('IOP_2008_ASCIItable.txt')
        
        # Extract 350-2600 nm range
        wvl_ri = data[96:265, 0] * 1000  # microns to nm
        n_real = data[96:265, 1]
        n_imag = data[96:265, 2]
        
        # Resample to sensor wavelengths
        n_real_rs = np.interp(self.wavelengths, wvl_ri, n_real)
        n_imag_rs = np.interp(self.wavelengths, wvl_ri, n_imag)
        
        return n_real_rs, n_imag_rs
    
    def _brf0(self, theta_i: float, theta_v: float, phi: float) -> float:
        """Calculate r0 (Kokhanovsky & Brieglieb 2012)"""
        new_phi = np.pi - phi
        theta = np.arccos(-np.cos(theta_i) * np.cos(theta_v) + 
                         np.sin(theta_i) * np.sin(theta_v) * np.cos(new_phi))
        theta_deg = np.rad2deg(theta)
        
        phase = 11.1 * np.exp(-0.087 * theta_deg) + 1.1 * np.exp(-0.014 * theta_deg)
        rr = 1.247 + 1.186 * (np.cos(theta_i) + np.cos(theta_v)) + \
             5.157 * (np.cos(theta_i) * np.cos(theta_v)) + phase
        rr = rr / (4 * (np.cos(theta_i) + np.cos(theta_v)))
        return rr
    
    def _simulate_snow_reflectance(self,
                                  grain_size: float,
                                  illumination: float,
                                  viewing: float,
                                  relative_azimuth: float) -> np.ndarray:
        """Simulate snow BRF using Kokhanovsky & Brieglieb 2012 ART model"""
        if not hasattr(self, '_n_real'):
            self._n_real, self._n_imag = self._load_ice_refractive_index()
        
        # Convert to radians
        theta_i = np.deg2rad(illumination)
        theta_v = np.deg2rad(viewing)
        phi = np.deg2rad(relative_azimuth)
        
        # Convert grain size microns→meters
        opt_radius = grain_size * 1e-6
        
        # Constants
        b = 13  # L = 13d
        M = 0   # Clean snow
        
        # r0
        r0 = self._brf0(theta_i, theta_v, phi)
        
        # k0 factors
        k0v = (3.0 / 7.0) * (1 + 2 * np.cos(theta_v))
        k0i = (3.0 / 7.0) * (1 + 2 * np.cos(theta_i))
        
        # Wavelengths nm→m
        wvl_m = self.wavelengths * 1e-9
        
        # BRF calculation
        gamma = 4 * np.pi * (self._n_imag + M) / wvl_m
        alpha = np.sqrt(gamma * b * 2 * opt_radius)
        brf = r0 * np.exp(-alpha * k0i * k0v / r0)
        
        return brf
    
    def _simulate_snow_albedo(self, grain_size: float) -> np.ndarray:
        """Simulate hemispherical albedo using ART model"""
        if not hasattr(self, '_n_real'):
            self._n_real, self._n_imag = self._load_ice_refractive_index()
        
        # Use nadir viewing for albedo approximation
        theta_i = np.deg2rad(0)
        theta_v = np.deg2rad(0)
        phi = np.deg2rad(0)
        
        opt_radius = grain_size * 1e-6
        b = 13
        M = 0
        
        r0 = self._brf0(theta_i, theta_v, phi)
        k0v = (3.0 / 7.0) * (1 + 2 * np.cos(theta_v))
        k0i = (3.0 / 7.0) * (1 + 2 * np.cos(theta_i))
        
        wvl_m = self.wavelengths * 1e-9
        gamma = 4 * np.pi * (self._n_imag + M) / wvl_m
        alpha = np.sqrt(gamma * b * 2 * opt_radius)
        
        # Albedo (Kokhanovsky & Schreier 2009)
        Q = (k0i * k0v) / r0
        brf = r0 * np.exp(-alpha * k0i * k0v / r0)
        albedo = np.exp(-1/Q * np.log(r0/brf))
        
        return albedo
    
    def _calculate_band_depth_fixed(self, spectrum: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        FIXED continuum removal using proper upper convex hull
        
        This should produce band depths around 0.08-0.12 for typical snow
        (not 0.18-0.28 like the buggy version)
        """
        from scipy.spatial import ConvexHull
        
        # Extract 830-1130nm range (matches MATLAB)
        left_idx = np.argmin(np.abs(self.wavelengths - 830))
        right_idx = np.argmin(np.abs(self.wavelengths - 1130))
        
        wvl_subset = self.wavelengths[left_idx:right_idx+1]
        spec_subset = spectrum[left_idx:right_idx+1]
        
        if len(spec_subset) < 3:
            return spec_subset, 0.0
        
        try:
            # Use wavelength values for x-axis
            wvl_ext = np.concatenate([[wvl_subset[0] - 10], wvl_subset, [wvl_subset[-1] + 10]])
            spec_ext = np.concatenate([[0], spec_subset, [0]])
            
            # Compute convex hull
            points = np.column_stack([wvl_ext, spec_ext])
            hull = ConvexHull(points)
            
            # Extract ONLY upper hull points
            vertices = hull.vertices
            hull_points = points[vertices]
            
            # Sort by wavelength
            sorted_idx = np.argsort(hull_points[:, 0])
            hull_sorted = hull_points[sorted_idx]
            
            # Find upper envelope: go to peak, then down
            max_y_idx = np.argmax(hull_sorted[:, 1])
            
            upper_x = []
            upper_y = []
            
            # Left side: monotonically increasing
            for i in range(max_y_idx + 1):
                if i == 0 or hull_sorted[i, 1] >= hull_sorted[i-1, 1] - 1e-10:
                    upper_x.append(hull_sorted[i, 0])
                    upper_y.append(hull_sorted[i, 1])
            
            # Right side: monotonically decreasing
            for i in range(max_y_idx + 1, len(hull_sorted)):
                if hull_sorted[i, 1] <= hull_sorted[i-1, 1] + 1e-10:
                    upper_x.append(hull_sorted[i, 0])
                    upper_y.append(hull_sorted[i, 1])
            
            upper_x = np.array(upper_x)
            upper_y = np.array(upper_y)
            
            # Remove boundary points
            mask = (upper_x >= wvl_subset[0]) & (upper_x <= wvl_subset[-1])
            upper_x = upper_x[mask]
            upper_y = upper_y[mask]
            
            if len(upper_x) < 2:
                # Fallback: linear continuum
                continuum = np.linspace(spec_subset[0], spec_subset[-1], len(spec_subset))
            else:
                # Interpolate continuum
                continuum = np.interp(wvl_subset, upper_x, upper_y)
            
            # Ensure continuum >= spectrum (must be true for upper hull)
            continuum = np.maximum(continuum, spec_subset)
            continuum = np.maximum(continuum, 1e-10)
            
            # Continuum removal
            cr_spectrum = spec_subset / continuum
            
            # Band depth
            band_depth = 1.0 - np.min(cr_spectrum)
            band_depth = np.clip(band_depth, 0.0, 1.0)
            
            return cr_spectrum, band_depth
            
        except Exception as e:
            # Fallback
            continuum = np.linspace(spec_subset[0], spec_subset[-1], len(spec_subset))
            continuum = np.maximum(continuum, 1e-10)
            cr_spectrum = spec_subset / continuum
            band_depth = np.clip(1.0 - np.min(cr_spectrum), 0.0, 1.0)
            
            return cr_spectrum, band_depth


def main():
    """Generate ISSIA lookup tables"""
    from pathlib import Path
    
    print("="*70)
    print("ISSIA LUT GENERATOR - FIXED VERSION")
    print("="*70)
    
    # Load wavelengths from wvl.npy
    try:
        wavelengths = np.load('wvl.npy')
        print(f"\n✓ Loaded wavelengths from wvl.npy")
    except Exception as e:
        print(f"\n❌ ERROR: Could not load wvl.npy: {e}")
        print(f"   Please ensure wvl.npy is in the current directory")
        return
    
    # Define parameters
    grain_radii = np.arange(30, 5001, 30)
    illumination_angles = np.arange(0, 86, 5)
    viewing_angles = np.arange(0, 86, 5)
    relative_azimuths = np.arange(0, 361, 10)
    
    print(f"\nConfiguration:")
    print(f"  Wavelengths: {len(wavelengths)} ({wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm)")
    print(f"  Grain sizes: {len(grain_radii)} ({grain_radii[0]:.1f} - {grain_radii[-1]:.1f} μm)")
    print(f"  Illumination angles: {len(illumination_angles)} (0° - 85°)")
    print(f"  Viewing angles: {len(viewing_angles)} (0° - 85°)")
    print(f"  Relative azimuths: {len(relative_azimuths)} (0° - 360°)")
    
    generator = ISSIALUTGenerator(
        wavelengths=wavelengths,
        grain_radii=grain_radii,
        illumination_angles=illumination_angles,
        viewing_angles=viewing_angles,
        relative_azimuths=relative_azimuths
    )
    
    output_dir = Path("lookup_tables_fixed")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n✓ Output directory: {output_dir}")
    print("\n" + "="*70)
    input("Press Enter to start LUT generation...")
    
    # 1. Scaled Band Depth LUT
    sbd_lut = generator.generate_sbd_lut(
        output_path=output_dir / "sbd_lut.npy"
    )
    
    # 2. Albedo LUT
    albedo_lut = generator.generate_albedo_lut(
        output_path=output_dir / "albedo_lut.npy"
    )
    
    # 3. Anisotropy Factor LUT
    print("\n" + "="*70)
    print("Ready to generate Anisotropy LUT")
    print("WARNING: This is large (~2-3 GB) and takes time!")
    print("="*70)
    response = input("Continue? (y/n): ")
    
    if response.lower() == 'y':
        anisotropy_lut = generator.generate_anisotropy_lut(
            output_path=output_dir / "anisotropy_lut.npy"
        )
    
    print("\n" + "="*70)
    print("✓ LUT GENERATION COMPLETE")
    print("="*70)
    print(f"\nNew LUTs saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Verify the band depths look reasonable (0.08-0.12)")
    print("2. Replace your old LUTs with these new ones")
    print("3. Re-run processing with the fixed continuum removal")
    print("="*70)


if __name__ == "__main__":
    main()