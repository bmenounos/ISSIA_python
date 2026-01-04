"""
Lookup Table (LUT) Generator for ISSIA

Generates the three required lookup tables:
1. Scaled Band Depth LUT - 4D array [illumination, viewing, azimuth, grain_size]
2. Anisotropy Factor LUT - 5D array [illumination, viewing, azimuth, grain_size, wavelength]
3. White-Sky Albedo LUT - 2D array [grain_size, wavelength]

Uses the ART (Asymptotic Radiative Transfer) model from Kokhanovsky & Zege (2004)
for generating physically accurate snow/ice reflectance spectra.
"""

import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
from pathlib import Path
from typing import Tuple, Optional
import warnings
from art_model import ARTSnowModel, simulate_snow_reflectance, simulate_snow_albedo


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
        
        Parameters:
        -----------
        wavelengths : np.ndarray
            Wavelength array (nm)
        grain_radii : np.ndarray
            Grain radius array (micrometers)
        illumination_angles : np.ndarray
            Illumination angles (degrees)
        viewing_angles : np.ndarray
            Viewing angles (degrees)
        relative_azimuths : np.ndarray
            Relative azimuth angles (degrees)
        """
        self.wavelengths = wavelengths
        self.grain_radii = grain_radii
        self.illumination_angles = illumination_angles
        self.viewing_angles = viewing_angles
        self.relative_azimuths = relative_azimuths
        
    def generate_sbd_lut(self,
                        output_path: Optional[Path] = None) -> np.ndarray:
        """
        Generate Scaled Band Depth (SBD) lookup table
        
        This LUT contains the depth of the 1030 nm absorption feature for
        different viewing geometries and grain sizes
        
        Parameters:
        -----------
        output_path : Path, optional
            Path to save the LUT
            
        Returns:
        --------
        sbd_lut : np.ndarray
            4D array [n_illum, n_view, n_azim, n_grains]
        """
        print("Generating Scaled Band Depth LUT...")
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
                    if count % 10 == 0:
                        print(f"Progress: {count}/{total} geometries")
                    
                    # Simulate reflectance for this geometry at all grain sizes
                    for g, grain_size in enumerate(self.grain_radii):
                        # Run radiative transfer model
                        spectrum = self._simulate_snow_reflectance(
                            grain_size, illum, view, azim
                        )
                        
                        # Calculate continuum-removed band depth
                        _, band_depth = self._calculate_band_depth(spectrum)
                        
                        sbd_lut[i, j, k, g] = band_depth
        
        if output_path:
            np.save(output_path, sbd_lut)
            print(f"Saved SBD LUT to: {output_path}")
        
        return sbd_lut
    
    def generate_anisotropy_lut(self,
                               output_path: Optional[Path] = None,
                               use_dask: bool = True) -> np.ndarray:
        """
        Generate Anisotropy Factor lookup table
        
        Contains the spectral anisotropy factor (ratio of directional to
        hemispherical reflectance) for all geometries, grain sizes, and wavelengths
        
        Parameters:
        -----------
        output_path : Path, optional
            Path to save the LUT
        use_dask : bool
            Use Dask for parallel generation (recommended for large LUTs)
            
        Returns:
        --------
        anisotropy_lut : np.ndarray
            5D array [n_illum, n_view, n_azim, n_grains, n_wavelengths]
        """
        print("Generating Anisotropy Factor LUT...")
        print(f"WARNING: This will create a large array!")
        
        shape = (len(self.illumination_angles),
                len(self.viewing_angles),
                len(self.relative_azimuths),
                len(self.grain_radii),
                len(self.wavelengths))
        
        size_gb = np.prod(shape) * 8 / 1e9  # 8 bytes per float64
        print(f"Expected size: {size_gb:.2f} GB")
        
        if use_dask:
            # Use Dask for memory-efficient generation
            anisotropy_lut = self._generate_anisotropy_dask(shape)
        else:
            anisotropy_lut = self._generate_anisotropy_numpy(shape)
        
        if output_path:
            print(f"Saving to: {output_path}")
            np.save(output_path, anisotropy_lut)
            print(f"Saved Anisotropy LUT")
        
        return anisotropy_lut
    
    def _generate_anisotropy_dask(self, shape: Tuple) -> np.ndarray:
        """Generate anisotropy LUT using Dask for memory efficiency"""
        
        def compute_chunk(illum_idx, view_idx, azim_idx, grain_idx):
            """Compute anisotropy for one geometry and grain size"""
            illum = self.illumination_angles[illum_idx]
            view = self.viewing_angles[view_idx]
            azim = self.relative_azimuths[azim_idx]
            grain = self.grain_radii[grain_idx]
            
            # Directional reflectance
            directional = self._simulate_snow_reflectance(
                grain, illum, view, azim
            )
            
            # Hemispherical (white-sky) reflectance
            hemispherical = self._simulate_snow_albedo(grain)
            
            # Anisotropy factor
            anisotropy = directional / (hemispherical + 1e-10)
            
            return anisotropy
        
        # Create meshgrid of indices
        illum_idx = np.arange(len(self.illumination_angles))
        view_idx = np.arange(len(self.viewing_angles))
        azim_idx = np.arange(len(self.relative_azimuths))
        grain_idx = np.arange(len(self.grain_radii))
        
        # Initialize array
        anisotropy_lut = np.zeros(shape, dtype=np.float32)
        
        # Compute with progress tracking
        total = len(illum_idx) * len(view_idx) * len(azim_idx) * len(grain_idx)
        count = 0
        
        for i in illum_idx:
            for j in view_idx:
                for k in azim_idx:
                    for g in grain_idx:
                        count += 1
                        if count % 100 == 0:
                            print(f"Progress: {count}/{total} ({100*count/total:.1f}%)")
                        
                        anisotropy_lut[i, j, k, g, :] = compute_chunk(i, j, k, g)
        
        return anisotropy_lut
    
    def _generate_anisotropy_numpy(self, shape: Tuple) -> np.ndarray:
        """Generate anisotropy LUT using NumPy"""
        anisotropy_lut = np.zeros(shape)
        
        total = np.prod(shape[:4])
        count = 0
        
        for i, illum in enumerate(self.illumination_angles):
            for j, view in enumerate(self.viewing_angles):
                for k, azim in enumerate(self.relative_azimuths):
                    for g, grain in enumerate(self.grain_radii):
                        count += 1
                        if count % 100 == 0:
                            print(f"Progress: {count}/{total}")
                        
                        # Directional reflectance
                        directional = self._simulate_snow_reflectance(
                            grain, illum, view, azim
                        )
                        
                        # Hemispherical reflectance
                        hemispherical = self._simulate_snow_albedo(grain)
                        
                        # Anisotropy factor
                        anisotropy_lut[i, j, k, g, :] = (
                            directional / (hemispherical + 1e-10)
                        )
        
        return anisotropy_lut
    
    def generate_albedo_lut(self,
                           output_path: Optional[Path] = None) -> np.ndarray:
        """
        Generate White-Sky Albedo lookup table
        
        Contains the spectral hemispherical (white-sky) albedo for each grain size
        
        Parameters:
        -----------
        output_path : Path, optional
            Path to save the LUT
            
        Returns:
        --------
        albedo_lut : np.ndarray
            2D array [n_grains, n_wavelengths]
        """
        print("Generating White-Sky Albedo LUT...")
        
        albedo_lut = np.zeros((len(self.grain_radii), len(self.wavelengths)))
        
        for g, grain_size in enumerate(self.grain_radii):
            if (g+1) % 10 == 0:
                print(f"Progress: {g+1}/{len(self.grain_radii)} grain sizes")
            
            # Simulate white-sky albedo
            albedo_lut[g, :] = self._simulate_snow_albedo(grain_size)
        
        if output_path:
            np.save(output_path, albedo_lut)
            print(f"Saved Albedo LUT to: {output_path}")
        
        return albedo_lut
    
    def _simulate_snow_reflectance(self,
                                  grain_size: float,
                                  illumination: float,
                                  viewing: float,
                                  relative_azimuth: float) -> np.ndarray:
        """
        Simulate directional snow reflectance using ART model
        
        Uses the Asymptotic Radiative Transfer (ART) model from 
        Kokhanovsky & Zege (2004) for accurate snow/ice reflectance.
        
        Parameters:
        -----------
        grain_size : float
            Effective grain radius (micrometers)
        illumination : float
            Illumination angle (degrees from zenith)
        viewing : float
            Viewing angle (degrees from zenith)
        relative_azimuth : float
            Relative azimuth angle (degrees)
            
        Returns:
        --------
        reflectance : np.ndarray
            Spectral BRDF [n_wavelengths]
        """
        # Use ART model for physically accurate simulation
        reflectance = simulate_snow_reflectance(
            wavelengths=self.wavelengths,
            grain_size_um=grain_size,
            solar_zenith=illumination,
            view_zenith=viewing,
            relative_azimuth=relative_azimuth,
            impurity_type=None,  # Clean snow for LUTs
            impurity_conc=0.0
        )
        
        return reflectance
    
    def _simulate_snow_albedo(self, grain_size: float) -> np.ndarray:
        """
        Simulate hemispherical (white-sky) snow albedo using ART model
        
        Uses the Asymptotic Radiative Transfer (ART) model for
        accurate spherical albedo calculation.
        
        Parameters:
        -----------
        grain_size : float
            Effective grain radius (micrometers)
            
        Returns:
        --------
        albedo : np.ndarray
            Spectral albedo [n_wavelengths]
        """
        # Use ART model for hemispherical albedo
        albedo = simulate_snow_albedo(
            wavelengths=self.wavelengths,
            grain_size_um=grain_size,
            impurity_type=None,  # Clean snow for LUTs
            impurity_conc=0.0
        )
        
        return albedo
    
    def _calculate_band_depth(self, spectrum: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate continuum-removed band depth for 1030 nm feature
        
        Parameters:
        -----------
        spectrum : np.ndarray
            Input spectrum
            
        Returns:
        --------
        continuum_removed : np.ndarray
            Continuum removed spectrum
        band_depth : float
            Scaled band depth at 1030 nm
        """
        # Find wavelength indices
        left_idx = np.argmin(np.abs(self.wavelengths - 950))
        center_idx = np.argmin(np.abs(self.wavelengths - 1030))
        
        # Find local maximum around 1100 nm
        search_start = np.argmin(np.abs(self.wavelengths - 1080))
        search_end = np.argmin(np.abs(self.wavelengths - 1120))
        local_max_idx = search_start + np.argmax(spectrum[search_start:search_end])
        
        # Create continuum line
        wl_continuum = [self.wavelengths[left_idx], self.wavelengths[local_max_idx]]
        r_continuum = [spectrum[left_idx], spectrum[local_max_idx]]
        
        continuum = np.interp(self.wavelengths, wl_continuum, r_continuum)
        
        # Continuum removal
        continuum_removed = spectrum / (continuum + 1e-10)
        
        # Band depth at 1030 nm
        band_depth = 1.0 - continuum_removed[center_idx]
        
        return continuum_removed, band_depth


def main():
    """
    Example LUT generation
    """
    from pathlib import Path
    
    # Define parameters to match your instrument
    # Example: AisaFENIX spectrometer (380-2500 nm, 451 bands)
    wavelengths = np.linspace(380, 2500, 451)
    
    # Grain size range: 30-5000 micrometers (logarithmic spacing)
    grain_radii = np.logspace(np.log10(30), np.log10(5000), 50)
    
    # Angular grids (balance resolution with computation time)
    illumination_angles = np.arange(0, 85, 5)  # 0-80° in 5° steps
    viewing_angles = np.arange(0, 65, 5)  # 0-60° in 5° steps
    relative_azimuths = np.arange(0, 185, 15)  # 0-180° in 15° steps
    
    print("Initializing LUT Generator...")
    print(f"Wavelengths: {len(wavelengths)}")
    print(f"Grain sizes: {len(grain_radii)} ({grain_radii[0]:.1f} - {grain_radii[-1]:.1f} μm)")
    print(f"Illumination angles: {len(illumination_angles)}")
    print(f"Viewing angles: {len(viewing_angles)}")
    print(f"Relative azimuths: {len(relative_azimuths)}")
    
    generator = ISSIALUTGenerator(
        wavelengths=wavelengths,
        grain_radii=grain_radii,
        illumination_angles=illumination_angles,
        viewing_angles=viewing_angles,
        relative_azimuths=relative_azimuths
    )
    
    output_dir = Path("lookup_tables")
    output_dir.mkdir(exist_ok=True)
    
    # Generate LUTs
    print("\n" + "="*60)
    print("Generating Lookup Tables")
    print("="*60)
    
    # 1. Scaled Band Depth LUT (relatively small)
    print("\n1. Generating Scaled Band Depth LUT...")
    sbd_lut = generator.generate_sbd_lut(
        output_path=output_dir / "sbd_lut.npy"
    )
    print(f"SBD LUT shape: {sbd_lut.shape}")
    
    # 2. Albedo LUT (small)
    print("\n2. Generating White-Sky Albedo LUT...")
    albedo_lut = generator.generate_albedo_lut(
        output_path=output_dir / "albedo_lut.npy"
    )
    print(f"Albedo LUT shape: {albedo_lut.shape}")
    
    # 3. Anisotropy Factor LUT (large - use Dask)
    print("\n3. Generating Anisotropy Factor LUT...")
    print("WARNING: This may take significant time and memory!")
    response = input("Continue? (y/n): ")
    
    if response.lower() == 'y':
        anisotropy_lut = generator.generate_anisotropy_lut(
            output_path=output_dir / "anisotropy_lut.npy",
            use_dask=True
        )
        print(f"Anisotropy LUT shape: {anisotropy_lut.shape}")
    else:
        print("Skipping anisotropy LUT generation")
    
    print("\n" + "="*60)
    print("LUT Generation Complete!")
    print("="*60)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
