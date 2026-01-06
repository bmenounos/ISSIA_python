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
        milestones = [int(total * p / 100) for p in [20, 40, 60, 80, 100]]
        
        for i, illum in enumerate(self.illumination_angles):
            for j, view in enumerate(self.viewing_angles):
                for k, azim in enumerate(self.relative_azimuths):
                    count += 1
                    if count in milestones:
                        print(f"Progress: {int(100 * count / total)}%")
                    
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
        milestones = [int(total * p / 100) for p in [20, 40, 60, 80, 100]]
        
        for i in illum_idx:
            for j in view_idx:
                for k in azim_idx:
                    for g in grain_idx:
                        count += 1
                        if count in milestones:
                            print(f"Progress: {int(100 * count / total)}%")
                        
                        anisotropy_lut[i, j, k, g, :] = compute_chunk(i, j, k, g)
        
        return anisotropy_lut
    
    def _generate_anisotropy_numpy(self, shape: Tuple) -> np.ndarray:
        """Generate anisotropy LUT using NumPy"""
        anisotropy_lut = np.zeros(shape)
        
        total = np.prod(shape[:4])
        count = 0
        milestones = [int(total * p / 100) for p in [20, 40, 60, 80, 100]]
        
        for i, illum in enumerate(self.illumination_angles):
            for j, view in enumerate(self.viewing_angles):
                for k, azim in enumerate(self.relative_azimuths):
                    for g, grain in enumerate(self.grain_radii):
                        count += 1
                        if count in milestones:
                            print(f"Progress: {int(100 * count / total)}%")
                        
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
        milestones = [int(len(self.grain_radii) * p / 100) for p in [20, 40, 60, 80, 100]]
        
        for g, grain_size in enumerate(self.grain_radii):
            if (g+1) in milestones:
                print(f"Progress: {int(100 * (g+1) / len(self.grain_radii))}%")
            
            # Simulate white-sky albedo
            albedo_lut[g, :] = self._simulate_snow_albedo(grain_size)
        
        if output_path:
            np.save(output_path, albedo_lut)
            print(f"Saved Albedo LUT to: {output_path}")
        
        return albedo_lut
    
    def _load_ice_refractive_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load Warren 2008 ice refractive index and resample to sensor wavelengths"""
        # Load refractive index data
        data = np.loadtxt('IOP_2008_ASCIItable.txt')
        
        # Extract 350-2600 nm range (MATLAB lines 97:265 → Python 96:265)
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
        """
        Simulate snow BRF using Kokhanovsky & Brieglieb 2012 ART model
        """
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
        """
        Simulate hemispherical albedo using ART model
        """
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
    
    def _calculate_band_depth(self, spectrum: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate continuum-removed band depth for 1030 nm feature using convex hull
        
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
        from scipy.spatial import ConvexHull
        
        # Extract 900-1130nm range (matches MATLAB driver_ART_LUT_generator.m)
        #left_idx = np.argmin(np.abs(self.wavelengths - 900))
        left_idx = np.argmin(np.abs(self.wavelengths - 830))
        right_idx = np.argmin(np.abs(self.wavelengths - 1130))
        
        wl_subset = self.wavelengths[left_idx:right_idx+1]
        spec_subset = spectrum[left_idx:right_idx+1]
        
        # Apply convex hull method
        spec_ext = np.concatenate([[0], spec_subset, [0]])
        x_ext = np.arange(len(spec_ext))
        
        points = np.column_stack([x_ext, spec_ext])
        try:
            hull = ConvexHull(points)
            
            K = hull.vertices.copy()
            K = K[2:]   # Remove first 2
            K = K[:-1]  # Remove last
            K = np.sort(K)  # Sort
            K = K - 1   # Adjust indices
            
            continuum = np.interp(np.arange(len(spec_subset)), K, spec_subset[K])
            continuum = np.where(continuum < 1e-10, 1e-10, continuum)
            
            continuum_removed = spec_subset / continuum
            
        except:
            # Fallback
            continuum = np.linspace(spec_subset[0], spec_subset[-1], len(spec_subset))
            continuum_removed = spec_subset / (continuum + 1e-10)
        
        # Band depth as min of CR spectrum (matches MATLAB)
        band_depth = 1.0 - continuum_removed.min()
        
        return continuum_removed, band_depth


def main():
    """
    Example LUT generation
    """
    from pathlib import Path

    import spectral.io.envi as envi


    data_dir = Path('/Volumes/aco-uvic/2022_Acquisitions/02_Processed/22_4012_07_PlaceGlacier/03_Hyper/02_Working/OUTPUT/subsets/')
    flight_line = '22_4012_07_2022-08-07_19-54-01-rect_img'
    
    hdr = envi.open(f"{data_dir}/{flight_line}_atm.hdr")
    wvl = [float(w) for w in hdr.metadata['wavelength']]
    
    wavelengths = np.array([float(w) for w in hdr.metadata['wavelength']])

    
    # Define parameters to match your instrument
    # Example: AisaFENIX spectrometer (380-2500 nm, 451 bands)
    #wavelengths = np.linspace(380, 2500, 451)
    
    # Grain size range: 30-5000 micrometers (linear spacing, 30 μm steps)
    grain_radii = np.arange(30, 5001, 30)  # Matches MATLAB: 30:30:5000
    
    # Angular grids (match MATLAB ISSIA)
    illumination_angles = np.arange(0, 86, 5)  # 0:5:85 = 18 values
    viewing_angles = np.arange(0, 86, 5)       # 0:5:85 = 18 values  
    relative_azimuths = np.arange(0, 361, 10)  # 0:10:360 = 37 values
    
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
