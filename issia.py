"""
ISSIA - Imaging Spectrometer Snow and Ice Algorithm
Python implementation with Dask parallelization

This code produces three surface property retrievals on a per flight line basis:
- Broadband albedo (BBA)
- Optical grain radius
- Radiative forcing by light absorbing particles (LAPs)

Based on the methodology described in:
Donahue et al. (2023), "Bridging the gap between airborne and spaceborne 
imaging spectroscopy for mountain glacier surface property retrievals"
Remote Sensing of Environment, 299:113849

Author: Python translation from MATLAB
Original MATLAB code by Christopher Donahue
"""

import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
import xarray as xr
import rasterio
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import map_coordinates

class ISSIAProcessor:
    """
    Main processor for ISSIA surface property retrievals
    """
    
    def __init__(self, 
                 wavelengths: np.ndarray,
                 grain_radii: np.ndarray,
                 illumination_angles: np.ndarray,
                 viewing_angles: np.ndarray,
                 relative_azimuths: np.ndarray,
                 coord_ref_sys_code: int = 32610,
                 chunk_size: Tuple[int, int] = (1024, 1024)):
        """
        Initialize ISSIA processor
        
        Parameters:
        -----------
        wavelengths : np.ndarray
            Wavelength array (nm)
        grain_radii : np.ndarray
            Grain radius array for LUTs (micrometers)
        illumination_angles : np.ndarray
            Illumination angle array for LUTs (degrees)
        viewing_angles : np.ndarray
            Viewing angle array for LUTs (degrees)
        relative_azimuths : np.ndarray
            Relative azimuth angle array for LUTs (degrees)
        coord_ref_sys_code : int
            EPSG code for coordinate reference system
        chunk_size : tuple
            Dask chunk size for processing (rows, cols)
        """
        self.wavelengths = wavelengths
        self.grain_radii = grain_radii
        self.illumination_angles = illumination_angles
        self.viewing_angles = viewing_angles
        self.relative_azimuths = relative_azimuths
        self.coord_ref_sys_code = coord_ref_sys_code
        self.chunk_size = chunk_size
        
        # LUTs will be loaded separately
        self.sbd_lut = None
        self.anisotropy_lut = None
        self.albedo_lut = None
        
    def load_lookup_tables(self, 
                          sbd_lut_path: Path,
                          anisotropy_lut_path: Path,
                          albedo_lut_path: Path):
        """
        Load pre-computed lookup tables
        
        Parameters:
        -----------
        sbd_lut_path : Path
            Path to scaled band depth LUT (4D array)
        anisotropy_lut_path : Path
            Path to anisotropy factor LUT (5D array)
        albedo_lut_path : Path
            Path to albedo LUT (2D array)
        """
        print("Loading lookup tables...")
        self.sbd_lut = np.load(sbd_lut_path)
        self.anisotropy_lut = np.load(anisotropy_lut_path, mmap_mode='r')  # Memory map for large file
        self.albedo_lut = np.load(albedo_lut_path)
        print("Lookup tables loaded successfully")
        
    def read_atcor_files(self, data_dir: Path, flight_line: str) -> Dict:
        """
        Read ATCOR-4 output files for a flight line
        
        Parameters:
        -----------
        data_dir : Path
            Directory containing ATCOR-4 output files
        flight_line : str
            Flight line identifier
            
        Returns:
        --------
        dict : Dictionary containing all input data arrays as Dask arrays
        """
        print(f"Reading ATCOR files for flight line: {flight_line}")
        
        # Read reflectance file (atm.dat)
        atm_file = data_dir / f"{flight_line}_atm.dat"
        reflectance = self._read_envi_file(atm_file)
        
        # Read global solar flux (eglo.dat)
        eglo_file = data_dir / f"{flight_line}_eglo.dat"
        global_flux = self._read_envi_file(eglo_file)
        
        # Read slope (slp.dat)
        slp_file = data_dir / f"{flight_line}_slp.dat"
        slope = self._read_single_band(slp_file)
        
        # Read aspect (asp.dat)
        asp_file = data_dir / f"{flight_line}_asp.dat"
        aspect = self._read_single_band(asp_file)
        
        # Read solar zenith and azimuth angles from .inn file
        inn_file = data_dir / f"{flight_line}.inn"
        solar_zenith, solar_azimuth = self._read_solar_zenith(inn_file)
        
        # Get geotransform and projection
        with rasterio.open(atm_file.with_suffix('.hdr')) as src:
            transform = src.transform
            crs = src.crs
            
        return {
            'reflectance': reflectance,
            'global_flux': global_flux,
            'slope': slope,
            'aspect': aspect,
            'solar_zenith': solar_zenith,
            'solar_azimuth': solar_azimuth,
            'transform': transform,
            'crs': crs
        }
    
    def _read_envi_file(self, filepath: Path, window=None) -> da.Array:
        """Read ENVI format file and return as Dask array"""
        # Ensure we're opening the .dat file, not .hdr
        filepath = Path(filepath)
        if filepath.suffix.lower() == '.hdr':
            filepath = filepath.with_suffix('.dat')
        
        # Read scale factor from header
        hdr_file = str(filepath).replace('.dat', '.hdr')
        scale_factor = 1.0
        try:
            with open(hdr_file, 'r') as f:
                for line in f:
                    if 'reflectance scale factor' in line.lower():
                        scale_factor = float(line.split('=')[1].strip())
                        break
        except (FileNotFoundError, OSError):
            # Header file not found or can't be read - use default scale factor
            pass
        
        with rasterio.open(str(filepath)) as src:
            if window:
                data = src.read(window=window)
            else:
                data = src.read()
            # Apply scale factor
            data = data.astype(float) / scale_factor
            # Convert zeros to NaN (data ignore value)
            data = np.where(data == 0, np.nan, data)
            # Convert to Dask array with chunking
            dask_data = da.from_array(data, chunks=(data.shape[0], 
                                                     self.chunk_size[0], 
                                                     self.chunk_size[1]))
        return dask_data
    
    def _read_single_band(self, filepath: Path, window=None) -> da.Array:
        """Read single band file and return as Dask array"""
        # Ensure we're opening the .dat file, not .hdr
        filepath = Path(filepath)
        if filepath.suffix.lower() == '.hdr':
            filepath = filepath.with_suffix('.dat')
        
        with rasterio.open(str(filepath)) as src:
            if window:
                data = src.read(1, window=window)
            else:
                data = src.read(1)
            dask_data = da.from_array(data, chunks=self.chunk_size)
        return dask_data
    
    def _read_solar_zenith(self, inn_file: Path) -> tuple:
        """
        Extract solar zenith and azimuth angles from ATCOR .inn file
        
        Returns:
        --------
        tuple : (solar_zenith, solar_azimuth) in degrees
        """
        with open(inn_file, 'r') as f:
            content = f.read()
            
        solar_zenith = None
        solar_azimuth = None
        
        # Look for line with "Solar zenith, azimuth"
        # Format: "38.1   177.2       Solar zenith, azimuth [degree]"
        for line in content.split('\n'):
            if 'solar zenith' in line.lower() and 'azimuth' in line.lower():
                # Extract two numbers before the text
                parts = line.split()
                try:
                    solar_zenith = float(parts[0])
                    solar_azimuth = float(parts[1])
                    break
                except (ValueError, IndexError):
                    continue
        
        if solar_zenith is None:
            raise ValueError(f"Could not find solar zenith angle in {inn_file}")
        if solar_azimuth is None:
            raise ValueError(f"Could not find solar azimuth angle in {inn_file}")
            
        return solar_zenith, solar_azimuth
    
    def continuum_removal(self, spectrum: np.ndarray, 
                         wavelengths: np.ndarray,
                         left_wl: float = 950.0,
                         right_wl: float = 1100.0,
                         center_wl: float = 1030.0) -> Tuple[np.ndarray, float]:
        """
        Perform continuum removal on spectrum for 1030 nm ice absorption feature
        
        Parameters:
        -----------
        spectrum : np.ndarray
            Input spectrum
        wavelengths : np.ndarray
            Wavelength array (nm)
        left_wl : float
            Left wavelength for continuum (nm)
        right_wl : float
            Right wavelength for continuum (nm)
        center_wl : float
            Center of absorption feature (nm)
            
        Returns:
        --------
        continuum_removed : np.ndarray
            Continuum removed spectrum
        band_depth : float
            Scaled band depth at center wavelength
        """
        # Find indices for wavelengths
        left_idx = np.argmin(np.abs(wavelengths - left_wl))
        right_idx = np.argmin(np.abs(wavelengths - right_wl))
        center_idx = np.argmin(np.abs(wavelengths - center_wl))
        
        # Get subset of spectrum and wavelengths
        wl_subset = wavelengths[left_idx:right_idx+1]
        spec_subset = spectrum[left_idx:right_idx+1]
        
        # Find local maximum around right boundary (shoulder of feature)
        # Search in region around 1100 nm
        search_start = np.argmin(np.abs(wavelengths - 1080.0))
        search_end = np.argmin(np.abs(wavelengths - 1120.0))
        local_max_idx = search_start + np.argmax(spectrum[search_start:search_end])
        
        # Create continuum line
        continuum = np.interp(wl_subset, 
                             [wavelengths[left_idx], wavelengths[local_max_idx]],
                             [spectrum[left_idx], spectrum[local_max_idx]])
        
        # Continuum removal
        continuum_removed = spec_subset / continuum
        
        # Calculate scaled band depth at center wavelength
        center_idx_local = np.argmin(np.abs(wl_subset - center_wl))
        band_depth = 1.0 - continuum_removed[center_idx_local]
        
        return continuum_removed, band_depth
    
    def calculate_local_illumination_angle(self, 
                                          solar_zenith: float,
                                          solar_azimuth: float,
                                          slope: da.Array,
                                          aspect: da.Array) -> da.Array:
        """
        Calculate local illumination angle accounting for topography
        
        Parameters:
        -----------
        solar_zenith : float
            Solar zenith angle (degrees)
        solar_azimuth : float
            Solar azimuth angle (degrees)
        slope : da.Array
            Terrain slope (degrees)
        aspect : da.Array
            Terrain aspect (degrees)
            
        Returns:
        --------
        local_illum : da.Array
            Local illumination angle (degrees)
        """
        # Convert to radians
        theta_s = np.deg2rad(solar_zenith)
        phi_s = np.deg2rad(solar_azimuth)
        slope_rad = da.deg2rad(slope)
        aspect_rad = da.deg2rad(aspect)
        
        # Calculate local illumination angle using cosine formula
        cos_local_illum = (da.cos(theta_s) * da.cos(slope_rad) + 
                          da.sin(theta_s) * da.sin(slope_rad) * 
                          da.cos(phi_s - aspect_rad))
        
        # Clip to valid range and convert back to degrees
        cos_local_illum = da.clip(cos_local_illum, -1.0, 1.0)
        local_illum = da.rad2deg(da.arccos(cos_local_illum))
        
        return local_illum
    
    def retrieve_grain_size(self,
                           band_depth_map: da.Array,
                           local_illum: da.Array,
                           viewing_angle: float = 0.0,
                           relative_azimuth: float = 0.0) -> da.Array:
        """
        Retrieve grain size from scaled band depth using LUT
        
        Parameters:
        -----------
        band_depth_map : da.Array
            Map of scaled band depths
        local_illum : da.Array
            Local illumination angles (degrees)
        viewing_angle : float
            Sensor viewing angle (degrees), typically nadir (0)
        relative_azimuth : float
            Relative azimuth angle (degrees)
            
        Returns:
        --------
        grain_size : da.Array
            Retrieved grain size (micrometers)
        """
        def lookup_grain_size(bd, illum):
            """Lookup grain size for given band depth and illumination"""
            # Handle NaN values
            if np.isnan(bd) or np.isnan(illum):
                return np.nan
            
            # Find closest indices in LUT
            illum_idx = np.argmin(np.abs(self.illumination_angles - illum))
            view_idx = np.argmin(np.abs(self.viewing_angles - viewing_angle))
            azim_idx = np.argmin(np.abs(self.relative_azimuths - relative_azimuth))
            
            # Get band depth vs grain size for this geometry
            bd_curve = self.sbd_lut[illum_idx, view_idx, azim_idx, :]
            
            # Find grain size that matches observed band depth
            if bd >= bd_curve.max():
                return self.grain_radii[-1]  # Maximum grain size
            elif bd <= bd_curve.min():
                return self.grain_radii[0]   # Minimum grain size
            else:
                # Interpolate
                grain_size = np.interp(bd, bd_curve, self.grain_radii)
                return grain_size
        
        # Use map_blocks instead of vectorize (removed in newer Dask)
        def lookup_wrapper(bd_block, illum_block):
            result = np.zeros_like(bd_block, dtype=float)
            for i in range(bd_block.shape[0]):
                for j in range(bd_block.shape[1]):
                    result[i, j] = lookup_grain_size(bd_block[i, j], illum_block[i, j])
            return result
        
        grain_size = da.map_blocks(lookup_wrapper, 
                                   band_depth_map, 
                                   local_illum,
                                   dtype=float)
        
        return grain_size
    
    def calculate_anisotropy_factor(self,
                                   grain_size: da.Array,
                                   local_illum: da.Array,
                                   viewing_angle: float = 0.0,
                                   relative_azimuth: float = 0.0) -> da.Array:
        """
        Calculate spectral anisotropy factor from LUT
        
        Parameters:
        -----------
        grain_size : da.Array
            Grain size map (micrometers)
        local_illum : da.Array
            Local illumination angles (degrees)
        viewing_angle : float
            Sensor viewing angle (degrees)
        relative_azimuth : float
            Relative azimuth angle (degrees)
            
        Returns:
        --------
        anisotropy : da.Array
            Spectral anisotropy factor (n_wavelengths, rows, cols)
        """
        def lookup_anisotropy(gs, illum):
            """Lookup anisotropy for given grain size and geometry"""
            if np.isnan(gs) or np.isnan(illum):
                return np.full(len(self.wavelengths), np.nan)
            
            # Find closest indices
            illum_idx = np.argmin(np.abs(self.illumination_angles - illum))
            view_idx = np.argmin(np.abs(self.viewing_angles - viewing_angle))
            azim_idx = np.argmin(np.abs(self.relative_azimuths - relative_azimuth))
            gs_idx = np.argmin(np.abs(self.grain_radii - gs))
            
            # Get anisotropy spectrum for this geometry and grain size
            aniso_spectrum = self.anisotropy_lut[illum_idx, view_idx, azim_idx, gs_idx, :]
            
            return aniso_spectrum
        
        # Use apply_gufunc for cleaner dimension handling
        def lookup_pixel_aniso(gs, illum):
            """Lookup anisotropy for a single pixel - returns spectrum"""
            if np.isnan(gs) or np.isnan(illum):
                return np.full(len(self.wavelengths), np.nan)
            
            illum_idx = np.argmin(np.abs(self.illumination_angles - illum))
            view_idx = np.argmin(np.abs(self.viewing_angles - viewing_angle))
            azim_idx = np.argmin(np.abs(self.relative_azimuths - relative_azimuth))
            gs_idx = np.argmin(np.abs(self.grain_radii - gs))
            
            return self.anisotropy_lut[illum_idx, view_idx, azim_idx, gs_idx, :]
        
        # Compute grain size and illumination (small arrays, can fit in memory)
        gs_array = grain_size.compute()
        illum_array = local_illum.compute()
        
        # Vectorized lookup
        from numpy import vectorize
        vec_lookup = vectorize(lookup_pixel_aniso, signature='(),()->(n)')
        
        # Apply lookup - output is (rows, cols, wavelengths)
        aniso_array = vec_lookup(gs_array, illum_array)
        
        # Transpose to (wavelengths, rows, cols)
        aniso_array = np.transpose(aniso_array, (2, 0, 1))
        
        # Convert back to dask
        anisotropy = da.from_array(aniso_array, chunks=(len(self.wavelengths), self.chunk_size[0], self.chunk_size[1]))
        
        return anisotropy
    
    def calculate_spectral_albedo(self,
                                 grain_size: da.Array,
                                 reflectance_hdrf: da.Array,
                                 anisotropy_factor: da.Array) -> da.Array:
        """
        Calculate spectral albedo from grain size
        
        Parameters:
        -----------
        grain_size : da.Array
            Grain size map (micrometers)
        reflectance_hdrf : da.Array
            HDRF corrected reflectance (n_bands, rows, cols)
        anisotropy_factor : da.Array
            Spectral anisotropy factor (n_bands, rows, cols)
            
        Returns:
        --------
        spectral_albedo : da.Array
            Spectral albedo (n_bands, rows, cols)
        """
        def lookup_albedo(gs):
            """Lookup white-sky albedo for given grain size"""
            if np.isnan(gs):
                return np.full(len(self.wavelengths), np.nan)
            
            gs_idx = np.argmin(np.abs(self.grain_radii - gs))
            return self.albedo_lut[gs_idx, :]
        
        # Use vectorize for cleaner dimension handling
        def lookup_pixel_albedo(gs):
            """Lookup white-sky albedo for a single pixel - returns spectrum"""
            if np.isnan(gs):
                return np.full(len(self.wavelengths), np.nan)
            
            gs_idx = np.argmin(np.abs(self.grain_radii - gs))
            return self.albedo_lut[gs_idx, :]
        
        # Compute grain size
        gs_array = grain_size.compute()
        
        # Vectorized lookup
        from numpy import vectorize
        vec_lookup = vectorize(lookup_pixel_albedo, signature='()->(n)')
        
        # Apply - output is (rows, cols, wavelengths)
        albedo_array = vec_lookup(gs_array)
        
        # Transpose to (wavelengths, rows, cols)
        albedo_array = np.transpose(albedo_array, (2, 0, 1))
        
        # Convert back to dask
        white_sky_albedo = da.from_array(albedo_array, chunks=(len(self.wavelengths), self.chunk_size[0], self.chunk_size[1]))
        
        # Calculate spectral albedo
        spectral_albedo = white_sky_albedo * (reflectance_hdrf / anisotropy_factor)
        
        return spectral_albedo
    
    def calculate_broadband_albedo(self,
                                  spectral_albedo: da.Array,
                                  solar_spectrum: np.ndarray) -> da.Array:
        """
        Calculate broadband albedo by integrating spectral albedo
        
        Parameters:
        -----------
        spectral_albedo : da.Array
            Spectral albedo (n_bands, rows, cols)
        solar_spectrum : np.ndarray
            Solar irradiance spectrum at ground (n_bands,)
            
        Returns:
        --------
        broadband_albedo : da.Array
            Broadband albedo (rows, cols)
        """
        # Integration weights from solar spectrum
        weights = solar_spectrum / solar_spectrum.sum()
        weights = weights.reshape(-1, 1, 1)
        
        # Weighted sum over wavelengths
        broadband_albedo = da.sum(spectral_albedo * weights, axis=0)
        
        return broadband_albedo
    
    def calculate_radiative_forcing(self,
                                   spectral_albedo_clean: da.Array,
                                   spectral_albedo_actual: da.Array,
                                   solar_spectrum: np.ndarray) -> da.Array:
        """
        Calculate radiative forcing by light absorbing particles
        
        Parameters:
        -----------
        spectral_albedo_clean : da.Array
            Spectral albedo for clean snow (n_bands, rows, cols)
        spectral_albedo_actual : da.Array
            Actual spectral albedo (n_bands, rows, cols)
        solar_spectrum : np.ndarray
            Solar irradiance at ground (n_bands,)
            
        Returns:
        --------
        rf_lap : da.Array
            Radiative forcing by LAPs (W/m²) (rows, cols)
        """
        # Calculate difference in absorbed radiation
        # RF_LAP = sum(solar * (albedo_clean - albedo_actual))
        
        weights = solar_spectrum.reshape(-1, 1, 1)
        
        albedo_diff = spectral_albedo_clean - spectral_albedo_actual
        rf_lap = da.sum(weights * albedo_diff, axis=0)
        
        return rf_lap
    
    def process_flight_line(self,
                          data_dir: Path,
                          flight_line: str,
                          output_dir: Path,
                          viewing_angle: float = 0.0,
                          solar_azimuth: float = 180.0) -> Dict[str, Path]:
        """
        Process a single flight line and generate retrievals
        
        Parameters:
        -----------
        data_dir : Path
            Directory containing input ATCOR files
        flight_line : str
            Flight line identifier
        output_dir : Path
            Directory for output GeoTIFFs
        viewing_angle : float
            Sensor viewing angle (degrees from nadir)
        solar_azimuth : float
            Solar azimuth angle (degrees)
            
        Returns:
        --------
        output_files : dict
            Dictionary of output file paths
        """
        print(f"\nProcessing flight line: {flight_line}")
        print("="*50)
        
        # Read input files
        data = self.read_atcor_files(data_dir, flight_line)
        
        # Extract arrays
        reflectance = data['reflectance']  # (n_bands, rows, cols)
        global_flux = data['global_flux']
        slope = data['slope']
        aspect = data['aspect']
        solar_zenith = data['solar_zenith']
        
        print(f"Data shape: {reflectance.shape}")
        print(f"Solar zenith angle: {solar_zenith:.2f}°")
        
        # Calculate local illumination angle
        print("Calculating local illumination angles...")
        local_illum = self.calculate_local_illumination_angle(
            solar_zenith, solar_azimuth, slope, aspect
        )
        
        # Calculate scaled band depth for each pixel
        print("Calculating scaled band depths...")
        band_depths = da.map_blocks(
            lambda spec_block: np.array([
                [self.continuum_removal(spec_block[:, i, j], self.wavelengths)[1]
                 for j in range(spec_block.shape[2])]
                for i in range(spec_block.shape[1])
            ]),
            reflectance,
            dtype=float,
            drop_axis=0,
            chunks=(self.chunk_size[0], self.chunk_size[1])
        )
        
        # Retrieve grain size
        print("Retrieving grain sizes...")
        grain_size = self.retrieve_grain_size(band_depths, local_illum, 
                                             viewing_angle, 0.0)
        
        # Calculate anisotropy factor
        print("Calculating anisotropy factors...")
        anisotropy = self.calculate_anisotropy_factor(
            grain_size, local_illum, viewing_angle, 0.0
        )
        
        # Convert HCRF to HDRF using anisotropy
        reflectance_hdrf = reflectance * anisotropy
        
        # Calculate spectral albedo
        print("Calculating spectral albedo...")
        spectral_albedo = self.calculate_spectral_albedo(
            grain_size, reflectance_hdrf, anisotropy
        )
        
        # Calculate broadband albedo
        print("Calculating broadband albedo...")
        # Use mean global flux across bands as solar spectrum
        mean_flux = global_flux.mean(axis=(1, 2)).compute()
        broadband_albedo = self.calculate_broadband_albedo(
            spectral_albedo, mean_flux
        )
        
        # Calculate radiative forcing (compare to clean snow with same grain size)
        print("Calculating radiative forcing...")
        # For clean snow, use the white-sky albedo directly
        clean_albedo = self.calculate_spectral_albedo(
            grain_size, reflectance_hdrf * 0 + anisotropy, anisotropy
        )
        rf_lap = self.calculate_radiative_forcing(
            clean_albedo, spectral_albedo, mean_flux
        )
        
        # Compute results with progress bar
        print("\nComputing results...")
        with ProgressBar():
            grain_size_result = grain_size.compute()
            broadband_albedo_result = broadband_albedo.compute()
            rf_lap_result = rf_lap.compute()
        
        # Save outputs as GeoTIFFs
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        # Save grain size
        grain_size_file = output_dir / f"{flight_line}_grain_size.tif"
        self._save_geotiff(grain_size_result, grain_size_file, 
                          data['transform'], data['crs'])
        output_files['grain_size'] = grain_size_file
        
        # Save broadband albedo
        albedo_file = output_dir / f"{flight_line}_broadband_albedo.tif"
        self._save_geotiff(broadband_albedo_result, albedo_file,
                          data['transform'], data['crs'])
        output_files['broadband_albedo'] = albedo_file
        
        # Save radiative forcing
        rf_file = output_dir / f"{flight_line}_radiative_forcing.tif"
        self._save_geotiff(rf_lap_result, rf_file,
                          data['transform'], data['crs'])
        output_files['radiative_forcing'] = rf_file
        
        print(f"\nProcessing complete for {flight_line}")
        print(f"Outputs saved to: {output_dir}")
        
        return output_files
    
    def _save_geotiff(self, data: np.ndarray, filepath: Path,
                     transform, crs):
        """Save array as GeoTIFF with georeferencing"""
        with rasterio.open(
            filepath, 'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            compress='lzw',
            nodata=np.nan
        ) as dst:
            dst.write(data, 1)
        
        print(f"Saved: {filepath}")


def main():
    """
    Example usage of ISSIA processor
    """
    from pathlib import Path
    
    # Define wavelength array (example for AisaFENIX 380-2500 nm, 451 bands)
    wavelengths = np.linspace(380, 2500, 451)
    
    # Define LUT dimensions (these should match your actual LUTs)
    grain_radii = np.logspace(np.log10(30), np.log10(5000), 100)  # 30-5000 μm
    illumination_angles = np.arange(0, 85, 5)  # 0-80° in 5° steps
    viewing_angles = np.arange(0, 65, 5)  # 0-60° in 5° steps  
    relative_azimuths = np.arange(0, 185, 15)  # 0-180° in 15° steps
    
    # Initialize processor
    processor = ISSIAProcessor(
        wavelengths=wavelengths,
        grain_radii=grain_radii,
        illumination_angles=illumination_angles,
        viewing_angles=viewing_angles,
        relative_azimuths=relative_azimuths,
        coord_ref_sys_code=32610,  # WGS84 / UTM Zone 10N
        chunk_size=(512, 512)
    )
    
    # Load lookup tables
    lut_dir = Path("lookup_tables")
    processor.load_lookup_tables(
        sbd_lut_path=lut_dir / "sbd_lut.npy",
        anisotropy_lut_path=lut_dir / "anisotropy_lut.npy",
        albedo_lut_path=lut_dir / "albedo_lut.npy"
    )
    
    # Process flight line(s)
    data_dir = Path("atcor_output")
    output_dir = Path("issia_results")
    
    flight_lines = ["flight_line_001", "flight_line_002"]
    
    for flight_line in flight_lines:
        try:
            output_files = processor.process_flight_line(
                data_dir=data_dir,
                flight_line=flight_line,
                output_dir=output_dir,
                viewing_angle=0.0,
                solar_azimuth=180.0
            )
            print(f"\nGenerated files:")
            for key, filepath in output_files.items():
                print(f"  {key}: {filepath}")
        except Exception as e:
            print(f"Error processing {flight_line}: {e}")
            continue


if __name__ == "__main__":
    main()
