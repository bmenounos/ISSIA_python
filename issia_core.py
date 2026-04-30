"""
ISSIA - Imaging Spectrometer Snow and Ice Algorithm
Python implementation

Core processor class providing base functionality for:
- ATCOR file I/O
- Local illumination angle calculation
- LUT-based grain size retrieval
- Spectral albedo and radiative forcing calculation
"""

import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
import rasterio
from pathlib import Path
from typing import Tuple, Dict, Optional
from functools import partial


# ============================================================================
# TOP-LEVEL FUNCTIONS FOR PICKLING (must be outside class for multiprocessing)
# ============================================================================

def _lookup_grain_size_block(bd_block, illum_block, sbd_lut, grain_radii,
                             illumination_angles, viewing_angles,
                             relative_azimuths, viewing_angle, relative_azimuth):
    """Lookup grain size for block of pixels - top-level for pickling"""
    min_i = min(bd_block.shape[0], illum_block.shape[0])
    min_j = min(bd_block.shape[1], illum_block.shape[1])
    result = np.zeros((min_i, min_j), dtype=float)
    
    for i in range(min_i):
        for j in range(min_j):
            bd = bd_block[i, j]
            illum = illum_block[i, j]
            
            if np.isnan(bd) or np.isnan(illum):
                result[i, j] = np.nan
                continue
            
            illum_idx = np.argmin(np.abs(illumination_angles - illum))
            view_idx = np.argmin(np.abs(viewing_angles - viewing_angle))
            azim_idx = np.argmin(np.abs(relative_azimuths - relative_azimuth))
            
            bd_curve = sbd_lut[illum_idx, view_idx, azim_idx, :].ravel()
            grain_radii_1d = np.asarray(grain_radii).ravel()
            
            if bd >= bd_curve.max():
                result[i, j] = grain_radii_1d[-1]
            elif bd <= bd_curve.min():
                result[i, j] = grain_radii_1d[0]
            else:
                result[i, j] = np.interp(bd, bd_curve, grain_radii_1d)
    
    return result


def _lookup_anisotropy_block(gs_block, illum_block, anisotropy_lut, grain_radii,
                             illumination_angles, viewing_angles,
                             relative_azimuths, viewing_angle, 
                             relative_azimuth, n_wavelengths):
    """Lookup anisotropy for block of pixels"""
    rows, cols = gs_block.shape
    result = np.zeros((n_wavelengths, rows, cols), dtype=float)
    
    for i in range(rows):
        for j in range(cols):
            gs = gs_block[i, j]
            illum = illum_block[i, j]
            
            if np.isnan(gs) or np.isnan(illum):
                result[:, i, j] = np.nan
            else:
                illum_idx = np.argmin(np.abs(illumination_angles - illum))
                view_idx = np.argmin(np.abs(viewing_angles - viewing_angle))
                azim_idx = np.argmin(np.abs(relative_azimuths - relative_azimuth))
                gs_idx = np.argmin(np.abs(grain_radii - gs))
                
                result[:, i, j] = anisotropy_lut[illum_idx, view_idx, azim_idx, gs_idx, :]
    
    return result


# ============================================================================
# ISSIAProcessor class
# ============================================================================

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
                 chunk_size: Tuple[int, int] = (1024, 1024)):
        """
        Initialize ISSIA processor
        
        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelength array in nm
        grain_radii : np.ndarray
            Grain radius array in micrometers (must match LUT)
        illumination_angles : np.ndarray
            Illumination angle array in degrees (must match LUT)
        viewing_angles : np.ndarray
            Viewing angle array in degrees (must match LUT)
        relative_azimuths : np.ndarray
            Relative azimuth array in degrees (must match LUT)
        chunk_size : tuple
            Dask chunk size for processing (rows, cols)
        """
        self.wavelengths = wavelengths
        self.grain_radii = grain_radii
        self.illumination_angles = illumination_angles
        self.viewing_angles = viewing_angles
        self.relative_azimuths = relative_azimuths
        self.chunk_size = chunk_size
        
        self.sbd_lut = None
        self.anisotropy_lut = None
        self.albedo_lut = None
        
    def load_lookup_tables(self, 
                          sbd_lut_path: Path,
                          anisotropy_lut_path: Path,
                          albedo_lut_path: Path):
        """
        Load pre-computed lookup tables
        
        Parameters
        ----------
        sbd_lut_path : Path
            Path to scaled band depth LUT (.npy)
        anisotropy_lut_path : Path
            Path to anisotropy factor LUT (.npy or .npz)
        albedo_lut_path : Path
            Path to albedo LUT (.npy)
        """
        print("Loading lookup tables...")
        self.sbd_lut = np.load(sbd_lut_path)
        self.anisotropy_lut = np.load(anisotropy_lut_path, mmap_mode='r')
        self.albedo_lut = np.load(albedo_lut_path)
        print("Lookup tables loaded successfully")
        
    def read_atcor_files(self, data_dir: Path, flight_line: str) -> Dict:
        """
        Read ATCOR-4 output files for a flight line
        
        Parameters
        ----------
        data_dir : Path
            Directory containing ATCOR output files
        flight_line : str
            Flight line identifier
            
        Returns
        -------
        dict
            Dictionary containing reflectance, flux, slope, aspect, 
            solar angles, transform, and CRS
        """
        print(f"Reading ATCOR files for flight line: {flight_line}")
        
        reflectance = self._read_envi_file(data_dir / f"{flight_line}_atm.dat")
        global_flux = self._read_envi_file(data_dir / f"{flight_line}_eglo.dat")
        slope = self._read_single_band(data_dir / f"{flight_line}_slp.dat")
        aspect = self._read_single_band(data_dir / f"{flight_line}_asp.dat")
        solar_zenith, solar_azimuth = self._read_solar_zenith(data_dir / f"{flight_line}.inn")
        
        with rasterio.open(str(data_dir / f"{flight_line}_atm.dat")) as src:
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
        filepath = Path(filepath)
        if filepath.suffix.lower() == '.hdr':
            filepath = filepath.with_suffix('.dat')
        
        hdr_file = str(filepath).replace('.dat', '.hdr')
        scale_factor = 1.0
        try:
            with open(hdr_file, 'r') as f:
                for line in f:
                    if 'reflectance scale factor' in line.lower():
                        scale_factor = float(line.split('=')[1].strip())
                        break
        except (FileNotFoundError, OSError):
            pass
        
        with rasterio.open(str(filepath)) as src:
            if window:
                data = src.read(window=window)
            else:
                data = src.read()
            data = data.astype(float) / scale_factor
            data = np.where(data == 0, np.nan, data)
            dask_data = da.from_array(data, chunks=(data.shape[0], 
                                                     self.chunk_size[0], 
                                                     self.chunk_size[1]))
        return dask_data
    
    def _read_single_band(self, filepath: Path, window=None) -> da.Array:
        """Read single band file and return as Dask array"""
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
    
    def _read_solar_zenith(self, inn_file: Path) -> Tuple[float, float]:
        """Extract solar zenith and azimuth angles from ATCOR .inn file"""
        with open(inn_file, 'r') as f:
            content = f.read()
            
        solar_zenith = None
        solar_azimuth = None
        
        for line in content.split('\n'):
            if 'solar zenith' in line.lower() and 'azimuth' in line.lower():
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
    
    def calculate_local_illumination_angle(self, 
                                          solar_zenith: float,
                                          solar_azimuth: float,
                                          slope: da.Array,
                                          aspect: da.Array) -> da.Array:
        """
        Calculate local illumination angle accounting for topography
        
        Parameters
        ----------
        solar_zenith : float
            Solar zenith angle in degrees
        solar_azimuth : float
            Solar azimuth angle in degrees
        slope : da.Array
            Slope array in degrees
        aspect : da.Array
            Aspect array in degrees
            
        Returns
        -------
        da.Array
            Local illumination angle in degrees
        """
        theta_s = np.deg2rad(solar_zenith)
        phi_s = np.deg2rad(solar_azimuth)
        slope_rad = da.deg2rad(slope)
        aspect_rad = da.deg2rad(aspect)
        
        cos_local_illum = (da.cos(theta_s) * da.cos(slope_rad) + 
                          da.sin(theta_s) * da.sin(slope_rad) * 
                          da.cos(phi_s - aspect_rad))
        
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
        
        Parameters
        ----------
        band_depth_map : da.Array
            Band depth values (2D)
        local_illum : da.Array
            Local illumination angles (2D)
        viewing_angle : float
            Viewing zenith angle in degrees
        relative_azimuth : float
            Relative azimuth angle in degrees
            
        Returns
        -------
        da.Array
            Grain size in micrometers
        """
        lookup_func = partial(
            _lookup_grain_size_block,
            sbd_lut=self.sbd_lut,
            grain_radii=self.grain_radii,
            illumination_angles=self.illumination_angles,
            viewing_angles=self.viewing_angles,
            relative_azimuths=self.relative_azimuths,
            viewing_angle=viewing_angle,
            relative_azimuth=relative_azimuth
        )
        
        grain_size = da.map_blocks(
            lookup_func,
            band_depth_map,
            local_illum,
            dtype=float
        )
        
        return grain_size
    
    def calculate_anisotropy_factor(self,
                                   grain_size: da.Array,
                                   local_illum: da.Array,
                                   viewing_angle: float = 0.0,
                                   relative_azimuth: float = 0.0) -> da.Array:
        """
        Calculate spectral anisotropy factor from LUT
        
        Parameters
        ----------
        grain_size : da.Array
            Grain size array (2D)
        local_illum : da.Array
            Local illumination angles (2D)
        viewing_angle : float
            Viewing zenith angle in degrees
        relative_azimuth : float
            Relative azimuth angle in degrees
            
        Returns
        -------
        da.Array
            Anisotropy factor (wavelengths, rows, cols)
        """
        gs_array = grain_size.compute()
        illum_array = local_illum.compute()
        
        if gs_array.shape != illum_array.shape:
            min_rows = min(gs_array.shape[0], illum_array.shape[0])
            min_cols = min(gs_array.shape[1], illum_array.shape[1])
            gs_array = gs_array[:min_rows, :min_cols]
            illum_array = illum_array[:min_rows, :min_cols]
        
        def lookup_pixel(gs, illum):
            if np.isnan(gs) or np.isnan(illum):
                return np.full(len(self.wavelengths), np.nan)
            
            illum_idx = np.argmin(np.abs(self.illumination_angles - illum))
            view_idx = np.argmin(np.abs(self.viewing_angles - viewing_angle))
            azim_idx = np.argmin(np.abs(self.relative_azimuths - relative_azimuth))
            gs_idx = np.argmin(np.abs(self.grain_radii - gs))
            
            return self.anisotropy_lut[illum_idx, view_idx, azim_idx, gs_idx, :]
        
        vec_lookup = np.vectorize(lookup_pixel, signature='(),()->(n)')
        aniso_array = vec_lookup(gs_array, illum_array)
        aniso_array = np.transpose(aniso_array, (2, 0, 1))
        
        anisotropy = da.from_array(aniso_array, chunks=aniso_array.shape)
        
        return anisotropy
    
    def calculate_spectral_albedo(self,
                                 grain_size: da.Array,
                                 reflectance_hdrf: da.Array,
                                 anisotropy_factor: da.Array) -> da.Array:
        """
        Calculate spectral albedo from HDRF reflectance and anisotropy factor
        
        Parameters
        ----------
        grain_size : da.Array
            Grain size array (not directly used but kept for API consistency)
        reflectance_hdrf : da.Array
            HDRF reflectance (wavelengths, rows, cols)
        anisotropy_factor : da.Array
            Anisotropy factor (wavelengths, rows, cols)
            
        Returns
        -------
        da.Array
            Spectral albedo (wavelengths, rows, cols)
        """
        return reflectance_hdrf * anisotropy_factor
    
    def lookup_clean_albedo(self, grain_size: da.Array) -> da.Array:
        """
        Look up clean snow albedo from LUT using measured grain size
        
        Parameters
        ----------
        grain_size : da.Array
            Grain size array (2D)
            
        Returns
        -------
        da.Array
            Spectral albedo for clean snow (wavelengths, rows, cols)
        """
        if isinstance(grain_size, da.Array):
            gs_array = grain_size.compute()
        else:
            gs_array = grain_size
        
        def lookup_pixel(gs):
            if np.isnan(gs):
                return np.full(len(self.wavelengths), np.nan)
            gs_idx = np.argmin(np.abs(self.grain_radii - gs))
            return self.albedo_lut[gs_idx, :]
        
        vec_lookup = np.vectorize(lookup_pixel, signature='()->(n)')
        albedo_array = vec_lookup(gs_array)
        albedo_array = np.transpose(albedo_array, (2, 0, 1))
        
        spectral_albedo_clean = da.from_array(
            albedo_array,
            chunks=(len(self.wavelengths), self.chunk_size[0], self.chunk_size[1])
        )
        
        return spectral_albedo_clean

    def calculate_broadband_albedo(self, spectral_albedo, global_flux):
        """
        Calculate broadband albedo by integrating spectral albedo
        
        Parameters
        ----------
        spectral_albedo : da.Array or np.ndarray
            Spectral albedo (wavelengths, rows, cols)
        global_flux : da.Array or np.ndarray
            Global irradiance flux (wavelengths, rows, cols)
            
        Returns
        -------
        np.ndarray
            Broadband albedo (rows, cols)
        """
        wvl_um = self.wavelengths / 1000.0
        numerator = np.trapezoid(spectral_albedo * global_flux, x=wvl_um, axis=0)
        denominator = np.trapezoid(global_flux, x=wvl_um, axis=0)
        return numerator / denominator
    
    def calculate_radiative_forcing(self,
                                   spectral_albedo_clean: da.Array,
                                   spectral_albedo_actual: da.Array,
                                   solar_spectrum: np.ndarray) -> da.Array:
        """
        Calculate radiative forcing by light absorbing particles
        
        Integration up to 1000nm following Painter et al. (2013)
        
        Parameters
        ----------
        spectral_albedo_clean : da.Array
            Clean snow spectral albedo
        spectral_albedo_actual : da.Array
            Actual measured spectral albedo
        solar_spectrum : np.ndarray
            Solar irradiance spectrum
            
        Returns
        -------
        da.Array
            Radiative forcing (W/m²)
        """
        idx_1000 = np.argmin(np.abs(self.wavelengths - 1000))
        
        albedo_diff = spectral_albedo_clean[:idx_1000+1] - spectral_albedo_actual[:idx_1000+1]
        albedo_diff = da.maximum(albedo_diff, 0)
        
        flux = solar_spectrum[:idx_1000+1].reshape(-1, 1, 1)
        wvl_um = (self.wavelengths[:idx_1000+1] / 1000).reshape(-1, 1, 1)
        
        integrand = albedo_diff * flux
        rf_lap = 0.1 * da.sum(
            (integrand[:-1] + integrand[1:]) / 2 * da.from_array(np.diff(wvl_um, axis=0), chunks=wvl_um.shape),
            axis=0
        )
        
        return rf_lap
    
    def _save_geotiff(self, data: np.ndarray, filepath: Path, transform, crs):
        """
        Save array as GeoTIFF with georeferencing
        
        Parameters
        ----------
        data : np.ndarray
            2D array to save
        filepath : Path
            Output file path
        transform : Affine
            Rasterio transform
        crs : CRS
            Coordinate reference system
        """
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
