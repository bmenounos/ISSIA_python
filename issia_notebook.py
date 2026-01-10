# issia_notebook.py - COMPLETE FIXED VERSION
import numpy as np
import dask.array as da
import rasterio
from pathlib import Path
from issia_core import ISSIAProcessor, _lookup_grain_size_block, _lookup_anisotropy_block

class ISSIAProcessorNotebook(ISSIAProcessor):
    def __init__(self, **kwargs):
        super().__init__(
            wavelengths=kwargs.get('wavelengths', np.zeros(1)),
            grain_radii=kwargs.get('grain_radii', np.zeros(1)),
            illumination_angles=kwargs.get('illumination_angles', np.zeros(1)),
            viewing_angles=kwargs.get('viewing_angles', np.zeros(1)),
            relative_azimuths=kwargs.get('relative_azimuths', np.zeros(1))
        )
        self.verbose = True
        self.chunk_size = kwargs.get('chunk_size', (256, 256))

    def load_lookup_tables(self, sbd_path, aniso_path, alb_path):
        super().load_lookup_tables(sbd_path, aniso_path, alb_path)
        
        # Only set grain_radii_albedo for albedo LUT lookups
        if hasattr(self, 'albedo_lut') and hasattr(self, 'grain_radii'):
            self.grain_radii_albedo = self.grain_radii

    def load_flight_line_data(self, base_dir, flight_line, subset=None):
        data_dir = Path(base_dir)
        window = None
        if subset:
            ymin, ymax, xmin, xmax = subset
            window = rasterio.windows.Window(xmin, ymin, xmax - xmin, ymax - ymin)

        refl = self._read_envi_file(data_dir / f"{flight_line}_atm.dat", window=window)
        flux = self._read_envi_file(data_dir / f"{flight_line}_eglo.dat", window=window)
        slope = self._read_single_band(data_dir / f"{flight_line}_slp.dat", window=window)
        aspect = self._read_single_band(data_dir / f"{flight_line}_asp.dat", window=window)
        zenith, azimuth = self._read_solar_zenith(data_dir / f"{flight_line}.inn")
        
        illum = self.calculate_local_illumination_angle(zenith, azimuth, slope, aspect)
        
        def _calc_bd_block(refl_block, wvl):
            out = np.zeros((refl_block.shape[1], refl_block.shape[2]))
            for i in range(refl_block.shape[1]):
                for j in range(refl_block.shape[2]):
                    res = self.continuum_removal(refl_block[:, i, j], wvl)
                    out[i, j] = res[1] if isinstance(res, tuple) else res
            return out

        bd_map = da.map_blocks(
            _calc_bd_block, refl, wvl=self.wavelengths,
            dtype=float, drop_axis=0, chunks=refl.chunks[1:]
        )

        return {
            'refl_hdrf': refl,
            'global_flux': flux, 
            'illumination': illum, 
            'bd_830_1130': bd_map,
        }

    def read_atcor_files(self, data_dir, flight_line, subset=None):
        """
        Override base class method to add subset parameter support.
        Returns dictionary with keys expected by run_issia_claude.py.
        """
        data_dir = Path(data_dir)
        window = None
        if subset:
            ymin, ymax, xmin, xmax = subset
            window = rasterio.windows.Window(xmin, ymin, xmax - xmin, ymax - ymin)

        # Read data files
        reflectance = self._read_envi_file(data_dir / f"{flight_line}_atm.dat", window=window)
        global_flux = self._read_envi_file(data_dir / f"{flight_line}_eglo.dat", window=window)
        slope = self._read_single_band(data_dir / f"{flight_line}_slp.dat", window=window)
        aspect = self._read_single_band(data_dir / f"{flight_line}_asp.dat", window=window)
        solar_zenith, solar_azimuth = self._read_solar_zenith(data_dir / f"{flight_line}.inn")
        
        # Get transform and CRS from reflectance file
        with rasterio.open(data_dir / f"{flight_line}_atm.dat") as src:
            if window:
                transform = src.window_transform(window)
            else:
                transform = src.transform
            crs = src.crs
        
        return {
            'reflectance': reflectance,
            'global_flux': global_flux,
            'solar_zenith': solar_zenith,
            'solar_azimuth': solar_azimuth,
            'slope': slope,
            'aspect': aspect,
            'transform': transform,
            'crs': crs
        }

    def calculate_radiative_forcing(self, grain_size, spectral_albedo_actual, global_flux):
        """
        Calculate radiative forcing - EXACT MATLAB implementation.
        
        MATLAB:
        image.rf(i,j) = 0.1*(trapz(wvl_specim_um(1:right_idx), 
                             rf_diff .* squeeze(sflux(i,j,1:right_idx))));
        
        Algorithm:
        1. Filter to wavelengths <= 1000 nm
        2. Convert wavelengths to micrometers (wvl / 1000)
        3. For each pixel: look up clean albedo from LUT using grain size
        4. Calculate diff = max(clean_albedo - actual_albedo, 0)
        5. Integrate: 0.1 * trapz(wavelength_um, diff * flux_pixel)
        """
        # Filter to wavelengths <= 1000 nm
        rf_mask = self.wavelengths <= 1000
        wvl_nm = self.wavelengths[rf_mask]  # Keep in nanometers
        actual_sub = spectral_albedo_actual[rf_mask, :, :]
        flux_sub = global_flux[rf_mask, :, :]

        def _rf_block(spec_block, flux_block, gs_2d, wvl_local, lut, radii, mask):
            """Process one chunk - matches MATLAB pixel-by-pixel logic."""
            rows, cols = spec_block.shape[1], spec_block.shape[2]
            result = np.zeros((rows, cols), dtype=np.float32)
            
            for i in range(rows):
                for j in range(cols):
                    gs = gs_2d[i, j]
                    if np.isnan(gs) or gs <= 0:
                        result[i, j] = np.nan
                        continue
                    
                    # Look up clean albedo for this grain size
                    gs_idx = np.argmin(np.abs(radii - gs))
                    albedo_clean = lut[gs_idx, mask]
                    
                    # Calculate albedo difference (clean - actual), clip to >= 0
                    rf_diff = np.maximum(albedo_clean - spec_block[:, i, j], 0)
                    
                    # Integration with wavelengths in nm
                    result[i, j] = np.trapz(rf_diff * flux_block[:, i, j], wvl_local)
            
            return result

        return da.map_blocks(
            _rf_block, actual_sub, flux_sub,
            gs_2d=grain_size.compute(),
            wvl_local=wvl_nm, 
            lut=self.albedo_lut, 
            radii=self.grain_radii_albedo,
            mask=rf_mask,
            dtype=np.float32, 
            drop_axis=0, 
            chunks=actual_sub.chunks[1:]
        )
