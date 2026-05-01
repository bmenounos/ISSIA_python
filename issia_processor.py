
"""
ISSIA Processor - Optimized for parallel processing

Extends ISSIAProcessor with:
- Numba JIT-compiled pixel processing
- Proper Dask multiprocessing scheduler
- Vectorized operations where possible
"""

import numpy as np
import dask.array as da
import dask
import rasterio
from pathlib import Path
from issia_core import ISSIAProcessor

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
    print("Numba available - using JIT compilation for speed")
except ImportError:
    HAS_NUMBA = False
    print("Warning: Numba not installed. Install with 'pip install numba' for 5-10x speedup")


# ============================================================================
# NUMBA-ACCELERATED FUNCTIONS
# ============================================================================

if HAS_NUMBA:
    @jit(nopython=True, parallel=True, cache=True)
    def _process_grain_size_numba(bd_block, illum_block, view_block, raa_block,
                                   sbd_lut, grain_radii, illum_angles, view_angles, azim_angles):
        """Numba-accelerated grain size retrieval"""
        rows, cols = bd_block.shape
        result = np.full((rows, cols), np.nan, dtype=np.float32)
        
        n_illum = len(illum_angles)
        n_view = len(view_angles)
        n_azim = len(azim_angles)
        n_grain = len(grain_radii)
        
        for i in prange(rows):
            for j in range(cols):
                bd = bd_block[i, j]
                illum = illum_block[i, j]
                view = view_block[i, j]
                raa = raa_block[i, j]
                
                if np.isnan(bd) or np.isnan(illum) or np.isnan(view) or np.isnan(raa):
                    continue
                
                # Find nearest LUT indices
                illum_idx = 0
                min_diff = abs(illum_angles[0] - illum)
                for k in range(1, n_illum):
                    diff = abs(illum_angles[k] - illum)
                    if diff < min_diff:
                        min_diff = diff
                        illum_idx = k
                
                view_idx = 0
                min_diff = abs(view_angles[0] - view)
                for k in range(1, n_view):
                    diff = abs(view_angles[k] - view)
                    if diff < min_diff:
                        min_diff = diff
                        view_idx = k
                
                raa_idx = 0
                min_diff = abs(azim_angles[0] - raa)
                for k in range(1, n_azim):
                    diff = abs(azim_angles[k] - raa)
                    if diff < min_diff:
                        min_diff = diff
                        raa_idx = k
                
                # Get band depth curve and interpolate
                bd_curve = sbd_lut[illum_idx, view_idx, raa_idx, :]
                
                # Linear interpolation to find grain size
                if bd <= bd_curve[0]:
                    result[i, j] = grain_radii[0]
                elif bd >= bd_curve[n_grain - 1]:
                    result[i, j] = grain_radii[n_grain - 1]
                else:
                    for k in range(n_grain - 1):
                        if bd_curve[k] <= bd <= bd_curve[k + 1]:
                            t = (bd - bd_curve[k]) / (bd_curve[k + 1] - bd_curve[k] + 1e-10)
                            result[i, j] = grain_radii[k] + t * (grain_radii[k + 1] - grain_radii[k])
                            break
        
        return result

    @jit(nopython=True, parallel=True, cache=True)
    def _process_anisotropy_numba(gs_block, illum_block, view_block, raa_block,
                                   aniso_lut, grain_radii, illum_angles, view_angles, azim_angles):
        """Numba-accelerated anisotropy calculation"""
        rows, cols = gs_block.shape
        n_wvl = aniso_lut.shape[-1]
        result = np.full((n_wvl, rows, cols), np.nan, dtype=np.float32)
        
        n_illum = len(illum_angles)
        n_view = len(view_angles)
        n_azim = len(azim_angles)
        n_grain = len(grain_radii)
        
        for i in prange(rows):
            for j in range(cols):
                gs = gs_block[i, j]
                illum = illum_block[i, j]
                view = view_block[i, j]
                raa = raa_block[i, j]
                
                if np.isnan(gs) or np.isnan(illum) or np.isnan(view) or np.isnan(raa):
                    continue
                
                # Find nearest LUT indices
                illum_idx = 0
                min_diff = abs(illum_angles[0] - illum)
                for k in range(1, n_illum):
                    diff = abs(illum_angles[k] - illum)
                    if diff < min_diff:
                        min_diff = diff
                        illum_idx = k
                
                view_idx = 0
                min_diff = abs(view_angles[0] - view)
                for k in range(1, n_view):
                    diff = abs(view_angles[k] - view)
                    if diff < min_diff:
                        min_diff = diff
                        view_idx = k
                
                raa_idx = 0
                min_diff = abs(azim_angles[0] - raa)
                for k in range(1, n_azim):
                    diff = abs(azim_angles[k] - raa)
                    if diff < min_diff:
                        min_diff = diff
                        raa_idx = k
                
                gs_idx = 0
                min_diff = abs(grain_radii[0] - gs)
                for k in range(1, n_grain):
                    diff = abs(grain_radii[k] - gs)
                    if diff < min_diff:
                        min_diff = diff
                        gs_idx = k
                
                # Copy anisotropy spectrum
                for w in range(n_wvl):
                    result[w, i, j] = aniso_lut[illum_idx, view_idx, raa_idx, gs_idx, w]
        
        return result

    @jit(nopython=True, parallel=True, cache=True)
    def _process_rf_numba(spec_block, flux_block, gs_block, wvl_um, albedo_lut, grain_radii, rf_mask):
        """Numba-accelerated radiative forcing calculation"""
        n_wvl, rows, cols = spec_block.shape
        result = np.full((rows, cols), np.nan, dtype=np.float32)
        n_grain = len(grain_radii)
        
        for i in prange(rows):
            for j in range(cols):
                gs = gs_block[i, j]
                
                if np.isnan(gs) or gs <= 0:
                    continue
                
                # Find grain size index
                gs_idx = 0
                min_diff = abs(grain_radii[0] - gs)
                for k in range(1, n_grain):
                    diff = abs(grain_radii[k] - gs)
                    if diff < min_diff:
                        min_diff = diff
                        gs_idx = k
                
                # Calculate RF integral
                integral = 0.0
                for w in range(n_wvl - 1):
                    if rf_mask[w] and rf_mask[w + 1]:
                        clean_w = albedo_lut[gs_idx, w]
                        clean_w1 = albedo_lut[gs_idx, w + 1]
                        
                        diff_w = max(0.0, clean_w - spec_block[w, i, j])
                        diff_w1 = max(0.0, clean_w1 - spec_block[w + 1, i, j])
                        
                        ## convert nm to micron
                        flux_w = flux_block[w, i, j] * 1000 
                        flux_w1 = flux_block[w + 1, i, j] * 1000
                        
                        # Trapezoidal integration
                        integral += 0.5 * (diff_w * flux_w + diff_w1 * flux_w1) * (wvl_um[w + 1] - wvl_um[w])
                
                result[i, j] = integral
        
        return result


# ============================================================================
# FALLBACK NUMPY IMPLEMENTATIONS (if Numba not available)
# ============================================================================

def _process_grain_size_numpy(bd_block, illum_block, view_block, raa_block,
                               sbd_lut, grain_radii, illum_angles, view_angles, azim_angles):
    """Vectorized numpy grain size retrieval (slower than Numba but faster than loops)"""
    # Vectorized index finding
    illum_idx = np.argmin(np.abs(illum_angles[:, None, None] - illum_block[None, :, :]), axis=0)
    view_idx = np.argmin(np.abs(view_angles[:, None, None] - view_block[None, :, :]), axis=0)
    raa_idx = np.argmin(np.abs(azim_angles[:, None, None] - raa_block[None, :, :]), axis=0)
    
    valid_mask = ~(np.isnan(bd_block) | np.isnan(illum_block) | 
                   np.isnan(view_block) | np.isnan(raa_block))
    
    result = np.full_like(bd_block, np.nan, dtype=np.float32)
    
    # Process valid pixels - still need loop for interpolation but much faster with vectorized indexing
    valid_idx = np.where(valid_mask)
    for idx in range(len(valid_idx[0])):
        i, j = valid_idx[0][idx], valid_idx[1][idx]
        bd_curve = sbd_lut[illum_idx[i, j], view_idx[i, j], raa_idx[i, j], :]
        result[i, j] = np.interp(bd_block[i, j], bd_curve, grain_radii)
    
    return result


def _process_anisotropy_numpy(gs_block, illum_block, view_block, raa_block,
                               aniso_lut, grain_radii, illum_angles, view_angles, azim_angles):
    """Vectorized numpy anisotropy calculation"""
    n_wvl = aniso_lut.shape[-1]
    
    illum_idx = np.argmin(np.abs(illum_angles[:, None, None] - illum_block[None, :, :]), axis=0)
    view_idx = np.argmin(np.abs(view_angles[:, None, None] - view_block[None, :, :]), axis=0)
    raa_idx = np.argmin(np.abs(azim_angles[:, None, None] - raa_block[None, :, :]), axis=0)
    gs_idx = np.argmin(np.abs(grain_radii[:, None, None] - gs_block[None, :, :]), axis=0)
    
    valid_mask = ~(np.isnan(gs_block) | np.isnan(illum_block) | 
                   np.isnan(view_block) | np.isnan(raa_block))
    
    result = np.full((n_wvl, gs_block.shape[0], gs_block.shape[1]), np.nan, dtype=np.float32)
    
    # Vectorized lookup for valid pixels
    valid_idx = np.where(valid_mask)
    for idx in range(len(valid_idx[0])):
        i, j = valid_idx[0][idx], valid_idx[1][idx]
        result[:, i, j] = aniso_lut[illum_idx[i, j], view_idx[i, j], raa_idx[i, j], gs_idx[i, j], :]
    
    return result


# ============================================================================
# OPTIMIZED PROCESSOR CLASS
# ============================================================================

class ISSIAProcessorOptimized(ISSIAProcessor):
    """
    Optimized ISSIA processor with parallel processing support
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            wavelengths=kwargs.get('wavelengths', np.zeros(1)),
            grain_radii=kwargs.get('grain_radii', np.zeros(1)),
            illumination_angles=kwargs.get('illumination_angles', np.zeros(1)),
            viewing_angles=kwargs.get('viewing_angles', np.zeros(1)),
            relative_azimuths=kwargs.get('relative_azimuths', np.zeros(1))
        )
        self.chunk_size = kwargs.get('chunk_size', (512, 512))
        
        # Configure Dask for multiprocessing
        n_workers = kwargs.get('n_workers', None)
        self._configure_dask(n_workers)
    
    def _configure_dask(self, n_workers=None):
        """Configure Dask for optimal parallel processing"""
        import os
        if n_workers is None:
            n_workers = os.cpu_count()
        
        # Use processes scheduler for CPU-bound work
        dask.config.set(scheduler='synchronous')  # For map_blocks with Numba
        print(f"Dask configured with synchronous scheduler (Numba handles parallelism)")

    def load_lookup_tables(self, sbd_path, aniso_path, alb_path):
        """Load lookup tables with optimized memory layout"""
        super().load_lookup_tables(sbd_path, aniso_path, alb_path)
        
        # Handle .npz format for anisotropy
        if str(aniso_path).endswith('.npz'):
            with np.load(aniso_path) as data:
                arr = data['data']
                # Convert dtype first (may be a no-op if already float32), then
                # ensure contiguous — avoids holding two 6.8 GB copies simultaneously
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                self.anisotropy_lut = np.ascontiguousarray(arr)
        else:
            # .npy path: convert in-place before making contiguous to cap peak RAM
            if self.anisotropy_lut.dtype != np.float32:
                self.anisotropy_lut = self.anisotropy_lut.astype(np.float32)
            self.anisotropy_lut = np.ascontiguousarray(self.anisotropy_lut)

        # Ensure all LUTs are contiguous float32
        self.sbd_lut = np.ascontiguousarray(self.sbd_lut).astype(np.float32)
        self.albedo_lut = np.ascontiguousarray(self.albedo_lut).astype(np.float32)
        
        # Pre-convert angle arrays to float32
        self._illum_angles_f32 = self.illumination_angles.astype(np.float32)
        self._view_angles_f32 = self.viewing_angles.astype(np.float32)
        self._azim_angles_f32 = self.relative_azimuths.astype(np.float32)
        self._grain_radii_f32 = self.grain_radii.astype(np.float32)
        
        if hasattr(self, 'grain_radii'):
            self.grain_radii_albedo = self.grain_radii

    def calculate_local_viewing_geometry(self, solar_zenith, solar_azimuth, slope, aspect, 
                                         viewing_zenith=0.0, viewing_azimuth=0.0):
        """Calculate local viewing geometry - fully vectorized"""
        # Convert to numpy if dask
        if isinstance(slope, da.Array):
            slope = slope.compute()
        if isinstance(aspect, da.Array):
            aspect = aspect.compute()
        
        theta_i = np.deg2rad(solar_zenith)
        phi_i = np.deg2rad(solar_azimuth)
        theta_v = np.deg2rad(viewing_zenith)
        phi_v = np.deg2rad(viewing_azimuth)
        slope_rad = np.deg2rad(slope)
        aspect_rad = np.deg2rad(aspect)
        
        mu_i = (np.cos(theta_i) * np.cos(slope_rad) + 
                np.sin(theta_i) * np.sin(slope_rad) * np.cos(phi_i - aspect_rad))
        mu_i = np.where(mu_i < 0.000001, np.nan, mu_i)
        
        mu_v = (np.cos(theta_v) * np.cos(slope_rad) + 
                np.sin(theta_v) * np.sin(slope_rad) * np.cos(phi_v - aspect_rad))
        
        theta_i_eff = np.arccos(np.clip(mu_i, -1, 1))
        theta_v_eff = np.arccos(np.clip(mu_v, -1, 1))
        theta_v_eff = np.where(theta_v_eff > np.deg2rad(90), np.nan, theta_v_eff)
        
        mu_az_numerator = (np.cos(theta_v) * np.cos(theta_i) + 
                          np.sin(theta_v) * np.sin(theta_i) * np.cos(phi_v - phi_i) - 
                          mu_i * mu_v)
        mu_az_denominator = np.sin(theta_i_eff) * np.sin(theta_v_eff)
        
        mu_az = np.where(mu_az_denominator == 0, 0, 
                        mu_az_numerator / mu_az_denominator)
        mu_az = np.clip(mu_az, -1, 1)
        raa_eff = np.arccos(mu_az)
        
        return np.rad2deg(theta_i_eff), np.rad2deg(theta_v_eff), np.rad2deg(raa_eff)

    def retrieve_grain_size_pixelwise(self, bd_map, illumination_angle, viewing_angle_map, relative_azimuth_map):
        """Pixel-wise grain size retrieval - Numba accelerated"""
        # Convert to numpy arrays
        bd = bd_map.compute() if isinstance(bd_map, da.Array) else bd_map
        illum = illumination_angle.compute() if isinstance(illumination_angle, da.Array) else illumination_angle
        view = viewing_angle_map.compute() if isinstance(viewing_angle_map, da.Array) else viewing_angle_map
        raa = relative_azimuth_map.compute() if isinstance(relative_azimuth_map, da.Array) else relative_azimuth_map
        
        # Ensure float32 for Numba
        bd = bd.astype(np.float32)
        illum = illum.astype(np.float32)
        view = view.astype(np.float32)
        raa = raa.astype(np.float32)
        
        if HAS_NUMBA:
            result = _process_grain_size_numba(
                bd, illum, view, raa,
                self.sbd_lut, self._grain_radii_f32,
                self._illum_angles_f32, self._view_angles_f32, self._azim_angles_f32
            )
        else:
            result = _process_grain_size_numpy(
                bd, illum, view, raa,
                self.sbd_lut, self._grain_radii_f32,
                self._illum_angles_f32, self._view_angles_f32, self._azim_angles_f32
            )
        
        return da.from_array(result, chunks=self.chunk_size)
    
    def calculate_anisotropy_factor_pixelwise(self, grain_size, illumination_angle, viewing_angle_map, relative_azimuth_map):
        """Pixel-wise anisotropy calculation - Numba accelerated"""
        gs = grain_size.compute() if isinstance(grain_size, da.Array) else grain_size
        illum = illumination_angle.compute() if isinstance(illumination_angle, da.Array) else illumination_angle
        view = viewing_angle_map.compute() if isinstance(viewing_angle_map, da.Array) else viewing_angle_map
        raa = relative_azimuth_map.compute() if isinstance(relative_azimuth_map, da.Array) else relative_azimuth_map
        
        gs = gs.astype(np.float32)
        illum = illum.astype(np.float32)
        view = view.astype(np.float32)
        raa = raa.astype(np.float32)
        
        if HAS_NUMBA:
            result = _process_anisotropy_numba(
                gs, illum, view, raa,
                self.anisotropy_lut, self._grain_radii_f32,
                self._illum_angles_f32, self._view_angles_f32, self._azim_angles_f32
            )
        else:
            result = _process_anisotropy_numpy(
                gs, illum, view, raa,
                self.anisotropy_lut, self._grain_radii_f32,
                self._illum_angles_f32, self._view_angles_f32, self._azim_angles_f32
            )
        
        n_wvl = self.anisotropy_lut.shape[-1]
        return da.from_array(result, chunks=(n_wvl, *self.chunk_size))

    def read_atcor_files(self, data_dir, flight_line, subset=None):
        """Read ATCOR-4 output files with optional spatial subset"""
        data_dir = Path(data_dir)
        window = None
        if subset:
            ymin, ymax, xmin, xmax = subset
            window = rasterio.windows.Window(xmin, ymin, xmax - xmin, ymax - ymin)

        reflectance = self._read_envi_file(data_dir / f"{flight_line}_atm.dat", window=window)
        global_flux = self._read_envi_file(data_dir / f"{flight_line}_eglo.dat", window=window)
        slope = self._read_single_band(data_dir / f"{flight_line}_slp.dat", window=window)
        aspect = self._read_single_band(data_dir / f"{flight_line}_asp.dat", window=window)
        solar_zenith, solar_azimuth = self._read_solar_zenith(data_dir / f"{flight_line}.inn")
        
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
        """Calculate radiative forcing - Numba accelerated"""
        rf_mask = self.wavelengths <= 1000
        wvl_um = (self.wavelengths / 1000.0).astype(np.float32)
        
        gs = grain_size.compute() if isinstance(grain_size, da.Array) else grain_size
        spec = spectral_albedo_actual.compute() if isinstance(spectral_albedo_actual, da.Array) else spectral_albedo_actual
        flux = global_flux.compute() if isinstance(global_flux, da.Array) else global_flux
        
        gs = gs.astype(np.float32)
        spec = spec.astype(np.float32)
        flux = flux.astype(np.float32)
        
        if HAS_NUMBA:
            result = _process_rf_numba(
                spec, flux, gs, wvl_um,
                self.albedo_lut, self._grain_radii_f32, rf_mask
            )
        else:
            # Fallback numpy implementation
            rows, cols = gs.shape
            result = np.full((rows, cols), np.nan, dtype=np.float32)
            
            for i in range(rows):
                for j in range(cols):
                    if np.isnan(gs[i, j]) or gs[i, j] <= 0:
                        continue
                    gs_idx = np.argmin(np.abs(self._grain_radii_f32 - gs[i, j]))
                    albedo_clean = self.albedo_lut[gs_idx, rf_mask]
                    rf_diff = np.maximum(albedo_clean - spec[rf_mask, i, j], 0)
                    ## convert flux from nm to micron
                    result[i, j] = np.trapezoid(rf_diff * (flux[rf_mask, i, j] * 1000.0), wvl_um[rf_mask])
        
        return da.from_array(result, chunks=self.chunk_size)

    def compute_albedo_rf_chunked(self, refl_masked, anisotropy, global_flux, grain_size,
                                   chunk_cols=256):
        """Compute broadband albedo and radiative forcing in column chunks.

        Avoids materializing the full spectral albedo array (~13 GB) by computing
        both products from the same spatial chunk, then discarding it.

        Returns
        -------
        broadband_albedo : np.ndarray (rows, cols)
        rf              : np.ndarray (rows, cols)
        """
        from tqdm import tqdm

        wvl_um = (self.wavelengths / 1000.0).astype(np.float32)
        rf_mask = (self.wavelengths <= 1000)

        refl = refl_masked.astype(np.float32)
        aniso = anisotropy.astype(np.float32)
        flux = (global_flux.compute() if isinstance(global_flux, da.Array) else global_flux).astype(np.float32)
        gs = (grain_size.compute() if isinstance(grain_size, da.Array) else grain_size).astype(np.float32)

        rows, cols = refl.shape[1], refl.shape[2]
        broadband = np.full((rows, cols), np.nan, dtype=np.float32)
        rf = np.full((rows, cols), np.nan, dtype=np.float32)

        # NaN→0 on flux so that absorption-window zero bands don't kill den_full.
        # Matches MATLAB which keeps zeros rather than NaN-masking them.
        flux_int = np.where(np.isnan(flux), 0.0, flux)
        den_full = np.trapezoid(flux_int, x=wvl_um, axis=0)  # (rows, cols) — flux denominator

        with tqdm(total=cols, desc="    albedo+RF", unit="col", leave=False) as pbar:
            for c0 in range(0, cols, chunk_cols):
                c1 = min(c0 + chunk_cols, cols)

                sa = refl[:, :, c0:c1] * aniso[:, :, c0:c1]
                fx = flux_int[:, :, c0:c1]

                # Replace NaN with 0 before integration — matches MATLAB which keeps
                # zeros in the trapz rather than propagating NaN from SWIR absorption
                # windows through to the broadband integral.
                sa_int = np.where(np.isnan(sa), 0.0, sa)

                # Broadband albedo — only valid where grain size was retrieved
                num = np.trapezoid(sa_int * fx, x=wvl_um, axis=0)
                den = den_full[:, c0:c1]
                gs_slice = gs[:, c0:c1]
                with np.errstate(divide='ignore', invalid='ignore'):
                    broadband[:, c0:c1] = np.where(
                        np.isfinite(gs_slice) & (den > 0),
                        (num / den).astype(np.float32),
                        np.nan
                    )

                # Radiative forcing
                if HAS_NUMBA:
                    rf[:, c0:c1] = _process_rf_numba(
                        sa, fx, gs[:, c0:c1], wvl_um,
                        self.albedo_lut, self._grain_radii_f32, rf_mask
                    )
                else:
                    nc = c1 - c0
                    for i in range(rows):
                        for j in range(nc):
                            g = gs[i, c0 + j]
                            if np.isnan(g) or g <= 0:
                                continue
                            gs_idx = np.argmin(np.abs(self._grain_radii_f32 - g))
                            clean = self.albedo_lut[gs_idx, rf_mask]
                            diff = np.maximum(0.0, clean - sa[rf_mask, i, j])
                            rf[i, c0 + j] = np.trapezoid(diff * fx[rf_mask, i, j] * 1000.0,
                                                      wvl_um[rf_mask])

                pbar.update(c1 - c0)

        return broadband, rf


# Alias for backward compatibility
ISSIAProcessorNotebook = ISSIAProcessorOptimized
