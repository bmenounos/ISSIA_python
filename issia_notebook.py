"""
ISSIA Processor - Jupyter Notebook Optimized Version

This version includes:
- Better progress tracking for notebooks
- tqdm progress bars
- Diagnostic output at each step
- Memory usage reporting
- Estimated time remaining

Use this in Jupyter notebooks for better interactivity.
"""

import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
import xarray as xr
import rasterio
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings
import time
import sys

# Import the base processor
from issia import ISSIAProcessor


class ISSIAProcessorNotebook(ISSIAProcessor):
    """
    ISSIA Processor optimized for Jupyter notebooks with progress tracking
    """
    
    def __init__(self, *args, verbose=True, **kwargs):
        """
        Initialize with verbosity option
        
        Parameters:
        -----------
        verbose : bool
            Print detailed progress information
        """
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        
    def _print(self, message, level='INFO'):
        """Print message if verbose"""
        if self.verbose:
            timestamp = time.strftime('%H:%M:%S')
            print(f"[{timestamp}] {level}: {message}")
            sys.stdout.flush()
    
    def read_atcor_files(self, data_dir: Path, flight_line: str, subset: tuple = None) -> Dict:
        """
        Read ATCOR-4 output files with progress tracking
        
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
        self._print(f"Reading ATCOR files for flight line: {flight_line}")
        start_time = time.time()
        
        # Check which files exist
        self._print("Checking for required files...")
        required_files = {
            'inn': f"{flight_line}.inn",
            'atm': f"{flight_line}_atm.dat",
            'eglo': f"{flight_line}_eglo.dat",
            'slp': f"{flight_line}_slp.dat",
            'asp': f"{flight_line}_asp.dat"
        }
        
        missing_files = []
        for key, filename in required_files.items():
            filepath = data_dir / filename
            if not filepath.exists():
                missing_files.append(filename)
            else:
                # Get file size
                size_mb = filepath.stat().st_size / (1024**2)
                self._print(f"  ✓ Found {filename} ({size_mb:.1f} MB)")
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
        
        # Load files with simple progress tracking
        file_tasks = [
            ('Reflectance', f"{flight_line}_atm.dat", 'envi'),
            ('Global Flux', f"{flight_line}_eglo.dat", 'envi'),
            ('Slope', f"{flight_line}_slp.dat", 'single'),
            ('Aspect', f"{flight_line}_asp.dat", 'single'),
            ('Solar Zenith', f"{flight_line}.inn", 'inn'),
        ]
        
        results = {}
        
        # Setup window if subset specified
        window = None
        if subset:
            r0, r1, c0, c1 = subset
            from rasterio.windows import Window
            window = Window(c0, r0, c1 - c0, r1 - r0)
            print(f"Reading subset window: rows {r0}-{r1}, cols {c0}-{c1}\n")
        
        print(f"\nLoading {len(file_tasks)} files...")
        for i, (name, filename, ftype) in enumerate(file_tasks, 1):
            file_start = time.time()
            print(f"  [{i}/{len(file_tasks)}] Loading {name}...", end='', flush=True)
            
            filepath = data_dir / filename
            
            if ftype == 'envi':
                data = self._read_envi_file(filepath, window=window)
                results[name.lower().replace(' ', '_')] = data
            elif ftype == 'single':
                data = self._read_single_band(filepath, window=window)
                results[name.lower()] = data
            elif ftype == 'inn':
                solar_zenith, solar_azimuth = self._read_solar_zenith(filepath)
                results['solar_zenith'] = solar_zenith
                results['solar_azimuth'] = solar_azimuth
            
            file_time = time.time() - file_start
            print(f" ✓ ({file_time:.1f}s)")

        
        # Get geotransform and projection from .dat file
        atm_file = data_dir / f"{flight_line}_atm.dat"
        with rasterio.open(str(atm_file)) as src:
            transform = src.transform
            crs = src.crs
            
            # Adjust transform if subset was used
            if subset:
                r0, r1, c0, c1 = subset
                from rasterio.transform import Affine
                transform = transform * Affine.translation(c0, r0)
        
        total_time = time.time() - start_time
        print(f"\n✓ All files loaded in {total_time:.1f}s")
        
        return {
            'reflectance': results['reflectance'],
            'global_flux': results['global_flux'],
            'slope': results['slope'],
            'aspect': results['aspect'],
            'solar_zenith': results['solar_zenith'],
            'solar_azimuth': results['solar_azimuth'],
            'transform': transform,
            'crs': crs
        }
    
    def process_flight_line(self,
                          data_dir: Path,
                          flight_line: str,
                          output_dir: Path,
                          viewing_angle: float = 0.0,
                          solar_azimuth: float = None,
                          subset: tuple = None) -> Dict[str, Path]:
        """
        Process a single flight line with detailed progress tracking
        
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
        solar_azimuth : float, optional
            Solar azimuth angle (degrees). If None, reads from .inn file
        subset : tuple, optional
            Spatial subset (row_start, row_end, col_start, col_end) for testing
            Example: (0, 500, 0, 500) processes only first 500x500 pixels
            
        Returns:
        --------
        output_files : dict
            Dictionary of output file paths
        """
        overall_start = time.time()
        
        print("\n" + "="*70)
        print(f"PROCESSING FLIGHT LINE: {flight_line}")
        if subset:
            print(f"SUBSET MODE: rows {subset[0]}-{subset[1]}, cols {subset[2]}-{subset[3]}")
        print("="*70 + "\n")
        
        # Read input files (with subset if specified)
        data = self.read_atcor_files(data_dir, flight_line, subset=subset)
        
        # Extract arrays and metadata
        reflectance = data['reflectance']
        global_flux = data['global_flux']
        slope = data['slope']
        aspect = data['aspect']
        solar_zenith = data['solar_zenith']
        transform = data['transform']
        crs = data['crs']
        
        # Use solar azimuth from file if not provided
        if solar_azimuth is None:
            solar_azimuth = data['solar_azimuth']
            self._print(f"Using solar azimuth from .inn file: {solar_azimuth:.2f}°")
        else:
            self._print(f"Using provided solar azimuth: {solar_azimuth:.2f}°")
        
        n_bands, n_rows, n_cols = reflectance.shape
        n_pixels = n_rows * n_cols
        
        print(f"\nDATA SUMMARY:")
        print(f"  Dimensions: {n_rows} rows × {n_cols} cols × {n_bands} bands")
        print(f"  Total pixels: {n_pixels:,}")
        print(f"  Chunk size: {self.chunk_size}")
        print(f"  Solar zenith: {solar_zenith:.2f}°")
        print(f"  Solar azimuth: {solar_azimuth:.2f}°")
        print()
        
        # Step 1: Calculate local illumination angle
        step_start = time.time()
        self._print("Step 1/6: Calculating local illumination angles...")
        local_illum = self.calculate_local_illumination_angle(
            solar_zenith, solar_azimuth, slope, aspect
        )
        step_time = time.time() - step_start
        self._print(f"  ✓ Completed in {step_time:.1f}s", level='SUCCESS')
        
        # Step 2: Calculate scaled band depth for each pixel
        step_start = time.time()
        self._print("Step 2/6: Calculating scaled band depths...")
        self._print("  (This extracts the 1030 nm ice absorption feature)")
        
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
        step_time = time.time() - step_start
        self._print(f"  ✓ Band depth array created in {step_time:.1f}s", level='SUCCESS')
        
        # Step 3: Retrieve grain size
        step_start = time.time()
        self._print("Step 3/6: Retrieving grain sizes from lookup tables...")
        grain_size = self.retrieve_grain_size(band_depths, local_illum, 
                                             viewing_angle, 0.0)
        step_time = time.time() - step_start
        self._print(f"  ✓ Grain size retrieval configured in {step_time:.1f}s", level='SUCCESS')
        
        # Step 4: Calculate anisotropy factor
        step_start = time.time()
        self._print("Step 4/6: Calculating anisotropy factors...")
        self._print("  (Converting HCRF to HDRF)")
        anisotropy = self.calculate_anisotropy_factor(
            grain_size, local_illum, viewing_angle, 0.0
        )
        step_time = time.time() - step_start
        self._print(f"  ✓ Anisotropy calculation configured in {step_time:.1f}s", level='SUCCESS')
        
        # Convert HCRF to HDRF
        reflectance_hdrf = reflectance * anisotropy
        
        # Step 5: Calculate spectral albedo
        step_start = time.time()
        self._print("Step 5/6: Calculating spectral albedo...")
        spectral_albedo = self.calculate_spectral_albedo(
            grain_size, reflectance_hdrf, anisotropy
        )
        step_time = time.time() - step_start
        self._print(f"  ✓ Spectral albedo configured in {step_time:.1f}s", level='SUCCESS')
        
        # Calculate broadband albedo
        self._print("  Calculating broadband albedo...")
        mean_flux = da.nanmean(global_flux, axis=(1, 2)).compute()
        broadband_albedo = self.calculate_broadband_albedo(
            spectral_albedo, mean_flux
        )
        
        # Step 6: Calculate radiative forcing
        step_start = time.time()
        self._print("Step 6/6: Calculating radiative forcing...")
        clean_albedo = self.calculate_spectral_albedo(
            grain_size, reflectance_hdrf * 0 + anisotropy, anisotropy
        )
        rf_lap = self.calculate_radiative_forcing(
            clean_albedo, spectral_albedo, mean_flux
        )
        step_time = time.time() - step_start
        self._print(f"  ✓ Radiative forcing configured in {step_time:.1f}s", level='SUCCESS')
        
        # Compute results with simple progress
        print("\n" + "="*70)
        print("COMPUTING RESULTS (this may take several minutes)...")
        print("="*70 + "\n")
        
        compute_tasks = [
            ('grain size', grain_size),
            ('broadband albedo', broadband_albedo),
            ('radiative forcing', rf_lap)
        ]
        
        results_computed = {}
        
        for i, (name, data) in enumerate(compute_tasks, 1):
            self._print(f"[{i}/3] Computing {name}...")
            compute_start = time.time()
            with ProgressBar():
                result = data.compute()
            compute_time = time.time() - compute_start
            self._print(f"  ✓ {name.title()} computed in {compute_time:.1f}s")
            results_computed[name.replace(' ', '_')] = result
        
        grain_size_result = results_computed['grain_size']
        broadband_albedo_result = results_computed['broadband_albedo']
        rf_lap_result = results_computed['radiative_forcing']
        
        # DEBUG: Check intermediate values
        print("\n" + "="*70)
        print("DEBUG DIAGNOSTICS:")
        print("="*70)
        
        # Check reflectance
        refl_sample = reflectance[:, :100, :100].compute()
        print(f"Reflectance NaN%: {np.isnan(refl_sample).mean()*100:.1f}%")
        print(f"Reflectance range: {np.nanmin(refl_sample):.4f} - {np.nanmax(refl_sample):.4f}")
        
        # Check anisotropy
        aniso_sample = anisotropy[:, :100, :100].compute()
        print(f"Anisotropy NaN%: {np.isnan(aniso_sample).mean()*100:.1f}%")
        print(f"Anisotropy range: {np.nanmin(aniso_sample):.4f} - {np.nanmax(aniso_sample):.4f}")
        print(f"Anisotropy zeros: {(aniso_sample == 0).sum()}")
        
        # Check spectral albedo
        spec_alb_sample = spectral_albedo[:, :100, :100].compute()
        print(f"Spectral albedo NaN%: {np.isnan(spec_alb_sample).mean()*100:.1f}%")
        print(f"Spectral albedo range: {np.nanmin(spec_alb_sample):.4f} - {np.nanmax(spec_alb_sample):.4f}")
        
        # Check solar flux
        flux_sample = global_flux[:, :100, :100].compute()
        print(f"Solar flux NaN%: {np.isnan(flux_sample).mean()*100:.1f}%")
        print(f"Solar flux range: {np.nanmin(flux_sample):.4f} - {np.nanmax(flux_sample):.4f}")
        mean_flux = da.nanmean(global_flux, axis=(1, 2)).compute()
        print(f"Mean flux per band - min: {np.nanmin(mean_flux):.4f}, max: {np.nanmax(mean_flux):.4f}")
        print(f"Mean flux NaN count: {np.isnan(mean_flux).sum()}/{len(mean_flux)}")
        
        # Check broadband albedo before final compute
        bba_sample = broadband_albedo[:100, :100].compute()
        print(f"Broadband albedo (before result) NaN%: {np.isnan(bba_sample).mean()*100:.1f}%")
        print(f"Broadband albedo range: {np.nanmin(bba_sample):.4f} - {np.nanmax(bba_sample):.4f}")
        
        print("="*70 + "\n")
        
        # Quick statistics
        print("\nRETRIEVAL STATISTICS:")
        print(f"  Grain Size: {np.nanmean(grain_size_result):.1f} ± {np.nanstd(grain_size_result):.1f} μm")
        print(f"              Range: {np.nanmin(grain_size_result):.1f} - {np.nanmax(grain_size_result):.1f} μm")
        print(f"  Broadband Albedo: {np.nanmean(broadband_albedo_result):.3f} ± {np.nanstd(broadband_albedo_result):.3f}")
        print(f"                    Range: {np.nanmin(broadband_albedo_result):.3f} - {np.nanmax(broadband_albedo_result):.3f}")
        print(f"  Radiative Forcing: {np.nanmean(rf_lap_result):.1f} ± {np.nanstd(rf_lap_result):.1f} W/m²")
        print(f"                     Range: {np.nanmin(rf_lap_result):.1f} - {np.nanmax(rf_lap_result):.1f} W/m²")
        print()
        
        # Save outputs as GeoTIFFs
        print("SAVING OUTPUTS...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        # Save grain size
        grain_size_file = output_dir / f"{flight_line}_grain_size.tif"
        self._save_geotiff(grain_size_result, grain_size_file, 
                          transform, crs)
        output_files['grain_size'] = grain_size_file
        
        # Save broadband albedo
        albedo_file = output_dir / f"{flight_line}_broadband_albedo.tif"
        self._save_geotiff(broadband_albedo_result, albedo_file,
                          transform, crs)
        output_files['broadband_albedo'] = albedo_file
        
        # Save radiative forcing
        rf_file = output_dir / f"{flight_line}_radiative_forcing.tif"
        self._save_geotiff(rf_lap_result, rf_file,
                          transform, crs)
        output_files['radiative_forcing'] = rf_file
        
        total_time = time.time() - overall_start
        
        print("\n" + "="*70)
        print(f"✓ PROCESSING COMPLETE in {total_time/60:.1f} minutes")
        print("="*70)
        print(f"\nOutputs saved to: {output_dir}")
        for key, filepath in output_files.items():
            print(f"  • {key}: {filepath.name}")
        print()
        
        return output_files


def process_with_monitoring(processor, data_dir, flight_line, output_dir, **kwargs):
    """
    Helper function to process with memory monitoring
    
    Parameters:
    -----------
    processor : ISSIAProcessorNotebook
        Initialized processor
    data_dir : Path
        Data directory
    flight_line : str
        Flight line name
    output_dir : Path
        Output directory
    **kwargs : dict
        Additional arguments for process_flight_line
        
    Returns:
    --------
    output_files : dict
        Output file paths
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    print(f"Initial memory usage: {process.memory_info().rss / 1024**3:.2f} GB\n")
    
    output_files = processor.process_flight_line(
        data_dir, flight_line, output_dir, **kwargs
    )
    
    print(f"\nFinal memory usage: {process.memory_info().rss / 1024**3:.2f} GB")
    
    return output_files


# Example notebook usage
if __name__ == "__main__":
    """
    Example for Jupyter notebook
    """
    from pathlib import Path
    import numpy as np
    
    print("ISSIA Notebook Processor - Example Usage\n")
    print("=" * 70)
    
    # Initialize with verbosity
    processor = ISSIAProcessorNotebook(
        wavelengths=np.linspace(380, 2500, 451),
        grain_radii=np.logspace(np.log10(30), np.log10(5000), 50),
        illumination_angles=np.arange(0, 85, 5),
        viewing_angles=np.arange(0, 65, 5),
        relative_azimuths=np.arange(0, 185, 15),
        coord_ref_sys_code=32610,
        chunk_size=(512, 512),
        verbose=True  # Enable progress messages
    )
    
    print("\nProcessor initialized with verbose mode ON")
    print("All processing steps will show detailed progress")
    print("\nReady to load lookup tables and process data!")
