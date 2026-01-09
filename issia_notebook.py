"""
ISSIA Processor - Jupyter Notebook Version with Lazy I/O

This version includes:
1. Multi-core support (fixed pickling) ✓
2. Lazy chunked I/O (fast network reading) ✓
3. Progress tracking for notebooks ✓

Key fix: Using da.from_delayed for proper lazy chunk reading
"""

import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
import dask
import rasterio
from rasterio.windows import Window
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings
import time
import sys

# Import the FIXED base processor
from issia_core import ISSIAProcessor


class ISSIAProcessorNotebook(ISSIAProcessor):
    """
    ISSIA Processor optimized for Jupyter notebooks
    WITH lazy chunked I/O for fast network reading
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
    
    def _read_envi_file(self, filepath: Path, window=None) -> da.Array:
        """
        Read ENVI file with WORKING lazy chunked I/O
        
        Key difference: For subsets, reads directly. For full files, creates
        lazy Dask array using da.from_delayed that reads chunks on-demand.
        """
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
            pass
        
        if window is not None:
            # SUBSET: Read immediately (small enough)
            with rasterio.open(str(filepath)) as src:
                data = src.read(window=window)
                data = data.astype(float) / scale_factor
                data = np.where(data == 0, np.nan, data)
                dask_data = da.from_array(data, chunks=(data.shape[0], 
                                                         self.chunk_size[0], 
                                                         self.chunk_size[1]))
            return dask_data
        
        else:
            # FULL FILE: Use lazy reading with da.from_delayed
            # This is the KEY optimization - only reads metadata, not data!
            with rasterio.open(str(filepath)) as src:
                n_bands = src.count
                n_rows = src.height
                n_cols = src.width
            
            shape = (n_bands, n_rows, n_cols)
            chunks = (n_bands, self.chunk_size[0], self.chunk_size[1])
            
            # Create delayed functions for each chunk
            @dask.delayed
            def read_chunk(filepath, row_start, row_end, col_start, col_end, scale_factor):
                """Read a single chunk from file"""
                with rasterio.open(str(filepath)) as src:
                    win = Window(col_start, row_start, 
                               col_end - col_start, 
                               row_end - row_start)
                    data = src.read(window=win)
                
                # Apply scale and mask
                data = data.astype(float) / scale_factor
                data = np.where(data == 0, np.nan, data)
                return data
            
            # Build array of delayed chunks
            delayed_chunks = []
            for row_start in range(0, n_rows, self.chunk_size[0]):
                row_end = min(row_start + self.chunk_size[0], n_rows)
                row_chunks = []
                
                for col_start in range(0, n_cols, self.chunk_size[1]):
                    col_end = min(col_start + self.chunk_size[1], n_cols)
                    
                    # Create delayed chunk
                    chunk = read_chunk(filepath, row_start, row_end, 
                                     col_start, col_end, scale_factor)
                    
                    # Convert to dask array
                    chunk_shape = (n_bands, row_end - row_start, col_end - col_start)
                    chunk_array = da.from_delayed(chunk, shape=chunk_shape, dtype=float)
                    row_chunks.append(chunk_array)
                
                delayed_chunks.append(row_chunks)
            
            # Concatenate all chunks
            rows = [da.concatenate(row_chunks, axis=2) for row_chunks in delayed_chunks]
            dask_data = da.concatenate(rows, axis=1)
            
            return dask_data
    
    def _read_single_band(self, filepath: Path, window=None) -> da.Array:
        """
        Read single band with lazy I/O
        """
        filepath = Path(filepath)
        if filepath.suffix.lower() == '.hdr':
            filepath = filepath.with_suffix('.dat')
        
        if window is not None:
            # Subset - read immediately
            with rasterio.open(str(filepath)) as src:
                data = src.read(1, window=window)
                dask_data = da.from_array(data, chunks=self.chunk_size)
            return dask_data
        
        else:
            # Full file - lazy reading
            with rasterio.open(str(filepath)) as src:
                n_rows = src.height
                n_cols = src.width
            
            shape = (n_rows, n_cols)
            
            @dask.delayed
            def read_chunk(filepath, row_start, row_end, col_start, col_end):
                """Read chunk on-demand"""
                with rasterio.open(str(filepath)) as src:
                    win = Window(col_start, row_start,
                               col_end - col_start,
                               row_end - row_start)
                    data = src.read(1, window=win)
                return data.astype(float)
            
            # Build array of delayed chunks
            delayed_chunks = []
            for row_start in range(0, n_rows, self.chunk_size[0]):
                row_end = min(row_start + self.chunk_size[0], n_rows)
                row_chunks = []
                
                for col_start in range(0, n_cols, self.chunk_size[1]):
                    col_end = min(col_start + self.chunk_size[1], n_cols)
                    
                    chunk = read_chunk(filepath, row_start, row_end, col_start, col_end)
                    chunk_shape = (row_end - row_start, col_end - col_start)
                    chunk_array = da.from_delayed(chunk, shape=chunk_shape, dtype=float)
                    row_chunks.append(chunk_array)
                
                delayed_chunks.append(row_chunks)
            
            # Concatenate
            rows = [da.concatenate(row_chunks, axis=1) for row_chunks in delayed_chunks]
            dask_data = da.concatenate(rows, axis=0)
            
            return dask_data
    
    def read_atcor_files(self, data_dir: Path, flight_line: str, subset: tuple = None) -> Dict:
        """
        Read ATCOR-4 output files with progress tracking
        NOW WITH WORKING LAZY I/O!
        """
        self._print(f"Reading ATCOR files for flight line: {flight_line}")
        start_time = time.time()
        
        # Check files
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
                size_mb = filepath.stat().st_size / (1024**2)
                self._print(f"  ✓ Found {filename} ({size_mb:.1f} MB)")
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
        
        # Setup window
        window = None
        if subset:
            r0, r1, c0, c1 = subset
            window = Window(c0, r0, c1 - c0, r1 - r0)
            print(f"\nReading subset window: rows {r0}-{r1}, cols {c0}-{c1}")
            print("(Will read immediately - small data)\n")
        else:
            print(f"\n⚡ LAZY I/O MODE: Creating delayed readers")
            print("(Chunks will be read on-demand during processing)\n")
        
        print(f"Loading {len(required_files)} files...")
        
        results = {}
        
        # Load files with timing
        file_start = time.time()
        print(f"  [1/5] Loading Reflectance...", end='', flush=True)
        atm_file = data_dir / f"{flight_line}_atm.dat"
        results['reflectance'] = self._read_envi_file(atm_file, window=window)
        print(f" ✓ ({time.time() - file_start:.1f}s)")
        
        file_start = time.time()
        print(f"  [2/5] Loading Global Flux...", end='', flush=True)
        eglo_file = data_dir / f"{flight_line}_eglo.dat"
        results['global_flux'] = self._read_envi_file(eglo_file, window=window)
        print(f" ✓ ({time.time() - file_start:.1f}s)")
        
        file_start = time.time()
        print(f"  [3/5] Loading Slope...", end='', flush=True)
        slp_file = data_dir / f"{flight_line}_slp.dat"
        results['slope'] = self._read_single_band(slp_file, window=window)
        print(f" ✓ ({time.time() - file_start:.1f}s)")
        
        file_start = time.time()
        print(f"  [4/5] Loading Aspect...", end='', flush=True)
        asp_file = data_dir / f"{flight_line}_asp.dat"
        results['aspect'] = self._read_single_band(asp_file, window=window)
        print(f" ✓ ({time.time() - file_start:.1f}s)")
        
        file_start = time.time()
        print(f"  [5/5] Loading Solar Angles...", end='', flush=True)
        inn_file = data_dir / f"{flight_line}.inn"
        solar_zenith, solar_azimuth = self._read_solar_zenith(inn_file)
        results['solar_zenith'] = solar_zenith
        results['solar_azimuth'] = solar_azimuth
        print(f" ✓ ({time.time() - file_start:.1f}s)")
        
        # Get geotransform
        with rasterio.open(str(atm_file)) as src:
            transform = src.transform
            crs = src.crs
            
            if subset:
                r0, r1, c0, c1 = subset
                from rasterio.transform import Affine
                transform = transform * Affine.translation(c0, r0)
        
        total_time = time.time() - start_time
        print(f"\n✓ All files loaded in {total_time:.1f}s")
        
        if not subset and total_time < 2:
            print(f"⚡ FAST! Lazy I/O working - data will be read during compute")
        
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
        """
        overall_start = time.time()
        
        print("\n" + "="*70)
        print(f"PROCESSING FLIGHT LINE: {flight_line}")
        if subset:
            print(f"SUBSET MODE: rows {subset[0]}-{subset[1]}, cols {subset[2]}-{subset[3]}")
        else:
            print(f"FULL IMAGE MODE with lazy I/O")
        print("="*70 + "\n")
        
        # Read input files
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
        
        n_bands, n_rows, n_cols = reflectance.shape
        n_pixels = n_rows * n_cols
        
        print(f"\nDATA SUMMARY:")
        print(f"  Dimensions: {n_rows} rows × {n_cols} cols × {n_bands} bands")
        print(f"  Total pixels: {n_pixels:,}")
        print(f"  Chunk size: {self.chunk_size}")
        print(f"  Solar zenith: {solar_zenith:.2f}°")
        print(f"  Solar azimuth: {solar_azimuth:.2f}°")
        print()
        
        # Processing steps
        step_start = time.time()
        self._print("Step 1/6: Calculating local illumination angles...")
        local_illum = self.calculate_local_illumination_angle(
            solar_zenith, solar_azimuth, slope, aspect
        )
        self._print(f"  ✓ Completed in {time.time() - step_start:.1f}s", level='SUCCESS')
        
        # NDSI mask
        step_start = time.time()
        self._print("Applying NDSI snow/ice mask...")
        snow_mask = self.calculate_ndsi(reflectance)
        snow_fraction = (snow_mask.sum() / snow_mask.size).compute()
        self._print(f"  Snow/ice fraction: {100*snow_fraction:.1f}%")
        reflectance = da.where(snow_mask, reflectance, np.nan)
        self._print(f"  ✓ NDSI masking completed in {time.time() - step_start:.1f}s", level='SUCCESS')
        
        # Band depths
        step_start = time.time()
        self._print("Step 2/6: Calculating scaled band depths...")
        self._print("  (Extracting 1030 nm ice absorption feature)")
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
        self._print(f"  ✓ Band depth array created in {time.time() - step_start:.1f}s", level='SUCCESS')
        
        # Grain size
        step_start = time.time()
        self._print("Step 3/6: Retrieving grain sizes from lookup tables...")
        grain_size = self.retrieve_grain_size(band_depths, local_illum, viewing_angle, 0.0)
        self._print(f"  ✓ Grain size retrieval configured in {time.time() - step_start:.1f}s", level='SUCCESS')
        
        # Anisotropy
        step_start = time.time()
        self._print("Step 4/6: Calculating anisotropy factors...")
        self._print("  (Converting HCRF to HDRF)")
        anisotropy = self.calculate_anisotropy_factor(grain_size, local_illum, viewing_angle, 0.0)
        reflectance_hdrf = reflectance * anisotropy
        self._print(f"  ✓ Anisotropy calculation configured in {time.time() - step_start:.1f}s", level='SUCCESS')
        
        # Spectral albedo
        step_start = time.time()
        self._print("Step 5/6: Calculating spectral albedo...")
        spectral_albedo = self.calculate_spectral_albedo(grain_size, reflectance_hdrf, anisotropy)
        
        self._print("  Calculating broadband albedo...")
        mean_flux = da.nanmean(global_flux, axis=(1, 2)).compute()
        broadband_albedo = self.calculate_broadband_albedo(spectral_albedo, mean_flux)
        self._print(f"  ✓ Spectral albedo configured in {time.time() - step_start:.1f}s", level='SUCCESS')
        
        # Radiative forcing
        step_start = time.time()
        self._print("Step 6/6: Calculating radiative forcing...")
        clean_albedo = self.calculate_spectral_albedo(
            grain_size, reflectance_hdrf * 0 + anisotropy, anisotropy
        )
        rf_lap = self.calculate_radiative_forcing(clean_albedo, spectral_albedo, mean_flux)
        self._print(f"  ✓ Radiative forcing configured in {time.time() - step_start:.1f}s", level='SUCCESS')
        
        # Compute results
        print("\n" + "="*70)
        print("COMPUTING RESULTS")
        if not subset:
            print("(Now reading chunks from network as needed...)")
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
        
        print("="*70 + "\n")
        
        # Statistics
        print("\nRETRIEVAL STATISTICS:")
        print(f"  Grain Size: {np.nanmean(grain_size_result):.1f} ± {np.nanstd(grain_size_result):.1f} μm")
        print(f"              Range: {np.nanmin(grain_size_result):.1f} - {np.nanmax(grain_size_result):.1f} μm")
        print(f"  Broadband Albedo: {np.nanmean(broadband_albedo_result):.3f} ± {np.nanstd(broadband_albedo_result):.3f}")
        print(f"                    Range: {np.nanmin(broadband_albedo_result):.3f} - {np.nanmax(broadband_albedo_result):.3f}")
        print(f"  Radiative Forcing: {np.nanmean(rf_lap_result):.1f} ± {np.nanstd(rf_lap_result):.1f} W/m²")
        print(f"                     Range: {np.nanmin(rf_lap_result):.1f} - {np.nanmax(rf_lap_result):.1f} W/m²")
        print()
        
        # Save outputs
        print("SAVING OUTPUTS...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        grain_size_file = output_dir / f"{flight_line}_grain_size.tif"
        self._save_geotiff(grain_size_result, grain_size_file, transform, crs)
        output_files['grain_size'] = grain_size_file
        
        albedo_file = output_dir / f"{flight_line}_broadband_albedo.tif"
        self._save_geotiff(broadband_albedo_result, albedo_file, transform, crs)
        output_files['broadband_albedo'] = albedo_file
        
        rf_file = output_dir / f"{flight_line}_radiative_forcing.tif"
        self._save_geotiff(rf_lap_result, rf_file, transform, crs)
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


if __name__ == "__main__":
    print("ISSIA Notebook Processor with Lazy I/O")
    print("✓ Multi-core support")
    print("✓ Lazy chunked I/O using da.from_delayed")
    print("✓ Progress tracking")
    print("\nReady for fast network processing!")
