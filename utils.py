"""
Utility functions for ISSIA processing

Includes file I/O, visualization, and helper functions
"""

import numpy as np
import dask.array as da
import rasterio
from rasterio.envi import open as open_envi
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import xarray as xr


def read_envi_header(hdr_file: Path) -> Dict:
    """
    Read ENVI header file and extract metadata
    
    Parameters:
    -----------
    hdr_file : Path
        Path to .hdr file
        
    Returns:
    --------
    metadata : dict
        Dictionary of header parameters
    """
    metadata = {}
    
    with open(hdr_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            
            # Remove braces
            if value.startswith('{'):
                value = value.strip('{}')
            
            # Try to convert to number
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Remove quotes
                value = value.strip('"\'')
            
            metadata[key] = value
    
    return metadata


def read_envi_bsq(dat_file: Path, hdr_file: Optional[Path] = None) -> Tuple[np.ndarray, Dict]:
    """
    Read ENVI BSQ format file
    
    Parameters:
    -----------
    dat_file : Path
        Path to .dat file
    hdr_file : Path, optional
        Path to .hdr file (auto-detected if not provided)
        
    Returns:
    --------
    data : np.ndarray
        Data array (bands, rows, cols) for BSQ
    metadata : dict
        Metadata from header
    """
    if hdr_file is None:
        hdr_file = dat_file.with_suffix('.hdr')
    
    # Read header
    metadata = read_envi_header(hdr_file)
    
    # Extract dimensions
    samples = metadata.get('samples', metadata.get('columns'))
    lines = metadata.get('lines', metadata.get('rows'))
    bands = metadata.get('bands', 1)
    
    # Read binary data
    dtype_map = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        12: np.uint16,
        13: np.uint32
    }
    
    data_type = metadata.get('data_type', 4)
    dtype = dtype_map.get(data_type, np.float32)
    
    # Read based on interleave
    interleave = metadata.get('interleave', 'bsq').lower()
    
    data = np.fromfile(dat_file, dtype=dtype)
    
    if interleave == 'bsq':
        data = data.reshape(bands, lines, samples)
    elif interleave == 'bil':
        data = data.reshape(lines, bands, samples).transpose(1, 0, 2)
    elif interleave == 'bip':
        data = data.reshape(lines, samples, bands).transpose(2, 0, 1)
    else:
        raise ValueError(f"Unknown interleave: {interleave}")
    
    return data, metadata


def write_envi_bsq(data: np.ndarray, 
                   output_file: Path,
                   wavelengths: Optional[np.ndarray] = None,
                   metadata: Optional[Dict] = None):
    """
    Write data in ENVI BSQ format
    
    Parameters:
    -----------
    data : np.ndarray
        Data array (bands, rows, cols)
    output_file : Path
        Output .dat file path
    wavelengths : np.ndarray, optional
        Wavelength array
    metadata : dict, optional
        Additional metadata
    """
    # Write binary data
    data.astype(np.float32).tofile(output_file)
    
    # Write header
    hdr_file = output_file.with_suffix('.hdr')
    
    with open(hdr_file, 'w') as f:
        f.write("ENVI\n")
        f.write(f"samples = {data.shape[2]}\n")
        f.write(f"lines = {data.shape[1]}\n")
        f.write(f"bands = {data.shape[0]}\n")
        f.write("header offset = 0\n")
        f.write("file type = ENVI Standard\n")
        f.write("data type = 4\n")  # float32
        f.write("interleave = bsq\n")
        f.write("byte order = 0\n")
        
        if wavelengths is not None:
            wl_str = ', '.join([f"{w:.6f}" for w in wavelengths])
            f.write(f"wavelength = {{\n{wl_str}\n}}\n")
        
        if metadata:
            for key, value in metadata.items():
                if key not in ['samples', 'lines', 'bands', 'data_type', 
                              'interleave', 'byte_order', 'wavelength']:
                    f.write(f"{key} = {value}\n")


def mosaic_flight_lines(input_files: List[Path],
                       output_file: Path,
                       method: str = 'mean') -> np.ndarray:
    """
    Mosaic multiple flight lines into a single output
    
    Parameters:
    -----------
    input_files : list of Path
        List of GeoTIFF files to mosaic
    output_file : Path
        Output mosaic file
    method : str
        Mosaicking method: 'mean', 'median', 'first', 'last'
        
    Returns:
    --------
    mosaic : np.ndarray
        Mosaicked array
    """
    from rasterio.merge import merge
    
    # Open all files
    src_files = [rasterio.open(f) for f in input_files]
    
    # Merge with specified method
    mosaic, out_transform = merge(src_files, method=method)
    
    # Get metadata from first file
    meta = src_files[0].meta.copy()
    meta.update({
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': out_transform
    })
    
    # Write output
    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(mosaic)
    
    # Close all files
    for src in src_files:
        src.close()
    
    return mosaic


def visualize_retrieval(data: np.ndarray,
                       title: str,
                       cmap: str = 'viridis',
                       vmin: Optional[float] = None,
                       vmax: Optional[float] = None,
                       output_file: Optional[Path] = None):
    """
    Create visualization of retrieval result
    
    Parameters:
    -----------
    data : np.ndarray
        2D array to visualize
    title : str
        Plot title
    cmap : str
        Colormap name
    vmin, vmax : float, optional
        Color scale limits
    output_file : Path, optional
        Save plot to file
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Mask NaN values
    masked_data = np.ma.masked_invalid(data)
    
    im = ax.imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def create_rgb_composite(reflectance: np.ndarray,
                        wavelengths: np.ndarray,
                        output_file: Optional[Path] = None) -> np.ndarray:
    """
    Create RGB composite from hyperspectral data
    
    Parameters:
    -----------
    reflectance : np.ndarray
        Reflectance data (bands, rows, cols)
    wavelengths : np.ndarray
        Wavelength array (nm)
    output_file : Path, optional
        Save RGB to file
        
    Returns:
    --------
    rgb : np.ndarray
        RGB composite (rows, cols, 3)
    """
    # Find closest bands to RGB wavelengths
    red_idx = np.argmin(np.abs(wavelengths - 650))
    green_idx = np.argmin(np.abs(wavelengths - 550))
    blue_idx = np.argmin(np.abs(wavelengths - 450))
    
    # Extract bands
    red = reflectance[red_idx, :, :]
    green = reflectance[green_idx, :, :]
    blue = reflectance[blue_idx, :, :]
    
    # Stack and normalize
    rgb = np.stack([red, green, blue], axis=-1)
    
    # Normalize to 0-1
    for i in range(3):
        band = rgb[:, :, i]
        p2, p98 = np.nanpercentile(band, [2, 98])
        rgb[:, :, i] = np.clip((band - p2) / (p98 - p2 + 1e-10), 0, 1)
    
    if output_file:
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.axis('off')
        plt.title('RGB Composite', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved RGB composite to: {output_file}")
    
    return rgb


def calculate_statistics(data: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate statistics for a retrieval
    
    Parameters:
    -----------
    data : np.ndarray
        2D array
    mask : np.ndarray, optional
        Boolean mask (True = valid)
        
    Returns:
    --------
    stats : dict
        Dictionary of statistics
    """
    if mask is not None:
        data = data[mask]
    else:
        data = data[~np.isnan(data)]
    
    if len(data) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'p5': np.nan,
            'p95': np.nan,
            'count': 0
        }
    
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'p5': np.percentile(data, 5),
        'p95': np.percentile(data, 95),
        'count': len(data)
    }
    
    return stats


def save_results_netcdf(outputs: Dict[str, np.ndarray],
                       output_file: Path,
                       wavelengths: Optional[np.ndarray] = None,
                       transform: Optional = None,
                       crs: Optional = None,
                       metadata: Optional[Dict] = None):
    """
    Save results to NetCDF format for easy analysis
    
    Parameters:
    -----------
    outputs : dict
        Dictionary of output arrays
    output_file : Path
        Output NetCDF file
    wavelengths : np.ndarray, optional
        Wavelength array
    transform : affine transform, optional
        Geospatial transform
    crs : CRS, optional
        Coordinate reference system
    metadata : dict, optional
        Additional metadata
    """
    # Create dataset
    data_vars = {}
    
    for key, array in outputs.items():
        if array.ndim == 2:
            data_vars[key] = (['y', 'x'], array)
        elif array.ndim == 3:
            data_vars[key] = (['band', 'y', 'x'], array)
    
    # Create coordinates
    coords = {}
    
    if transform is not None:
        # Create x, y coordinates from transform
        rows, cols = outputs[list(outputs.keys())[0]].shape[-2:]
        x = transform[2] + np.arange(cols) * transform[0]
        y = transform[5] + np.arange(rows) * transform[4]
        coords['x'] = x
        coords['y'] = y
    
    if wavelengths is not None:
        coords['band'] = wavelengths
    
    # Create dataset
    ds = xr.Dataset(data_vars, coords=coords)
    
    # Add attributes
    if metadata:
        ds.attrs.update(metadata)
    
    if crs:
        ds.attrs['crs'] = str(crs)
    
    # Save to NetCDF
    ds.to_netcdf(output_file)
    print(f"Saved results to NetCDF: {output_file}")


def batch_process_directory(processor,
                            data_dir: Path,
                            output_dir: Path,
                            pattern: str = "*_atm.dat",
                            **kwargs) -> List[Dict]:
    """
    Batch process all flight lines in a directory
    
    Parameters:
    -----------
    processor : ISSIAProcessor
        Initialized ISSIA processor
    data_dir : Path
        Directory containing ATCOR files
    output_dir : Path
        Output directory
    pattern : str
        File pattern to match flight lines
    **kwargs : dict
        Additional arguments for process_flight_line
        
    Returns:
    --------
    results : list of dict
        List of output file dictionaries
    """
    # Find all flight lines
    atm_files = sorted(data_dir.glob(pattern))
    
    # Extract flight line identifiers
    flight_lines = []
    for f in atm_files:
        # Remove _atm.dat suffix
        flight_line = f.stem.replace('_atm', '')
        flight_lines.append(flight_line)
    
    print(f"Found {len(flight_lines)} flight lines to process")
    
    results = []
    
    for i, flight_line in enumerate(flight_lines, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(flight_lines)}: {flight_line}")
        print(f"{'='*60}")
        
        try:
            output_files = processor.process_flight_line(
                data_dir=data_dir,
                flight_line=flight_line,
                output_dir=output_dir,
                **kwargs
            )
            results.append(output_files)
        except Exception as e:
            print(f"ERROR processing {flight_line}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Successfully processed: {len(results)}/{len(flight_lines)} flight lines")
    print(f"{'='*60}")
    
    return results


def main():
    """
    Example usage of utility functions
    """
    from pathlib import Path
    
    # Example: Read ENVI file
    dat_file = Path("example_atm.dat")
    if dat_file.exists():
        data, metadata = read_envi_bsq(dat_file)
        print(f"Data shape: {data.shape}")
        print(f"Metadata: {metadata}")
    
    # Example: Create visualization
    if False:  # Set to True to run example
        example_data = np.random.randn(100, 100) * 50 + 300
        visualize_retrieval(
            example_data,
            title="Example Grain Size (μm)",
            cmap='YlOrRd',
            vmin=50,
            vmax=500,
            output_file=Path("example_grain_size.png")
        )


if __name__ == "__main__":
    main()
