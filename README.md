# ISSIA - Imaging Spectrometer Snow and Ice Algorithm
## Python Implementation with Dask Parallelization

This is a Python translation of the MATLAB ISSIA (Imaging Spectrometer - Snow and Ice Algorithm) code with Dask parallelization for efficient processing of large hyperspectral datasets.

### Overview

ISSIA produces three surface property retrievals on a per-flight-line basis:
- **Broadband Albedo (BBA)**: Integrated solar reflectance across the spectrum
- **Optical Grain Radius**: Effective snow/ice grain size in micrometers
- **Radiative Forcing by Light Absorbing Particles (RF_LAP)**: Additional radiative forcing from impurities

### Methodology

The algorithm is based on the methodology described in:

**Donahue et al. (2023)**, "Bridging the gap between airborne and spaceborne imaging spectroscopy for mountain glacier surface property retrievals", *Remote Sensing of Environment*, 299:113849

Key components:
1. **Continuum Removal**: Applied to the 1030 nm ice absorption feature
2. **Scaled Band Depth**: Used for grain size retrieval from lookup tables
3. **Topographic Correction**: Accounts for local illumination geometry
4. **Anisotropy Correction**: Converts HCRF to HDRF using bidirectional effects
5. **Spectral Albedo**: Calculated from grain size and corrected reflectance
6. **Broadband Integration**: Weighted by solar irradiance spectrum

## Installation

### Requirements

```bash
# Core dependencies
numpy>=1.20.0
dask[array]>=2021.0.0
xarray>=0.19.0
scipy>=1.7.0
rasterio>=1.2.0
matplotlib>=3.3.0

# Optional for enhanced performance
dask[distributed]  # For distributed computing
numba>=0.54.0      # For JIT compilation
```

### Install

```bash
# Clone repository
git clone https://github.com/yourusername/issia-python.git
cd issia-python

# Install dependencies
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate issia
```

## Usage

### 1. Generate Lookup Tables (LUTs)

First, you need to generate the three required lookup tables. This only needs to be done once per instrument configuration.

```python
from lut_generator import ISSIALUTGenerator
import numpy as np
from pathlib import Path

# Define instrument parameters
wavelengths = np.linspace(380, 2500, 451)  # AisaFENIX example
grain_radii = np.logspace(np.log10(30), np.log10(5000), 50)
illumination_angles = np.arange(0, 85, 5)
viewing_angles = np.arange(0, 65, 5)
relative_azimuths = np.arange(0, 185, 15)

# Initialize generator
generator = ISSIALUTGenerator(
    wavelengths=wavelengths,
    grain_radii=grain_radii,
    illumination_angles=illumination_angles,
    viewing_angles=viewing_angles,
    relative_azimuths=relative_azimuths
)

# Generate LUTs
output_dir = Path("lookup_tables")
output_dir.mkdir(exist_ok=True)

sbd_lut = generator.generate_sbd_lut(
    output_path=output_dir / "sbd_lut.npy"
)

albedo_lut = generator.generate_albedo_lut(
    output_path=output_dir / "albedo_lut.npy"
)

# This one is large - may take time!
anisotropy_lut = generator.generate_anisotropy_lut(
    output_path=output_dir / "anisotropy_lut.npy",
    use_dask=True
)
```

**Note**: The LUT generator uses simplified radiative transfer models. For production use, replace the `_simulate_snow_reflectance()` and `_simulate_snow_albedo()` methods with calls to proper radiative transfer models like:
- SNICAR (Snow, Ice, and Aerosol Radiative model)
- TARTES (Two-streAm Radiative TransfEr in Snow)
- DISORT
- Or similar validated snow/ice RT models

### 2. Process Flight Lines

Once LUTs are generated, process your ATCOR-4 output files:

```python
from issia import ISSIAProcessor
from pathlib import Path
import numpy as np

# Initialize processor
processor = ISSIAProcessor(
    wavelengths=np.linspace(380, 2500, 451),
    grain_radii=np.logspace(np.log10(30), np.log10(5000), 50),
    illumination_angles=np.arange(0, 85, 5),
    viewing_angles=np.arange(0, 65, 5),
    relative_azimuths=np.arange(0, 185, 15),
    coord_ref_sys_code=32610,  # WGS84 / UTM Zone 10N
    chunk_size=(512, 512)  # Dask chunk size
)

# Load LUTs
processor.load_lookup_tables(
    sbd_lut_path=Path("lookup_tables/sbd_lut.npy"),
    anisotropy_lut_path=Path("lookup_tables/anisotropy_lut.npy"),
    albedo_lut_path=Path("lookup_tables/albedo_lut.npy")
)

# Process a flight line
output_files = processor.process_flight_line(
    data_dir=Path("atcor_output"),
    flight_line="flight_001",
    output_dir=Path("issia_results"),
    viewing_angle=0.0,
    solar_azimuth=180.0
)

# Output files will be GeoTIFFs:
# - flight_001_grain_size.tif
# - flight_001_broadband_albedo.tif
# - flight_001_radiative_forcing.tif
```

### 3. Batch Processing

Process multiple flight lines efficiently:

```python
from utils import batch_process_directory

results = batch_process_directory(
    processor=processor,
    data_dir=Path("atcor_output"),
    output_dir=Path("issia_results"),
    pattern="*_atm.dat",
    viewing_angle=0.0,
    solar_azimuth=180.0
)
```

### 4. Mosaic Flight Lines

Combine multiple flight lines into a single mosaic:

```python
from utils import mosaic_flight_lines
from pathlib import Path

# Mosaic grain size retrievals
grain_size_files = sorted(Path("issia_results").glob("*_grain_size.tif"))

mosaic = mosaic_flight_lines(
    input_files=grain_size_files,
    output_file=Path("issia_results/grain_size_mosaic.tif"),
    method='mean'  # or 'median', 'first', 'last'
)
```

### 5. Visualization

Create visualizations of results:

```python
from utils import visualize_retrieval
import rasterio

# Load result
with rasterio.open("issia_results/flight_001_grain_size.tif") as src:
    grain_size = src.read(1)

# Visualize
visualize_retrieval(
    data=grain_size,
    title="Optical Grain Size (μm)",
    cmap='YlOrRd',
    vmin=50,
    vmax=500,
    output_file="grain_size_visualization.png"
)
```

## Input Files

ISSIA requires ATCOR-4 output files for each flight line:

1. **{flight_line}.inn**: ATCOR-4 configuration file (contains solar zenith angle)
2. **{flight_line}_atm.dat**: Atmospherically corrected reflectance (ENVI format)
3. **{flight_line}_eglo.dat**: Modeled global solar flux at ground (ENVI format)
4. **{flight_line}_slp.dat**: Terrain slope map in degrees (ENVI format)
5. **{flight_line}_asp.dat**: Terrain aspect map in degrees (ENVI format)

**Important**: All `.dat` files must have the same spatial extent and resolution. Perform spatial subsetting in ATCOR-4 if needed.

## Output Files

For each flight line, ISSIA generates three GeoTIFF files:

1. **{flight_line}_grain_size.tif**: Optical grain radius (micrometers)
2. **{flight_line}_broadband_albedo.tif**: Broadband albedo (0-1)
3. **{flight_line}_radiative_forcing.tif**: Radiative forcing by LAPs (W/m²)

Pixels without valid retrievals are stored as NaN.

## Dask Parallelization

The Python implementation uses Dask for efficient parallel processing:

- **Chunked Processing**: Large datasets are processed in manageable chunks
- **Lazy Evaluation**: Computation only occurs when results are needed
- **Memory Efficiency**: Only required chunks are loaded into memory
- **Scalability**: Can scale from laptops to clusters

Adjust chunk size based on available memory:

```python
# For systems with more RAM
processor = ISSIAProcessor(..., chunk_size=(1024, 1024))

# For memory-constrained systems
processor = ISSIAProcessor(..., chunk_size=(256, 256))
```

### Distributed Computing

For very large datasets, use Dask distributed:

```python
from dask.distributed import Client

# Start local cluster
client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB')

# Process normally - Dask will distribute automatically
processor.process_flight_line(...)

client.close()
```

## Performance Considerations

1. **LUT Size**: The anisotropy factor LUT can be very large (10+ GB). Consider:
   - Reducing angular resolution if geometry is consistent
   - Using memory mapping (`mmap_mode='r'`) for large LUTs
   - Generating only the required angular range

2. **Chunk Size**: Optimal chunk size depends on:
   - Available RAM
   - Number of spectral bands
   - Spatial dimensions
   - Number of workers

3. **Processing Time**: For a typical 1m resolution flight line:
   - Setup: ~1 minute
   - Processing: ~5-20 minutes (depending on chunk size and hardware)
   - Much faster than the original MATLAB (~40 minutes)

## Differences from MATLAB Version

### Improvements:
- **Dask Parallelization**: Faster processing of large datasets
- **Memory Efficiency**: Chunked processing reduces memory requirements
- **Modular Design**: Easier to extend and customize
- **Better I/O**: Support for multiple file formats
- **Visualization Tools**: Built-in plotting capabilities

### Equivalent Functionality:
- Same retrieval algorithms
- Same LUT interpolation methods
- Same topographic correction
- Same output formats (GeoTIFF)

### Not Yet Implemented:
- Automatic ATCOR-4 file detection (requires explicit file naming)
- Built-in LUT download (must generate or provide LUTs)
- Interactive parameter tuning (parameters set in code)

## Troubleshooting

### Common Issues:

**1. Memory Errors**
```python
# Reduce chunk size
processor = ISSIAProcessor(..., chunk_size=(256, 256))
```

**2. Slow LUT Generation**
```python
# Use coarser angular resolution
illumination_angles = np.arange(0, 85, 10)  # 10° instead of 5°
```

**3. Import Errors**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
```

**4. File Not Found**
- Check that file naming matches expected pattern
- Ensure .hdr files exist alongside .dat files
- Verify all required ATCOR files are present

## Citation

If you use this code, please cite:

**Donahue, C.P., Menounos, B., Viner, N., Skiles, S.M., Beffort, S., Denouden, T., Gonzalez Arriola, S., White, R., Heathfield, D., 2023.** Bridging the gap between airborne and spaceborne imaging spectroscopy for mountain glacier surface property retrievals. *Remote Sensing of Environment* 299, 113849. https://doi.org/10.1016/j.rse.2023.113849

Original MATLAB code: https://github.com/donahuechristopher/ISSIA

## License

Apache-2.0 License (same as original MATLAB version)

## Contributing

Contributions welcome! Areas for improvement:
- Integration with validated RT models (SNICAR, TARTES)
- GPU acceleration for LUT generation and processing
- Additional surface property retrievals
- Quality control metrics
- Uncertainty quantification

## Contact

For questions about the Python implementation, open an issue on GitHub.

For questions about the methodology, contact the original author:
Christopher Donahue: donahue.christopher@outlook.com

## Acknowledgments

- Original MATLAB ISSIA by Christopher Donahue (Hakai/UNBC Airborne Coastal Observatory)
- Python translation and Dask integration
- Based on methodology from Donahue et al. (2023)
