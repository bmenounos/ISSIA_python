"""
ISSIA Processing Configuration Example

Copy this file and modify for your specific processing needs
"""

from pathlib import Path

# ==============================================================================
# INPUT DATA CONFIGURATION
# ==============================================================================

# Directory containing your ATCOR-corrected hyperspectral data
DATA_DIR = Path('/Users/menounos/Desktop/issia_local')

# Flight line name (subdirectory name in DATA_DIR)
FLIGHT_LINE = '24_4012_05_2024-06-06_17-54-38-rect_img'

# Output directory for results
OUTPUT_DIR = Path("issia_results")

# ==============================================================================
# PROCESSING PARAMETERS
# ==============================================================================

# NDSI threshold for snow/ice masking
# 0.4 = standard (recommended)
# 0.3 = more lenient (includes more pixels)
# 0.5 = more strict (cleaner snow only)
NDSI_THRESHOLD = 0.4

# Subset processing (None for full image)
# Format: (row_start, row_end, col_start, col_end)
# Examples:
#   SUBSET = None                          # Full image
#   SUBSET = (500, 1500, 300, 1300)       # Specific region
#   SUBSET = (1000, 1100, 500, 600)       # Small test (100×100)
SUBSET = None

# Chunk size for Dask processing
# Larger = faster but more memory
# Smaller = slower but less memory
# Recommended: (1024, 1024) for 8+ GB RAM
#              (512, 512) for 4 GB RAM
CHUNK_SIZE = (1024, 1024)

# ==============================================================================
# LOOKUP TABLES
# ==============================================================================

# Directory containing lookup tables
LUT_DIR = Path("lookup_tables")

# LUT filenames (usually don't need to change these)
SBD_LUT = "sbd_lut.npy"
ANISOTROPY_LUT = "anisotropy_lut.npy"
ALBEDO_LUT = "albedo_lut.npy"

# ==============================================================================
# LUT PARAMETERS (must match your LUT generation)
# ==============================================================================

# Grain radii in micrometers
GRAIN_RADII_START = 30
GRAIN_RADII_END = 5001
GRAIN_RADII_STEP = 30

# Illumination angles in degrees
ILLUM_ANGLES_START = 0
ILLUM_ANGLES_END = 86
ILLUM_ANGLES_STEP = 5

# Viewing angles in degrees
VIEW_ANGLES_START = 0
VIEW_ANGLES_END = 86
VIEW_ANGLES_STEP = 5

# Relative azimuth angles in degrees
REL_AZIMUTH_START = 0
REL_AZIMUTH_END = 361
REL_AZIMUTH_STEP = 10

# Coordinate reference system (EPSG code)
# 32610 = UTM Zone 10N (Western North America)
# Change to match your study area
CRS_CODE = 32610

# ==============================================================================
# WAVELENGTH FILE
# ==============================================================================

# File containing wavelength array
WAVELENGTH_FILE = "wvl.npy"

# ==============================================================================
# PERFORMANCE TUNING
# ==============================================================================

# Number of workers (None = automatic detection, uses 75% of CPU cores)
# Set to specific number to limit: NUM_WORKERS = 4
NUM_WORKERS = None

# Dask cache size
DASK_CACHE_SIZE = '2GB'

# ==============================================================================
# ADVANCED OPTIONS
# ==============================================================================

# Continuum removal wavelength range
# 830-1130 nm is MATLAB standard (recommended)
CONTINUUM_LEFT_WL = 830.0
CONTINUUM_RIGHT_WL = 1130.0

# Viewing geometry (for nadir-looking sensor)
VIEWING_ANGLE = 0.0
RELATIVE_AZIMUTH = 0.0

# Verbose output
VERBOSE = False
