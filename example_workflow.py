"""
Example Workflow for ISSIA Processing

This script demonstrates a complete workflow from LUT generation to
flight line processing and visualization.
"""

import numpy as np
from pathlib import Path
from issia import ISSIAProcessor
from lut_generator import ISSIALUTGenerator
from utils import (batch_process_directory, mosaic_flight_lines, 
                   visualize_retrieval, create_rgb_composite,
                   calculate_statistics)
import rasterio


def setup_directories():
    """Create necessary directories"""
    dirs = ['lookup_tables', 'atcor_output', 'issia_results', 'visualizations']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    return {d: Path(d) for d in dirs}


def generate_luts(output_dir: Path, 
                  wavelengths: np.ndarray,
                  quick_mode: bool = False):
    """
    Generate all required lookup tables
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save LUTs
    wavelengths : np.ndarray
        Wavelength array for your instrument
    quick_mode : bool
        Use coarser resolution for faster generation (testing only)
    """
    print("\n" + "="*70)
    print("STEP 1: GENERATING LOOKUP TABLES")
    print("="*70)
    
    # Define LUT parameters
    if quick_mode:
        print("WARNING: Quick mode enabled - using coarse resolution")
        print("These LUTs are for TESTING ONLY - not for production use!")
        grain_radii = np.logspace(np.log10(30), np.log10(5000), 20)
        illumination_angles = np.arange(0, 85, 10)
        viewing_angles = np.arange(0, 65, 10)
        relative_azimuths = np.arange(0, 185, 30)
    else:
        grain_radii = np.logspace(np.log10(30), np.log10(5000), 50)
        illumination_angles = np.arange(0, 85, 5)
        viewing_angles = np.arange(0, 65, 5)
        relative_azimuths = np.arange(0, 185, 15)
    
    print(f"\nLUT Parameters:")
    print(f"  Wavelengths: {len(wavelengths)}")
    print(f"  Grain sizes: {len(grain_radii)} ({grain_radii[0]:.1f} - {grain_radii[-1]:.1f} μm)")
    print(f"  Illumination angles: {len(illumination_angles)} (0-80°)")
    print(f"  Viewing angles: {len(viewing_angles)} (0-60°)")
    print(f"  Relative azimuths: {len(relative_azimuths)} (0-180°)")
    
    # Initialize generator
    generator = ISSIALUTGenerator(
        wavelengths=wavelengths,
        grain_radii=grain_radii,
        illumination_angles=illumination_angles,
        viewing_angles=viewing_angles,
        relative_azimuths=relative_azimuths
    )
    
    # Generate LUTs
    print("\n1.1 Generating Scaled Band Depth LUT...")
    sbd_lut = generator.generate_sbd_lut(
        output_path=output_dir / "sbd_lut.npy"
    )
    
    print("\n1.2 Generating White-Sky Albedo LUT...")
    albedo_lut = generator.generate_albedo_lut(
        output_path=output_dir / "albedo_lut.npy"
    )
    
    print("\n1.3 Generating Anisotropy Factor LUT...")
    print("WARNING: This may take significant time and memory!")
    anisotropy_lut = generator.generate_anisotropy_lut(
        output_path=output_dir / "anisotropy_lut.npy",
        use_dask=True
    )
    
    print("\n✓ LUT generation complete!")
    return grain_radii, illumination_angles, viewing_angles, relative_azimuths


def initialize_processor(wavelengths: np.ndarray,
                        grain_radii: np.ndarray,
                        illumination_angles: np.ndarray,
                        viewing_angles: np.ndarray,
                        relative_azimuths: np.ndarray,
                        lut_dir: Path,
                        coord_ref_sys: int = 32610) -> ISSIAProcessor:
    """
    Initialize and configure ISSIA processor
    
    Parameters:
    -----------
    wavelengths : np.ndarray
        Wavelength array
    grain_radii : np.ndarray
        Grain radii array from LUT generation
    illumination_angles : np.ndarray
        Illumination angles from LUT generation
    viewing_angles : np.ndarray
        Viewing angles from LUT generation
    relative_azimuths : np.ndarray
        Relative azimuths from LUT generation
    lut_dir : Path
        Directory containing LUTs
    coord_ref_sys : int
        EPSG code for CRS
        
    Returns:
    --------
    processor : ISSIAProcessor
        Configured processor
    """
    print("\n" + "="*70)
    print("STEP 2: INITIALIZING PROCESSOR")
    print("="*70)
    
    processor = ISSIAProcessor(
        wavelengths=wavelengths,
        grain_radii=grain_radii,
        illumination_angles=illumination_angles,
        viewing_angles=viewing_angles,
        relative_azimuths=relative_azimuths,
        coord_ref_sys_code=coord_ref_sys,
        chunk_size=(512, 512)
    )
    
    print("\nLoading lookup tables...")
    processor.load_lookup_tables(
        sbd_lut_path=lut_dir / "sbd_lut.npy",
        anisotropy_lut_path=lut_dir / "anisotropy_lut.npy",
        albedo_lut_path=lut_dir / "albedo_lut.npy"
    )
    
    print("✓ Processor initialized and ready!")
    return processor


def process_data(processor: ISSIAProcessor,
                data_dir: Path,
                output_dir: Path):
    """
    Process all flight lines in directory
    
    Parameters:
    -----------
    processor : ISSIAProcessor
        Configured processor
    data_dir : Path
        Directory with ATCOR-4 files
    output_dir : Path
        Output directory
        
    Returns:
    --------
    results : list
        List of output file dictionaries
    """
    print("\n" + "="*70)
    print("STEP 3: PROCESSING FLIGHT LINES")
    print("="*70)
    
    results = batch_process_directory(
        processor=processor,
        data_dir=data_dir,
        output_dir=output_dir,
        pattern="*_atm.dat",
        viewing_angle=0.0,
        solar_azimuth=180.0
    )
    
    print(f"\n✓ Processed {len(results)} flight lines successfully!")
    return results


def create_mosaics(output_dir: Path):
    """
    Create mosaics from individual flight lines
    
    Parameters:
    -----------
    output_dir : Path
        Directory containing individual retrievals
    """
    print("\n" + "="*70)
    print("STEP 4: CREATING MOSAICS")
    print("="*70)
    
    # Mosaic each retrieval type
    retrieval_types = ['grain_size', 'broadband_albedo', 'radiative_forcing']
    
    for retrieval in retrieval_types:
        files = sorted(output_dir.glob(f"*_{retrieval}.tif"))
        
        if len(files) > 1:
            print(f"\nMosaicking {retrieval}...")
            print(f"  Found {len(files)} flight lines")
            
            mosaic_file = output_dir / f"{retrieval}_mosaic.tif"
            mosaic = mosaic_flight_lines(
                input_files=files,
                output_file=mosaic_file,
                method='mean'
            )
            
            print(f"  ✓ Saved mosaic: {mosaic_file}")
        else:
            print(f"\nSkipping {retrieval} mosaic (only {len(files)} file)")


def create_visualizations(output_dir: Path, viz_dir: Path):
    """
    Create visualizations of results
    
    Parameters:
    -----------
    output_dir : Path
        Directory with processing results
    viz_dir : Path
        Directory for visualizations
    """
    print("\n" + "="*70)
    print("STEP 5: CREATING VISUALIZATIONS")
    print("="*70)
    
    # Find mosaic or first flight line files
    grain_size_file = (output_dir / "grain_size_mosaic.tif" 
                       if (output_dir / "grain_size_mosaic.tif").exists()
                       else sorted(output_dir.glob("*_grain_size.tif"))[0])
    
    albedo_file = (output_dir / "broadband_albedo_mosaic.tif"
                   if (output_dir / "broadband_albedo_mosaic.tif").exists()
                   else sorted(output_dir.glob("*_broadband_albedo.tif"))[0])
    
    rf_file = (output_dir / "radiative_forcing_mosaic.tif"
               if (output_dir / "radiative_forcing_mosaic.tif").exists()
               else sorted(output_dir.glob("*_radiative_forcing.tif"))[0])
    
    # Load data
    with rasterio.open(grain_size_file) as src:
        grain_size = src.read(1)
    
    with rasterio.open(albedo_file) as src:
        albedo = src.read(1)
    
    with rasterio.open(rf_file) as src:
        rf_lap = src.read(1)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    print("  - Grain size...")
    visualize_retrieval(
        grain_size,
        title="Optical Grain Size (μm)",
        cmap='YlOrRd',
        vmin=50,
        vmax=500,
        output_file=viz_dir / "grain_size.png"
    )
    
    print("  - Broadband albedo...")
    visualize_retrieval(
        albedo,
        title="Broadband Albedo",
        cmap='gray_r',
        vmin=0.3,
        vmax=0.95,
        output_file=viz_dir / "broadband_albedo.png"
    )
    
    print("  - Radiative forcing...")
    visualize_retrieval(
        rf_lap,
        title="Radiative Forcing by LAPs (W/m²)",
        cmap='RdPu',
        vmin=0,
        vmax=100,
        output_file=viz_dir / "radiative_forcing.png"
    )
    
    # Calculate and print statistics
    print("\nRetrieval Statistics:")
    print("-" * 50)
    
    for name, data in [('Grain Size (μm)', grain_size),
                       ('Broadband Albedo', albedo),
                       ('Radiative Forcing (W/m²)', rf_lap)]:
        stats = calculate_statistics(data)
        print(f"\n{name}:")
        print(f"  Mean:   {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Std:    {stats['std']:.2f}")
        print(f"  Range:  {stats['min']:.2f} - {stats['max']:.2f}")
        print(f"  P5-P95: {stats['p5']:.2f} - {stats['p95']:.2f}")
        print(f"  Valid pixels: {stats['count']}")
    
    print(f"\n✓ Visualizations saved to: {viz_dir}")


def main():
    """
    Complete ISSIA workflow
    """
    print("="*70)
    print("ISSIA - IMAGING SPECTROMETER SNOW AND ICE ALGORITHM")
    print("Python Implementation with Dask")
    print("="*70)
    
    # Setup
    dirs = setup_directories()
    
    # Define instrument wavelengths
    # Example: AisaFENIX (380-2500 nm, 451 bands)
    wavelengths = np.linspace(380, 2500, 451)
    
    # OPTION 1: Generate LUTs (first time only)
    print("\nDo you need to generate lookup tables?")
    print("  y - Yes, generate new LUTs (SLOW - may take hours)")
    print("  q - Yes, generate quick LUTs (FAST - for testing only)")
    print("  n - No, use existing LUTs")
    
    response = input("Choice (y/q/n): ").lower()
    
    if response in ['y', 'q']:
        grain_radii, illum, view, azim = generate_luts(
            dirs['lookup_tables'],
            wavelengths,
            quick_mode=(response == 'q')
        )
    else:
        print("\nUsing existing LUTs...")
        # These should match your LUT dimensions
        grain_radii = np.logspace(np.log10(30), np.log10(5000), 50)
        illum = np.arange(0, 85, 5)
        view = np.arange(0, 65, 5)
        azim = np.arange(0, 185, 15)
    
    # Initialize processor
    processor = initialize_processor(
        wavelengths, grain_radii, illum, view, azim,
        dirs['lookup_tables'],
        coord_ref_sys=32610  # Adjust for your area
    )
    
    # Check if we have input data
    atcor_files = list(dirs['atcor_output'].glob("*_atm.dat"))
    
    if len(atcor_files) == 0:
        print("\n" + "!"*70)
        print("WARNING: No ATCOR-4 files found in atcor_output/")
        print("!"*70)
        print("\nPlease add your ATCOR-4 output files to:")
        print(f"  {dirs['atcor_output'].absolute()}")
        print("\nRequired files for each flight line:")
        print("  - {flight_line}.inn")
        print("  - {flight_line}_atm.dat + .hdr")
        print("  - {flight_line}_eglo.dat + .hdr")
        print("  - {flight_line}_slp.dat + .hdr")
        print("  - {flight_line}_asp.dat + .hdr")
        print("\nExiting...")
        return
    
    # Process data
    results = process_data(
        processor,
        dirs['atcor_output'],
        dirs['issia_results']
    )
    
    # Create mosaics if multiple flight lines
    create_mosaics(dirs['issia_results'])
    
    # Create visualizations
    create_visualizations(
        dirs['issia_results'],
        dirs['visualizations']
    )
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE!")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  Processing: {dirs['issia_results'].absolute()}")
    print(f"  Visualizations: {dirs['visualizations'].absolute()}")
    print("\nOutput files:")
    print("  - *_grain_size.tif: Optical grain radius (μm)")
    print("  - *_broadband_albedo.tif: Broadband albedo (0-1)")
    print("  - *_radiative_forcing.tif: RF by LAPs (W/m²)")
    print("  - *_mosaic.tif: Mosaicked results (if multiple flight lines)")
    print("="*70)


if __name__ == "__main__":
    main()
