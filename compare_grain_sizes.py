#!/usr/bin/env python
import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

def load_geotiff(filepath):
    """Load a GeoTIFF and return data, transform, and bounds."""
    with rasterio.open(filepath) as src:
        data = src.read(1)
        transform = src.transform
        bounds = src.bounds
        crs = src.crs
    return data, transform, bounds, crs

def get_common_extent(bounds1, bounds2):
    """Get the overlapping extent of two bounding boxes."""
    common_bounds = (
        max(bounds1.left, bounds2.left),
        max(bounds1.bottom, bounds2.bottom),
        min(bounds1.right, bounds2.right),
        min(bounds1.top, bounds2.top)
    )
    return common_bounds

def crop_to_common_extent(data, transform, bounds, common_bounds):
    """Crop raster data to common extent."""
    left, bottom, right, top = common_bounds
    
    # Convert spatial coordinates to pixel coordinates
    col_start = int((left - bounds.left) / transform[0])
    col_end = int((right - bounds.left) / transform[0])
    row_start = int((bounds.top - top) / abs(transform[4]))
    row_end = int((bounds.top - bottom) / abs(transform[4]))
    
    # Ensure indices are valid
    col_start = max(0, col_start)
    col_end = min(data.shape[1], col_end)
    row_start = max(0, row_start)
    row_end = min(data.shape[0], row_end)
    
    cropped = data[row_start:row_end, col_start:col_end]
    return cropped

def compare_products(matlab_file, python_file, product_name):
    """Compare MATLAB and Python products."""
    print(f"\n{'='*70}")
    print(f"Comparing {product_name}")
    print(f"{'='*70}")
    
    # Load data
    matlab_data, matlab_transform, matlab_bounds, matlab_crs = load_geotiff(matlab_file)
    python_data, python_transform, python_bounds, python_crs = load_geotiff(python_file)
    
    print(f"MATLAB shape: {matlab_data.shape}")
    print(f"Python shape: {python_data.shape}")
    print(f"MATLAB bounds: {matlab_bounds}")
    print(f"Python bounds: {python_bounds}")
    
    # Get common extent
    common_bounds = get_common_extent(matlab_bounds, python_bounds)
    print(f"Common bounds: {common_bounds}")
    
    # Crop to common extent
    matlab_crop = crop_to_common_extent(matlab_data, matlab_transform, matlab_bounds, common_bounds)
    python_crop = crop_to_common_extent(python_data, python_transform, python_bounds, common_bounds)
    
    print(f"Cropped MATLAB shape: {matlab_crop.shape}")
    print(f"Cropped Python shape: {python_crop.shape}")
    
    # Ensure same shape (handle off-by-one differences)
    min_rows = min(matlab_crop.shape[0], python_crop.shape[0])
    min_cols = min(matlab_crop.shape[1], python_crop.shape[1])
    matlab_crop = matlab_crop[:min_rows, :min_cols]
    python_crop = python_crop[:min_rows, :min_cols]
    
    # Mask invalid values
    valid_mask = ~(np.isnan(matlab_crop) | np.isnan(python_crop) | 
                   (matlab_crop == 0) | (python_crop == 0))
    
    matlab_valid = matlab_crop[valid_mask]
    python_valid = python_crop[valid_mask]
    
    print(f"\nValid pixels: {np.sum(valid_mask)} / {valid_mask.size} ({100*np.sum(valid_mask)/valid_mask.size:.1f}%)")
    
    if len(matlab_valid) == 0:
        print("WARNING: No valid overlapping pixels found!")
        return None
    
    # Statistics
    diff = python_valid - matlab_valid
    percent_diff = 100 * diff / np.abs(matlab_valid)
    
    print(f"\nMATLAB stats:")
    print(f"  Mean: {np.mean(matlab_valid):.3f}")
    print(f"  Std:  {np.std(matlab_valid):.3f}")
    print(f"  Range: [{np.min(matlab_valid):.3f}, {np.max(matlab_valid):.3f}]")
    
    print(f"\nPython stats:")
    print(f"  Mean: {np.mean(python_valid):.3f}")
    print(f"  Std:  {np.std(python_valid):.3f}")
    print(f"  Range: [{np.min(python_valid):.3f}, {np.max(python_valid):.3f}]")
    
    print(f"\nDifference (Python - MATLAB):")
    print(f"  Mean: {np.mean(diff):.3f}")
    print(f"  Std:  {np.std(diff):.3f}")
    print(f"  Range: [{np.min(diff):.3f}, {np.max(diff):.3f}]")
    print(f"  Mean % diff: {np.mean(percent_diff):.2f}%")
    print(f"  Median % diff: {np.median(percent_diff):.2f}%")
    
    # Correlation
    r, p = stats.pearsonr(matlab_valid, python_valid)
    print(f"\nPearson correlation: r={r:.6f}, p={p:.2e}")
    
    # RMSE
    rmse = np.sqrt(np.mean(diff**2))
    nrmse = 100 * rmse / (np.max(matlab_valid) - np.min(matlab_valid))
    print(f"RMSE: {rmse:.3f}")
    print(f"Normalized RMSE: {nrmse:.2f}%")
    
    return {
        'matlab_crop': matlab_crop,
        'python_crop': python_crop,
        'valid_mask': valid_mask,
        'matlab_valid': matlab_valid,
        'python_valid': python_valid,
        'diff': diff,
        'percent_diff': percent_diff,
        'r': r,
        'rmse': rmse
    }

def plot_comparison(results_dict, product_name, output_dir):
    """Create comparison plots for a product."""
    if results_dict is None:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{product_name} Comparison: MATLAB vs Python', fontsize=16, fontweight='bold')
    
    # MATLAB data
    im1 = axes[0, 0].imshow(results_dict['matlab_crop'], cmap='viridis')
    axes[0, 0].set_title('MATLAB')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Python data
    im2 = axes[0, 1].imshow(results_dict['python_crop'], cmap='viridis')
    axes[0, 1].set_title('Python')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Difference map
    diff_map = np.full(results_dict['matlab_crop'].shape, np.nan)
    diff_map[results_dict['valid_mask']] = results_dict['diff']
    vmax = np.nanmax(np.abs(diff_map))
    im3 = axes[0, 2].imshow(diff_map, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 2].set_title('Difference (Python - MATLAB)')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Scatter plot
    axes[1, 0].scatter(results_dict['matlab_valid'], results_dict['python_valid'], 
                       alpha=0.3, s=1)
    axes[1, 0].plot([results_dict['matlab_valid'].min(), results_dict['matlab_valid'].max()],
                    [results_dict['matlab_valid'].min(), results_dict['matlab_valid'].max()],
                    'r--', lw=2, label='1:1 line')
    axes[1, 0].set_xlabel('MATLAB')
    axes[1, 0].set_ylabel('Python')
    axes[1, 0].set_title(f'Scatter (r={results_dict["r"]:.4f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Difference histogram
    axes[1, 1].hist(results_dict['diff'], bins=50, edgecolor='black')
    axes[1, 1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Difference')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Difference Distribution (RMSE={results_dict["rmse"]:.3f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Percent difference histogram
    axes[1, 2].hist(results_dict['percent_diff'], bins=50, edgecolor='black')
    axes[1, 2].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 2].set_xlabel('Percent Difference (%)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Percent Difference Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f'{product_name.lower().replace(" ", "_")}_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Directories
    matlab_dir = Path("matlab_outputs")
    python_dir = Path("issia_results_optimized")
    output_dir = Path("comparison_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Flight line base name
    flight_line = "24_4012_05_2024-06-06_17-54-38-rect_img"
    
    # Products to compare (matlab_suffix, python_suffix)
    products = {
        'Grain Size': ('HSI_grainsize', 'gs'),
        'Albedo': ('HSI_albedo', 'albedo'),
        'Radiative Forcing': ('HSI_radiativeforcing', 'rf')
    }
    
    results = {}
    
    for product_name, (matlab_suffix, python_suffix) in products.items():
        matlab_file = matlab_dir / f"{flight_line}_{matlab_suffix}.tif"
        python_file = python_dir / f"{flight_line}_{python_suffix}.tif"
        
        if not matlab_file.exists():
            print(f"\nWARNING: MATLAB file not found: {matlab_file}")
            continue
        if not python_file.exists():
            print(f"\nWARNING: Python file not found: {python_file}")
            continue
        
        # Compare
        result = compare_products(matlab_file, python_file, product_name)
        
        if result is not None:
            results[product_name] = result
            plot_comparison(result, product_name, output_dir)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for product_name, result in results.items():
        print(f"\n{product_name}:")
        print(f"  Correlation: r={result['r']:.6f}")
        print(f"  RMSE: {result['rmse']:.3f}")
        print(f"  Mean % diff: {np.mean(result['percent_diff']):.2f}%")
