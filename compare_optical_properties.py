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
    col_start = int((left - bounds.left) / transform[0])
    col_end = int((right - bounds.left) / transform[0])
    row_start = int((bounds.top - top) / abs(transform[4]))
    row_end = int((bounds.top - bottom) / abs(transform[4]))
    col_start = max(0, col_start)
    row_start = max(0, row_start)
    return data[row_start:row_end, col_start:col_end]

def compare_products(matlab_file, python_file, product_name):
    """Calculate statistics and prepare data for plotting."""
    m_data, m_trans, m_bounds, m_crs = load_geotiff(matlab_file)
    p_data, p_trans, p_bounds, p_crs = load_geotiff(python_file)
    
    # Use geospatial bounds to properly align, not just dimensions
    common_bounds = get_common_extent(m_bounds, p_bounds)
    m_crop = crop_to_common_extent(m_data, m_trans, m_bounds, common_bounds)
    p_crop = crop_to_common_extent(p_data, p_trans, p_bounds, common_bounds)
    
    # Match dimensions after geospatial crop
    min_rows = min(m_crop.shape[0], p_crop.shape[0])
    min_cols = min(m_crop.shape[1], p_crop.shape[1])
    m_crop = m_crop[:min_rows, :min_cols]
    p_crop = p_crop[:min_rows, :min_cols]
    
    mask = ~np.isnan(m_crop) & ~np.isnan(p_crop) & (m_crop != 0)
    if not np.any(mask):
        return None
        
    m_valid = m_crop[mask]
    p_valid = p_crop[mask]
    
    rmse = np.sqrt(np.mean((m_valid - p_valid)**2))
    correlation = stats.pearsonr(m_valid, p_valid)[0]
    mean_pct_diff = np.mean(np.abs((p_valid - m_valid) / m_valid)) * 100
    
    return {
        'matlab': m_crop,
        'python': p_crop,
        'rmse': rmse,
        'correlation': correlation,
        'mean_pct_diff': mean_pct_diff,
        'mask': mask
    }

def plot_comparison(result, product_name, output_dir):
    """Plot comparison with tightly cropped maps and perfectly scaled colorbars."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    
    # ROW 1: SPATIAL MAPS
    rows, cols = np.where(result['mask'])
    y_min, y_max, x_min, x_max = rows.min(), rows.max(), cols.min(), cols.max()
    
    vmin, vmax = np.nanmin(result['matlab']), np.nanmax(result['matlab'])
    diff_data = result['python'] - result['matlab']
    vlimit = np.nanpercentile(np.abs(diff_data[result['mask']]), 98)

    # 1. MATLAB
    im0 = axes[0, 0].imshow(result['matlab'], cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'MATLAB {product_name}')
    
    # 2. Python
    im1 = axes[0, 1].imshow(result['python'], cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Python {product_name}')
    
    # 3. Difference
    im2 = axes[0, 2].imshow(diff_data, cmap='RdBu_r', vmin=-vlimit, vmax=vlimit)
    axes[0, 2].set_title('Difference (Py - Mat)')

    # Apply zoom and scale colorbars to match data height
    for ax, im in zip(axes[0, :], [im0, im1, im2]):
        ax.set_ylim(y_max + 5, y_min - 5)
        ax.set_xlim(x_min - 5, x_max + 5)
        ax.axis('tight')
        # fraction=0.046 and pad=0.04 ensures the colorbar height matches the plot height
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ROW 2: STATISTICAL PLOTS
    mask = result['mask']
    m_val, p_val = result['matlab'][mask], result['python'][mask]
    diff_val = diff_data[mask]

    # 4. Scatter
    axes[1, 0].scatter(m_val[::10], p_val[::10], s=1, alpha=0.3)
    ax_min, ax_max = min(np.nanmin(m_val), np.nanmin(p_val)), max(np.nanmax(m_val), np.nanmax(p_val))
    axes[1, 0].plot([ax_min, ax_max], [ax_min, ax_max], 'r--', label='1:1')
    axes[1, 0].set_title(f'Correlation (r={result["correlation"]:.4f})')
    axes[1, 0].set_box_aspect(1)

    # 5. Abs Diff
    axes[1, 1].hist(diff_val, bins=50, edgecolor='black', color='steelblue')
    axes[1, 1].axvline(0, color='red', linestyle='--')
    axes[1, 1].set_title(f'Abs Diff (RMSE={result["rmse"]:.3f})')
    axes[1, 1].set_box_aspect(1)

    # 6. Rel Diff
    pct_diff = (diff_val / np.where(m_val == 0, np.nan, m_val)) * 100
    axes[1, 2].hist(pct_diff, bins=50, color='orange', edgecolor='black')
    axes[1, 2].axvline(0, color='red', linestyle='--')
    axes[1, 2].set_title(f'Relative Diff (Mean={result["mean_pct_diff"]:.2f}%)')
    axes[1, 2].set_box_aspect(1)

    fig.suptitle(f'Comparison: {product_name}', fontsize=18, fontweight='bold')
    plt.savefig(output_dir / f"{product_name.lower().replace(' ', '_')}_comparison.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    matlab_dir = Path("matlab_outputs")
    python_dir = Path("products")
    output_dir = Path("comparison_plots")
    output_dir.mkdir(exist_ok=True)
    
    flight_line = "24_4012_05_2024-06-06_17-54-38-rect_img"
    
    # No hardcoded subset - auto-matches Python dimensions
    
    products = {
        'Grain Size': ('HSI_grainsize', 'gs'),
        'Albedo': ('HSI_albedo', 'albedo'),
        'Radiative Forcing': ('HSI_radiativeforcing', 'rf')
    }
    
    results_summary = {}
    
    for name, (m_suf, p_suf) in products.items():
        m_file = matlab_dir / f"{flight_line}_{m_suf}.tif"
        p_file = python_dir / f"{flight_line}_{p_suf}.tif"
        if m_file.exists() and p_file.exists():
            print(f"Comparing {name}...")
            res = compare_products(m_file, p_file, name)
            if res:
                results_summary[name] = res
                plot_comparison(res, name, output_dir)

    if results_summary:
        print(f"\n{'='*75}\n{'Product':<20} | {'RMSE':<12} | {'Pearson R':<12} | {'Mean % Diff':<12}\n{'-'*75}")
        for n, r in results_summary.items():
            print(f"{n:<20} | {r['rmse']:<12.4f} | {r['correlation']:<12.4f} | {r['mean_pct_diff']:<12.2f}%")
