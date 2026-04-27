import os
import glob
import argparse
import numpy as np
from osgeo import gdal
from scipy.ndimage import distance_transform_edt

def create_global_weight_map(file_path):
    """
    Calculates the weight map for the entire flightline relative to its 
    actual data boundaries before any tiling occurs.
    """
    src_ds = gdal.Open(file_path)
    if src_ds is None:
        return None, None
        
    band = src_ds.GetRasterBand(1).ReadAsArray()
    
    # Identify valid data area
    # Note: If 0 is a valid retrieval, use np.isnan(band) only
    mask = np.where(np.isnan(band) | (band <= 0), 0, 1)
    
    # Calculate Euclidean Distance to the nearest NoData pixel
    weight = distance_transform_edt(mask).astype(np.float32)
    
    # Normalize weights 0 to 1
    max_dist = weight.max()
    if max_dist > 0:
        weight /= max_dist
        
    # Keep the weight map in a MEMory dataset for fast warping
    mem_drv = gdal.GetDriverByName('MEM')
    w_ds = mem_drv.Create('', src_ds.RasterXSize, src_ds.RasterYSize, 1, gdal.GDT_Float32)
    w_ds.SetGeoTransform(src_ds.GetGeoTransform())
    w_ds.SetProjection(src_ds.GetProjection())
    w_ds.GetRasterBand(1).WriteArray(weight)
    
    return src_ds, w_ds

def get_tile_data(src_ds, w_ds, target_vrt, x, y, win_x, win_y):
    """
    Warps a specific window of both the data and the weight map 
    into the target mosaic grid.
    """
    mem_drv = gdal.GetDriverByName('MEM')
    orig_geo = target_vrt.GetGeoTransform()
    
    # Calculate the geo-transform for this specific tile/window
    tile_geo = (
        orig_geo[0] + x * orig_geo[1], orig_geo[1], 0,
        orig_geo[3] + y * orig_geo[5], 0, orig_geo[5]
    )
    
    def warp_op(input_ds):
        tmp_ds = mem_drv.Create('', win_x, win_y, 1, gdal.GDT_Float32)
        tmp_ds.SetGeoTransform(tile_geo)
        tmp_ds.SetProjection(target_vrt.GetProjection())
        # Use Bilinear for smooth physical parameters (Albedo, Grain Size)
        gdal.ReprojectImage(input_ds, tmp_ds, input_ds.GetProjection(), 
                            target_vrt.GetProjection(), gdal.GRA_Bilinear)
        return tmp_ds.GetRasterBand(1).ReadAsArray()

    data_tile = warp_op(src_ds)
    weight_tile = warp_op(w_ds)
    
    # Replace NaNs with 0 for the weighted sum calculation
    return np.nan_to_num(data_tile), np.nan_to_num(weight_tile)

def main():
    parser = argparse.ArgumentParser(description="Artifact-free weighted mosaic for physical retrievals.")
    parser.add_argument("-i", "--input", required=True, help="Input wildcard (e.g. 'folder/*.tif')")
    parser.add_argument("-o", "--output", required=True, help="Output filename")
    parser.add_argument("-t", "--tile_size", type=int, default=2048, help="Tile size for processing")
    
    args = parser.parse_args()

    # Resolve wildcard
    files = glob.glob(args.input)
    if not files:
        print(f"No files found for: {args.input}")
        return

    print(f"Step 1: Pre-calculating global weights for {len(files)} files...")
    # Store tuples of (Source Dataset, Weight Dataset)
    prepared = []
    for f in files:
        s_ds, w_ds = create_global_weight_map(f)
        if s_ds:
            prepared.append((s_ds, w_ds))

    # Determine Global Extent
    temp_vrt_path = 'global_temp.vrt'
    gdal.BuildVRT(temp_vrt_path, files)
    vrt_ds = gdal.Open(temp_vrt_path)
    x_total, y_total = vrt_ds.RasterXSize, vrt_ds.RasterYSize

    # Create Output File
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(args.output, x_total, y_total, 1, gdal.GDT_Float32, 
                           options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])
    out_ds.SetGeoTransform(vrt_ds.GetGeoTransform())
    out_ds.SetProjection(vrt_ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(np.nan)

    print(f"Step 2: Processing mosaic in {args.tile_size}px tiles...")
    
    for y in range(0, y_total, args.tile_size):
        win_y = min(args.tile_size, y_total - y)
        for x in range(0, x_total, args.tile_size):
            win_x = min(args.tile_size, x_total - x)
            
            tile_sum = np.zeros((win_y, win_x), dtype=np.float32)
            tile_w_acc = np.zeros((win_y, win_x), dtype=np.float32)
            
            for s_ds, w_ds in prepared:
                d_tile, w_tile = get_tile_data(s_ds, w_ds, vrt_ds, x, y, win_x, win_y)
                tile_sum += (d_tile * w_tile)
                tile_w_acc += w_tile
            
            # Divide by total weight to get the final feathered value
            valid = tile_w_acc > 0
            final_tile = np.full((win_y, win_x), np.nan, dtype=np.float32)
            final_tile[valid] = tile_sum[valid] / tile_w_acc[valid]
            
            out_band.WriteArray(final_tile, x, y)
        print(f" Progress: Row {y} of {y_total} complete")

    print(f"Done! Clean mosaic saved to: {args.output}")
    
    # Cleanup
    out_ds = None
    vrt_ds = None
    if os.path.exists(temp_vrt_path):
        os.remove(temp_vrt_path)

if __name__ == "__main__":
    main()
