"""
Seamless weighted mosaic for ISSIA flight line products.

Uses distance-transform edge weighting so overlap zones blend smoothly and
ATCOR edge artifacts (strongest at flight line boundaries) are suppressed.

Usage — single product:
    python mosaic.py -i "output/*_albedo.tif" -o mosaic_albedo.tif

Usage — all three products from a batch output directory:
    python mosaic.py --batch-dir /path/to/output --mosaic-dir /path/to/mosaics

Parameters:
    --edge-setback   Pixels to trim from each flight line edge before distance
                     weighting (default 50). Handles ATCOR correction artefacts
                     that are strongest near the swath boundary.
    --tile-size      Processing tile size in pixels (default 2048).
"""

import os
import glob
import argparse
from pathlib import Path

import numpy as np
from osgeo import gdal
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


PRODUCTS = ["gs", "albedo", "rf"]


def _build_weight(band, edge_setback):
    """Distance-transform weight for one flight line raster band.

    Pixels within `edge_setback` of any nodata edge are forced to zero weight,
    then the remaining valid region is weighted by distance to its own edge.
    """
    valid = np.where(np.isnan(band) | (band <= 0), 0, 1).astype(np.uint8)

    if edge_setback > 0:
        # Erode the valid mask by edge_setback pixels
        from scipy.ndimage import binary_erosion
        struct = np.ones((2 * edge_setback + 1, 2 * edge_setback + 1), dtype=bool)
        valid = binary_erosion(valid, structure=struct, border_value=0).astype(np.uint8)

    weight = distance_transform_edt(valid).astype(np.float32)
    max_w = weight.max()
    if max_w > 0:
        weight /= max_w
    return weight


def _to_mem_ds(array, ref_ds):
    """Wrap a numpy array in an in-memory GDAL dataset with the same georef as ref_ds."""
    drv = gdal.GetDriverByName("MEM")
    ds = drv.Create("", ref_ds.RasterXSize, ref_ds.RasterYSize, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(ref_ds.GetGeoTransform())
    ds.SetProjection(ref_ds.GetProjection())
    ds.GetRasterBand(1).WriteArray(array)
    return ds


def _warp_tile(src_ds, target_ds, x, y, win_x, win_y):
    """Reproject a spatial tile of src_ds onto the target grid."""
    drv = gdal.GetDriverByName("MEM")
    gt = target_ds.GetGeoTransform()
    tile_gt = (
        gt[0] + x * gt[1], gt[1], 0,
        gt[3] + y * gt[5], 0, gt[5],
    )
    tmp = drv.Create("", win_x, win_y, 1, gdal.GDT_Float32)
    tmp.SetGeoTransform(tile_gt)
    tmp.SetProjection(target_ds.GetProjection())
    gdal.ReprojectImage(src_ds, tmp,
                        src_ds.GetProjection(), target_ds.GetProjection(),
                        gdal.GRA_Bilinear)
    return tmp.GetRasterBand(1).ReadAsArray()


def mosaic_files(files, output_path, tile_size=2048, edge_setback=50):
    """Create a seamless weighted mosaic from a list of GeoTIFF files.

    Parameters
    ----------
    files : list[str]
        Input GeoTIFF paths (one per flight line).
    output_path : str | Path
        Output mosaic GeoTIFF path.
    tile_size : int
        Spatial processing tile size in pixels.
    edge_setback : int
        Pixels to trim from each flight line edge before weighting.
    """
    files = sorted(files)
    output_path = str(output_path)

    print(f"  Mosaicking {len(files)} flight lines → {Path(output_path).name}")

    # Pre-compute per-file weight maps (full resolution, stored in memory)
    prepared = []
    for f in tqdm(files, desc="  weights", leave=False):
        src = gdal.Open(f)
        if src is None:
            tqdm.write(f"  Warning: cannot open {f}, skipping")
            continue
        band = src.GetRasterBand(1).ReadAsArray().astype(np.float32)
        nodata = src.GetRasterBand(1).GetNoDataValue()
        if nodata is not None:
            band[band == nodata] = np.nan
        w = _build_weight(band, edge_setback)
        w_ds = _to_mem_ds(w, src)
        prepared.append((src, w_ds))

    if not prepared:
        print("  No valid files found.")
        return

    # Determine output extent via VRT — force Float32 so mixed-type files don't cause skips
    vrt_path = output_path + ".tmp.vrt"
    gdal.BuildVRT(vrt_path, files,
                  options=gdal.BuildVRTOptions(outputType=gdal.GDT_Float32))
    vrt = gdal.Open(vrt_path)
    x_total, y_total = vrt.RasterXSize, vrt.RasterYSize

    # Create output GeoTIFF
    drv = gdal.GetDriverByName("GTiff")
    out_ds = drv.Create(output_path, x_total, y_total, 1, gdal.GDT_Float32,
                        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES",
                                 "BLOCKXSIZE=512", "BLOCKYSIZE=512"])
    out_ds.SetGeoTransform(vrt.GetGeoTransform())
    out_ds.SetProjection(vrt.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(float("nan"))

    n_tiles_y = (y_total + tile_size - 1) // tile_size
    n_tiles_x = (x_total + tile_size - 1) // tile_size
    n_tiles = n_tiles_y * n_tiles_x

    with tqdm(total=n_tiles, desc="  tiles", unit="tile", leave=False) as pbar:
        for y in range(0, y_total, tile_size):
            win_y = min(tile_size, y_total - y)
            for x in range(0, x_total, tile_size):
                win_x = min(tile_size, x_total - x)

                tile_sum = np.zeros((win_y, win_x), dtype=np.float32)
                tile_w = np.zeros((win_y, win_x), dtype=np.float32)

                for src_ds, w_ds in prepared:
                    d = _warp_tile(src_ds, vrt, x, y, win_x, win_y)
                    w = _warp_tile(w_ds, vrt, x, y, win_x, win_y)
                    np.nan_to_num(d, copy=False)
                    np.nan_to_num(w, copy=False)
                    tile_sum += d * w
                    tile_w += w

                valid = tile_w > 0
                result = np.full((win_y, win_x), np.nan, dtype=np.float32)
                result[valid] = tile_sum[valid] / tile_w[valid]
                out_band.WriteArray(result, x, y)
                pbar.update(1)

    out_ds.FlushCache()
    out_ds = None
    vrt = None
    if os.path.exists(vrt_path):
        os.remove(vrt_path)

    print(f"  Saved: {output_path}")


def mosaic_batch(batch_dir, mosaic_dir, tile_size=2048, edge_setback=50,
                 flight_lines=None):
    """Mosaic all three products (gs, albedo, rf) from a batch output directory.

    Parameters
    ----------
    flight_lines : list[str], optional
        If provided, only mosaic files belonging to these flight line IDs.
        Prevents stale files from other acquisitions mixing into the mosaic.
    """
    batch_dir = Path(batch_dir)
    mosaic_dir = Path(mosaic_dir)
    mosaic_dir.mkdir(parents=True, exist_ok=True)

    for product in PRODUCTS:
        if flight_lines is not None:
            files = sorted(
                batch_dir / f"{fl}_{product}.tif"
                for fl in flight_lines
                if (batch_dir / f"{fl}_{product}.tif").exists()
            )
        else:
            files = sorted(batch_dir.glob(f"*_{product}.tif"))

        # Exclude any existing mosaic outputs from the input list
        files = [f for f in files if not Path(f).name.startswith("mosaic_")]

        if not files:
            print(f"  No {product} files found in {batch_dir}")
            continue
        out_path = mosaic_dir / f"mosaic_{product}.tif"
        mosaic_files([str(f) for f in files], out_path,
                     tile_size=tile_size, edge_setback=edge_setback)


def main():
    parser = argparse.ArgumentParser(
        description="Seamless weighted mosaic for ISSIA flight line products.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", help="Input wildcard, e.g. 'folder/*_albedo.tif'")
    group.add_argument("--batch-dir", help="Batch output dir — mosaics all three products")

    parser.add_argument("-o", "--output", help="Output path (required with --input)")
    parser.add_argument("--mosaic-dir", default="mosaics",
                        help="Output dir for mosaics (used with --batch-dir)")
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--edge-setback", type=int, default=50,
                        help="Pixels to trim from swath edges before weighting")

    args = parser.parse_args()

    if args.batch_dir:
        mosaic_batch(args.batch_dir, args.mosaic_dir,
                     tile_size=args.tile_size, edge_setback=args.edge_setback)
    else:
        if not args.output:
            parser.error("--output is required when using --input")
        files = glob.glob(args.input)
        if not files:
            print(f"No files found for: {args.input}")
            return
        mosaic_files(files, args.output,
                     tile_size=args.tile_size, edge_setback=args.edge_setback)


if __name__ == "__main__":
    main()
