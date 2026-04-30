#!/usr/bin/env python
"""Check validity of input files for a flight line."""
import argparse
import rasterio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True)
parser.add_argument('--flight-line', required=True)
args = parser.parse_args()

for suffix in ['_slp.dat', '_atm.dat', '_eglo.dat', '_asp.dat']:
    path = f"{args.data_dir}/{args.flight_line}{suffix}"
    try:
        with rasterio.open(path) as src:
            data = src.read(1)
            nd = src.nodata
            valid = data[data != nd] if nd is not None else data[data != 0]
            if len(valid):
                print(f"{suffix}: nodata={nd}, valid_px={len(valid)}/{data.size}, range={valid.min():.2f}–{valid.max():.2f}")
            else:
                print(f"{suffix}: ALL NODATA (nodata={nd})")
    except Exception as e:
        print(f"{suffix}: ERROR — {e}")
