#!/usr/bin/env python
"""Check validity and spatial overlap of input files for a flight line."""
import argparse
import rasterio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True)
parser.add_argument('--flight-line', required=True)
args = parser.parse_args()

print(f"Flight line: {args.flight_line}\n")

stats = {}
for suffix in ['_slp.dat', '_atm.dat', '_eglo.dat', '_asp.dat']:
    path = f"{args.data_dir}/{args.flight_line}{suffix}"
    try:
        with rasterio.open(path) as src:
            data = src.read(1)
            nd = src.nodata
            if nd is not None:
                valid_mask = data != nd
            else:
                valid_mask = data != 0
            n_valid = int(np.sum(valid_mask))
            print(f"{suffix}: shape={data.shape}, nodata={nd}, "
                  f"valid={n_valid}/{data.size} ({100*n_valid/data.size:.1f}%)")
            if n_valid:
                rows = np.where(valid_mask.any(axis=1))[0]
                cols = np.where(valid_mask.any(axis=0))[0]
                print(f"  valid rows: {rows[0]}–{rows[-1]}  "
                      f"valid cols: {cols[0]}–{cols[-1]}")
            stats[suffix] = valid_mask
    except Exception as e:
        print(f"{suffix}: ERROR — {e}")
        stats[suffix] = None

# Show overlap between ATM and slope
print()
if '_atm.dat' in stats and '_slp.dat' in stats:
    atm = stats['_atm.dat']
    slp = stats['_slp.dat']
    if atm is not None and slp is not None and atm.shape == slp.shape:
        overlap = atm & slp
        n_overlap = int(np.sum(overlap))
        print(f"ATM ∩ slope overlap: {n_overlap}/{atm.size} ({100*n_overlap/atm.size:.1f}%)")
        if n_overlap:
            rows = np.where(overlap.any(axis=1))[0]
            cols = np.where(overlap.any(axis=0))[0]
            print(f"  overlap rows: {rows[0]}–{rows[-1]}  "
                  f"overlap cols: {cols[0]}–{cols[-1]}")
    elif atm is not None and slp is not None:
        print(f"ATM shape {atm.shape} ≠ slope shape {slp.shape} — cannot compute overlap")
