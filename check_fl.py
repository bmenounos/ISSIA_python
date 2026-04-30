#!/usr/bin/env python
"""Check validity and spatial overlap of input files for a flight line."""
import argparse
import rasterio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True)
parser.add_argument('--flight-line', required=True)
parser.add_argument('--wvl-path', default='wvl.npy')
args = parser.parse_args()

print(f"Flight line: {args.flight_line}\n")

wavelengths = np.load(args.wvl_path)
idx_600  = int(np.argmin(np.abs(wavelengths - 600)))
idx_1500 = int(np.argmin(np.abs(wavelengths - 1500)))
idx_560  = int(np.argmin(np.abs(wavelengths - 560)))
print(f"Wavelengths: {len(wavelengths)} bands, {wavelengths[0]:.0f}–{wavelengths[-1]:.0f} nm")
print(f"Key bands: 560nm→{idx_560}  600nm→{idx_600}  1500nm→{idx_1500}\n")

stats = {}
for suffix in ['_slp.dat', '_atm.dat', '_eglo.dat', '_asp.dat']:
    path = f"{args.data_dir}/{args.flight_line}{suffix}"
    try:
        with rasterio.open(path) as src:
            nd = src.nodata
            # Check band 1 for overall spatial validity
            b1 = src.read(1)
            valid_mask = (b1 != nd) if nd is not None else (b1 != 0)
            n_valid = int(np.sum(valid_mask))
            print(f"{suffix}: shape={b1.shape}, nodata={nd}, "
                  f"valid={n_valid}/{b1.size} ({100*n_valid/b1.size:.1f}%)")
            if n_valid:
                rows = np.where(valid_mask.any(axis=1))[0]
                cols = np.where(valid_mask.any(axis=0))[0]
                print(f"  valid rows: {rows[0]}–{rows[-1]}  cols: {cols[0]}–{cols[-1]}")

            # For ATM file, also check the key spectral bands
            if suffix == '_atm.dat' and src.count >= max(idx_600, idx_1500) + 1:
                for label, idx in [('560nm', idx_560), ('600nm', idx_600), ('1500nm', idx_1500)]:
                    band = src.read(idx + 1).astype(np.float32)  # rasterio bands are 1-indexed
                    n_zero = int(np.sum(band == 0))
                    n_nd   = int(np.sum(band == nd)) if nd is not None else 0
                    n_valid_band = band.size - n_zero - n_nd
                    print(f"  band {idx} ({label}): {n_valid_band}/{band.size} non-zero non-nodata  "
                          f"range={float(band[band > 0].min()) if n_valid_band else 'N/A':.0f}–"
                          f"{float(band.max()):.0f}")

            stats[suffix] = valid_mask
    except Exception as e:
        print(f"{suffix}: ERROR — {e}")
        stats[suffix] = None
    print()

# Overlap between ATM and slope
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
            print(f"  overlap rows: {rows[0]}–{rows[-1]}  cols: {cols[0]}–{cols[-1]}")
