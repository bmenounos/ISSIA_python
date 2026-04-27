#!/usr/bin/env python
"""
ISSIA - Batch Flight Line Processing

Processes multiple ATCOR flight lines for snow property retrieval.
Can process all flight lines in a directory or a specific list.

Usage:
    # Process all flight lines in directory:
    python run_issia_batch.py --data-dir /path/to/data --output-dir /path/to/output
    
    # Process specific flight lines:
    python run_issia_batch.py --data-dir /path/to/data --output-dir /path/to/output \
        --flight-lines flight1 flight2 flight3
        
    # Process from file list:
    python run_issia_batch.py --data-dir /path/to/data --output-dir /path/to/output \
        --flight-list flight_lines.txt
"""

import numpy as np
import argparse
import time
from pathlib import Path
import warnings
from datetime import datetime
import traceback

from run_issia import process_flight_line

# Check for Numba
try:
    import numba
    print(f"Numba {numba.__version__} available - parallel processing enabled")
except ImportError:
    print("Warning: Install numba for 5-10x speedup: pip install numba")

warnings.filterwarnings('ignore')


def find_flight_lines(data_dir):
    """
    Find all flight lines in a directory based on _atm.dat files
    
    Parameters
    ----------
    data_dir : Path
        Directory containing ATCOR files
        
    Returns
    -------
    list
        List of flight line identifiers
    """
    data_dir = Path(data_dir)
    atm_files = sorted(data_dir.glob("*_atm.dat"))
    
    flight_lines = []
    for f in atm_files:
        # Extract flight line ID by removing _atm.dat suffix
        flight_line = f.stem.replace('_atm', '')
        
        # Verify all required files exist
        required = [
            f"{flight_line}_atm.dat",
            f"{flight_line}_eglo.dat",
            f"{flight_line}_slp.dat",
            f"{flight_line}_asp.dat",
            f"{flight_line}.inn"
        ]
        
        if all((data_dir / req).exists() for req in required):
            flight_lines.append(flight_line)
        else:
            print(f"Warning: Incomplete files for {flight_line}, skipping")
    
    return flight_lines


def batch_process(data_dir, output_dir, lut_dir, wvl_path, flight_lines=None, 
                 save_diagnostics=False, continue_on_error=True, n_workers=None):
    """
    Process multiple flight lines
    
    Parameters
    ----------
    data_dir : Path
        Directory containing ATCOR files
    output_dir : Path
        Output directory for results
    lut_dir : Path
        Directory containing lookup tables
    wvl_path : Path
        Path to wavelength file
    flight_lines : list, optional
        List of flight line IDs (if None, auto-detect)
    save_diagnostics : bool
        Whether to save diagnostic outputs
    continue_on_error : bool
        Continue processing other flight lines if one fails
        
    Returns
    -------
    dict
        Summary of processing results
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    lut_dir = Path(lut_dir)
    wvl_path = Path(wvl_path)
    
    # Find flight lines if not specified
    if flight_lines is None:
        flight_lines = find_flight_lines(data_dir)
    
    if not flight_lines:
        print("ERROR: No flight lines found to process")
        return None
    
    print("="*70)
    print(f"ISSIA BATCH PROCESSING")
    print(f"="*70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Flight lines to process: {len(flight_lines)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results tracking
    results = {
        'successful': [],
        'failed': [],
        'skipped': [],
        'stats': {}
    }
    
    t0_batch = time.time()
    
    for i, flight_line in enumerate(flight_lines, 1):
        print(f"\n{'#'*70}")
        print(f"# Flight line {i}/{len(flight_lines)}: {flight_line}")
        print(f"{'#'*70}")
        
        # Skip if all three outputs already exist
        expected = [output_dir / f"{flight_line}_{p}.tif" for p in ("gs", "albedo", "rf")]
        if all(p.exists() for p in expected):
            print(f"  ↩ Skipping (outputs exist)")
            results['skipped'].append(flight_line)
            continue

        try:
            t0 = time.time()
            output_files = process_flight_line(
                data_dir=data_dir,
                flight_line=flight_line,
                output_dir=output_dir,
                lut_dir=lut_dir,
                wvl_path=wvl_path,
                subset=None,
                save_diagnostics=save_diagnostics,
                n_workers=n_workers
            )
            
            elapsed = time.time() - t0
            results['successful'].append(flight_line)
            results['stats'][flight_line] = {
                'time': elapsed,
                'outputs': output_files
            }
            print(f"\n✓ {flight_line} completed in {elapsed:.1f}s")
            
        except Exception as e:
            results['failed'].append(flight_line)
            results['stats'][flight_line] = {
                'error': str(e)
            }
            print(f"\n✗ {flight_line} FAILED: {e}")
            
            if not continue_on_error:
                print("\nStopping batch due to error (use --continue-on-error to override)")
                break
            
            traceback.print_exc()
    
    # Print summary
    total_time = time.time() - t0_batch
    
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {len(results['successful'])}/{len(flight_lines)}")
    print(f"Skipped:    {len(results['skipped'])} (outputs already exist)")
    print(f"Failed:     {len(results['failed'])}")
    
    if results['failed']:
        print(f"\nFailed flight lines:")
        for fl in results['failed']:
            error = results['stats'][fl].get('error', 'Unknown error')
            print(f"  - {fl}: {error}")
    
    if results['successful']:
        avg_time = sum(results['stats'][fl]['time'] for fl in results['successful']) / len(results['successful'])
        print(f"\nAverage processing time: {avg_time:.1f}s per flight line")
    
    print("="*70)
    
    # Save processing log
    log_file = output_dir / "batch_processing_log.txt"
    with open(log_file, 'w') as f:
        f.write(f"ISSIA Batch Processing Log\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Total flight lines: {len(flight_lines)}\n")
        f.write(f"Successful: {len(results['successful'])}\n")
        f.write(f"Failed: {len(results['failed'])}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n\n")
        
        f.write("Successful:\n")
        for fl in results['successful']:
            f.write(f"  - {fl} ({results['stats'][fl]['time']:.1f}s)\n")
        
        if results['failed']:
            f.write("\nFailed:\n")
            for fl in results['failed']:
                f.write(f"  - {fl}: {results['stats'][fl].get('error', 'Unknown')}\n")
    
    print(f"Processing log saved to: {log_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='ISSIA - Batch process multiple flight lines',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing ATCOR output files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--lut-dir', type=str, default='luts',
                       help='Directory containing lookup tables')
    parser.add_argument('--wvl-path', type=str, default='wvl.npy',
                       help='Path to wavelength file')
    parser.add_argument('--flight-lines', type=str, nargs='+', default=None,
                       help='Specific flight line IDs to process')
    parser.add_argument('--flight-list', type=str, default=None,
                       help='Text file with flight line IDs (one per line)')
    parser.add_argument('--diagnostics', action='store_true',
                       help='Save diagnostic outputs')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue processing if a flight line fails')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker threads for parallel processing')
    parser.add_argument('--mosaic', action='store_true',
                       help='After processing, mosaic all flight lines per product (gs, albedo, rf)')
    parser.add_argument('--mosaic-dir', type=str, default=None,
                       help='Output directory for mosaics (default: <output-dir>/mosaics)')
    parser.add_argument('--edge-setback', type=int, default=50,
                       help='Pixels to trim from swath edges before mosaic weighting')

    args = parser.parse_args()

    # Determine flight lines to process
    flight_lines = None

    if args.flight_lines:
        flight_lines = args.flight_lines
    elif args.flight_list:
        with open(args.flight_list, 'r') as f:
            flight_lines = [line.strip() for line in f if line.strip()]

    results = batch_process(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        lut_dir=Path(args.lut_dir),
        wvl_path=Path(args.wvl_path),
        flight_lines=flight_lines,
        save_diagnostics=args.diagnostics,
        continue_on_error=args.continue_on_error,
        n_workers=args.workers
    )

    if args.mosaic and results and (results['successful'] or results['skipped']):
        from mosaic import mosaic_batch
        mosaic_dir = Path(args.mosaic_dir) if args.mosaic_dir else Path(args.output_dir) / 'mosaics'
        n_ready = len(results['successful']) + len(results['skipped'])
        print(f"\nMosaicking {n_ready} flight lines...")
        all_lines = results['successful'] + results['skipped']
        mosaic_batch(Path(args.output_dir), mosaic_dir, edge_setback=args.edge_setback,
                     flight_lines=all_lines)
        print(f"Mosaics saved to: {mosaic_dir}")


if __name__ == "__main__":
    main()
