#!/usr/bin/env bash
# Process ISSIA flight lines from local scratch to avoid network I/O bottleneck.
# Copies required files to /tmp, runs batch, copies results back, then cleans up.
#
# Usage:
#   ./run_local_batch.sh \
#       --data-dir  /mnt/aco-uvic/.../OUTPUT/clipped \
#       --output-dir /mnt/aco-uvic/BrianM_Temp/test_hyper \
#       --lut-dir   /mnt/aco-uvic/BrianM_Temp/ISSIA_python/luts \
#       [--wvl-path wvl.npy] \
#       [--chunk-rows 256]

set -euo pipefail

# ---------- defaults ----------
DATA_DIR=""
OUTPUT_DIR=""
LUT_DIR=""
WVL_PATH="$(cd "$(dirname "$0")" && pwd)/wvl.npy"
CHUNK_ROWS=256
ISSIA_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRATCH="/tmp/issia_$$"

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)   DATA_DIR="$2";   shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --lut-dir)    LUT_DIR="$2";    shift 2 ;;
        --wvl-path)   WVL_PATH="$2";   shift 2 ;;
        --chunk-rows) CHUNK_ROWS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "$DATA_DIR"   ]] && { echo "Error: --data-dir is required";   exit 1; }
[[ -z "$OUTPUT_DIR" ]] && { echo "Error: --output-dir is required";  exit 1; }
[[ -z "$LUT_DIR"    ]] && { echo "Error: --lut-dir is required";     exit 1; }

SCRATCH_DATA="${SCRATCH}/input"
SCRATCH_LUTS="${SCRATCH}/luts"
SCRATCH_OUT="${SCRATCH}/output"

# ---------- cleanup on exit ----------
trap 'echo ""; echo "Cleaning up ${SCRATCH}..."; rm -rf "${SCRATCH}"; echo "Scratch removed."' EXIT

mkdir -p "${SCRATCH_DATA}" "${SCRATCH_OUT}"

# ---------- discover flight lines ----------
echo "========================================================================"
echo "Discovering flight lines in ${DATA_DIR}..."
REQUIRED_EXTS=("_atm.dat" "_eglo.dat" "_slp.dat" "_asp.dat" ".inn")
FLIGHT_LINES=()

for atm in "${DATA_DIR}"/*_atm.dat; do
    [[ -f "$atm" ]] || continue
    base="$(basename "$atm" _atm.dat)"
    complete=true
    for ext in "${REQUIRED_EXTS[@]}"; do
        [[ -f "${DATA_DIR}/${base}${ext}" ]] || { complete=false; break; }
    done
    $complete && FLIGHT_LINES+=("$base")
done

N=${#FLIGHT_LINES[@]}
[[ $N -eq 0 ]] && { echo "Error: no complete flight lines found in ${DATA_DIR}"; exit 1; }
echo "Found ${N} flight line(s) to process."
echo "========================================================================"

# ---------- copy LUTs ----------
echo ""
echo "[1/4] Copying LUTs to local scratch..."
rsync -a --info=progress2 "${LUT_DIR}/" "${SCRATCH_LUTS}/"
echo "      LUTs ready."

# ---------- copy flight line input files ----------
echo ""
echo "[2/4] Copying flight line data to local scratch..."
OPTIONAL_EXTS=("_atm.hdr" "_eglo.hdr" "_slp.hdr" "_asp.hdr")

i=0
for fl in "${FLIGHT_LINES[@]}"; do
    i=$((i + 1))
    printf "  [%d/%d] %s\n" "$i" "$N" "$fl"
    for ext in "${REQUIRED_EXTS[@]}" "${OPTIONAL_EXTS[@]}"; do
        src="${DATA_DIR}/${fl}${ext}"
        [[ -f "$src" ]] && rsync -a --info=progress2 "$src" "${SCRATCH_DATA}/" || true
    done
done
echo "      Data copy complete."

# ---------- run batch one flight line at a time, copy after each ----------
echo ""
echo "[3/4] Running ISSIA batch processing from local scratch..."
mkdir -p "${OUTPUT_DIR}"

i=0
for fl in "${FLIGHT_LINES[@]}"; do
    i=$((i + 1))
    echo "========================================================================"
    echo "# Flight line ${i}/${N}: ${fl}"
    echo "========================================================================"
    # Skip if all 3 products already exist in the destination
    if [[ -f "${OUTPUT_DIR}/${fl}_gs.tif" && -f "${OUTPUT_DIR}/${fl}_albedo.tif" && -f "${OUTPUT_DIR}/${fl}_rf.tif" ]]; then
        echo "  -> Already in ${OUTPUT_DIR}, skipping."
        continue
    fi
    NUMBA_THREADING_LAYER=omp NUMBA_NUM_THREADS=$(nproc) \
        python "${ISSIA_DIR}/run_issia_batch.py" \
            --data-dir     "${SCRATCH_DATA}" \
            --output-dir   "${SCRATCH_OUT}" \
            --lut-dir      "${SCRATCH_LUTS}" \
            --wvl-path     "${WVL_PATH}" \
            --chunk-rows   "${CHUNK_ROWS}" \
            --flight-lines "${fl}" \
            --continue-on-error

    echo ""
    echo "  -> Copying ${fl} results to ${OUTPUT_DIR}..."
    cp "${SCRATCH_OUT}/${fl}"_*.tif "${OUTPUT_DIR}/"
    echo "  -> Done. Results available in ${OUTPUT_DIR}/"
done

# ---------- mosaic after all flight lines complete ----------
echo ""
echo "[4/4] Building mosaics..."
python "${ISSIA_DIR}/mosaic_issia.py" --input-dir "${OUTPUT_DIR}"
cp "${SCRATCH_OUT}/mosaics"/*.tif "${OUTPUT_DIR}/mosaics/" 2>/dev/null || true
echo "      Done. Results in ${OUTPUT_DIR}/"
# trap handles cleanup
