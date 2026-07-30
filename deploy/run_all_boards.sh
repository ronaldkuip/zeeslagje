#!/usr/bin/env bash
# Runs solve_boards_file against deel_0002.csv .. deel_0346.csv, one at a
# time (sequential — a shared low-resource VPS is exactly why cargo build
# --release couldn't run there either, so this deliberately doesn't fan
# out across files, only within one via the existing thread argument).
#
# Usage (edit the variables below first, or override via env vars):
#   ./run_all_boards.sh
# Recommended: run it under nohup/tmux/screen so it survives an SSH
# disconnect over a run this long, e.g.:
#   nohup ./run_all_boards.sh > run_all_boards.log 2>&1 &
#   disown

set -uo pipefail

# --- Configuration — adjust these for your VPS layout -----------------
DATA_DIR="${DATA_DIR:-$HOME}"            # where deel_NNNN.csv files live
OUT_DIR="${OUT_DIR:-$HOME/results}"      # where per-file results/error logs go
BINARY="${BINARY:-$HOME/solve_boards_file}"
THREADS="${THREADS:-$(nproc)}"
START="${START:-2}"
END="${END:-346}"
# ------------------------------------------------------------------------

mkdir -p "$OUT_DIR"

if [[ ! -x "$BINARY" ]]; then
    echo "error: $BINARY not found or not executable (chmod +x it first)" >&2
    exit 1
fi

echo "Running deel_$(printf '%04d' "$START") .. deel_$(printf '%04d' "$END") with $THREADS threads"
echo "Input dir: $DATA_DIR | Output dir: $OUT_DIR"
echo

for n in $(seq -f "%04g" "$START" "$END"); do
    input="$DATA_DIR/deel_${n}.csv"
    results="$OUT_DIR/deel_${n}.results.txt"
    errors="$OUT_DIR/deel_${n}.errors.log"

    if [[ ! -f "$input" ]]; then
        echo "[$(date '+%H:%M:%S')] deel_${n}: SKIPPED (not found: $input)"
        continue
    fi

    echo "[$(date '+%H:%M:%S')] deel_${n}: starting..."
    start_ts=$(date +%s)
    if "$BINARY" "$input" "$THREADS" "$errors" > "$results" 2>>"$OUT_DIR/deel_${n}.stderr.log"; then
        elapsed=$(( $(date +%s) - start_ts ))
        echo "[$(date '+%H:%M:%S')] deel_${n}: done in ${elapsed}s -> $results"
    else
        echo "[$(date '+%H:%M:%S')] deel_${n}: FAILED (exit $?) -- see $OUT_DIR/deel_${n}.stderr.log" >&2
    fi
done

echo
echo "All done. Per-file results/errors in $OUT_DIR/"
