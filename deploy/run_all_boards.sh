#!/usr/bin/env bash
# Runs solve_boards_file against deel_NNNN.csv files (default deel_0001..
# deel_0346), one at a time (sequential — a shared low-resource VPS is
# exactly why cargo build --release couldn't run there either, so this
# deliberately doesn't fan out across files, only within one via the
# existing thread argument).
#
# Per-file output (results.txt/errors.log/stderr.log in OUT_DIR) lands as
# soon as that one file finishes, so you can inspect progress well before
# the whole run is done — no need to wait for every file. Additionally:
#   - Every board resolved in SPECIAL_SALVO_COUNTS salvos (see
#     solve_boards_file.rs) gets its full board layout appended, as JSON,
#     to SPECIAL_DIR/salvos_NN.jsonl — accumulated across every file in
#     this run, not just the current one (solve_boards_file opens these in
#     append mode for exactly this reason).
#   - Once every file is done, OUT_DIR/combined_summary.txt aggregates the
#     per-file totals (processed/invalid/unresolved + a merged salvo
#     histogram) into one whole-run picture, plus a line count per special
#     case file.
#
# Usage (edit the variables below first, or override via env vars):
#   ./run_all_boards.sh
# Recommended: run it under nohup/tmux/screen so it survives an SSH
# disconnect over a run this long, e.g.:
#   nohup ./run_all_boards.sh > run_all_boards.log 2>&1 &
#   disown

set -uo pipefail

# --- Configuration — adjust these for your VPS layout -----------------
DATA_DIR="${DATA_DIR:-$HOME}"                       # where deel_NNNN.csv files live
OUT_DIR="${OUT_DIR:-$DATA_DIR}"                     # where per-file results/error logs go — same dir as the input by default
SPECIAL_DIR="${SPECIAL_DIR:-$OUT_DIR/special_cases}" # where the 6 salvos_NN.jsonl files accumulate
BINARY="${BINARY:-$HOME/solve_boards_file}"
THREADS="${THREADS:-$(nproc)}"
START="${START:-1}"
END="${END:-346}"
# ------------------------------------------------------------------------

mkdir -p "$OUT_DIR" "$SPECIAL_DIR"

if [[ ! -x "$BINARY" ]]; then
    echo "error: $BINARY not found or not executable (chmod +x it first)" >&2
    exit 1
fi

echo "Running deel_$(printf '%04d' "$START") .. deel_$(printf '%04d' "$END") with $THREADS threads"
echo "Input dir: $DATA_DIR | Output dir: $OUT_DIR | Special cases: $SPECIAL_DIR"
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
    if "$BINARY" "$input" "$THREADS" "$errors" "$SPECIAL_DIR" > "$results" 2>>"$OUT_DIR/deel_${n}.stderr.log"; then
        elapsed=$(( $(date +%s) - start_ts ))
        echo "[$(date '+%H:%M:%S')] deel_${n}: done in ${elapsed}s -> $results"
    else
        echo "[$(date '+%H:%M:%S')] deel_${n}: FAILED (exit $?) -- see $OUT_DIR/deel_${n}.stderr.log" >&2
    fi
done

echo
echo "All done. Per-file results/errors in $OUT_DIR/. Building combined summary..."

# --- Combined summary across every deel_NNNN.results.txt produced this run ---
# Deliberately plain bash/sed/awk (no gawk-only asorti/match(...,arr), no
# grep -P) — this VPS's /bin/awk may well be mawk, not gawk.
combined="$OUT_DIR/combined_summary.txt"
{
    echo "Combined summary — deel_$(printf '%04d' "$START")..deel_$(printf '%04d' "$END")"
    echo "Generated: $(date)"
} > "$combined"

total_processed=0
total_invalid=0
total_unresolved=0
declare -A combined_hist

shopt -s nullglob
for results_file in "$OUT_DIR"/deel_*.results.txt; do
    p=$(sed -n 's/^Processed \([0-9]*\) boards.*/\1/p' "$results_file")
    i=$(sed -n 's/^Invalid boards: *\([0-9]*\).*/\1/p' "$results_file")
    u=$(sed -n 's/.*cap): *\([0-9]*\)/\1/p' "$results_file")
    total_processed=$(( total_processed + ${p:-0} ))
    total_invalid=$(( total_invalid + ${i:-0} ))
    total_unresolved=$(( total_unresolved + ${u:-0} ))
    while IFS=, read -r salvos count; do
        [[ "$salvos" =~ ^[0-9]+$ ]] || continue
        combined_hist[$salvos]=$(( ${combined_hist[$salvos]:-0} + count ))
    done < <(awk '/^salvos,count$/{flag=1;next} flag && NF' "$results_file")
done
shopt -u nullglob

{
    echo
    echo "Total boards processed: $total_processed"
    echo "Total invalid:          $total_invalid"
    echo "Total unresolved:       $total_unresolved"
    echo
    echo "salvos,count"
    for k in "${!combined_hist[@]}"; do echo "$k,${combined_hist[$k]}"; done | sort -t, -k1,1n
    echo
    echo "Special-case file line counts ($SPECIAL_DIR):"
    wc -l "$SPECIAL_DIR"/salvos_*.jsonl 2>/dev/null
} >> "$combined"

echo "Combined summary written to $combined"
