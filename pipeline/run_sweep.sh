#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

# Edit these arrays to change the 10 x 10 parameter grid.
EPOCHS=(100 200 300 500 750 1000 1500 2000 3000 5000)
LEARNING_RATES=(0.005 0.01 0.02 0.03 0.04 0.06 0.08 0.10 0.15 1.0)

if [[ "${1:-}" == "--worker" ]]; then
  if [[ $# -ne 2 ]]; then
    echo "Internal usage: $0 --worker SWEEP_DIR" >&2
    exit 2
  fi
  SWEEP_DIR="$2"
else
  STAMP="$(date +%Y%m%d-%H%M%S-%N)"
  SWEEP_DIR="$OUTPUT_DIR/sweeps/$STAMP"
  mkdir -p "$SWEEP_DIR"
  nohup bash "$SCRIPT_DIR/run_sweep.sh" --worker "$SWEEP_DIR" \
    >"$SWEEP_DIR/sweep.log" 2>&1 < /dev/null &
  PID=$!
  printf '%s\n' "$PID" > "$SWEEP_DIR/sweep.pid"
  echo "Started 100-run sweep in the background (PID $PID)."
  echo "Progress log: $SWEEP_DIR/sweep.log"
  echo "Summary CSV: $SWEEP_DIR/summary.csv"
  echo "Stop it with: kill $PID"
  exit 0
fi

mkdir -p "$SWEEP_DIR"
SUMMARY_FILE="$SWEEP_DIR/summary.csv"
printf 'epochs,learning_rate,exit_status,log_file\n' > "$SUMMARY_FILE"

cd "$SCRIPT_DIR"
TOTAL=$(( ${#EPOCHS[@]} * ${#LEARNING_RATES[@]} ))
RUN=0
CURRENT_PID=""

stop_current_run() {
  if [[ -n "$CURRENT_PID" ]]; then
    kill "$CURRENT_PID" 2>/dev/null || true
  fi
  exit 143
}
trap stop_current_run TERM INT

for epoch_count in "${EPOCHS[@]}"; do
  for learning_rate in "${LEARNING_RATES[@]}"; do
    RUN=$((RUN + 1))
    LOG_FILE="$SWEEP_DIR/epochs-${epoch_count}_lr-${learning_rate}.log"
    echo "[$RUN/$TOTAL] epochs=$epoch_count learning_rate=$learning_rate"

    # A fresh process for every combination releases CUDA and Dr.Jit memory
    # when it exits, before the next combination starts.

    python -u main.py \
      --epochs "$epoch_count" \
      --learning-rate "$learning_rate" >"$LOG_FILE" 2>&1 &
    CURRENT_PID=$!
    if wait "$CURRENT_PID"; then
      STATUS=0
    else
      STATUS=$?
      echo "Run failed with exit status $STATUS; see $LOG_FILE" >&2
    fi
    CURRENT_PID=""

    printf '%s,%s,%s,%s\n' \
      "$epoch_count" "$learning_rate" "$STATUS" "$(basename -- "$LOG_FILE")" \
      >> "$SUMMARY_FILE"
  done
done

echo "Sweep complete. Summary: $SUMMARY_FILE"
