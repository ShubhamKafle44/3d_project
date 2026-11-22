#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

# Edit these arrays to choose the sweep. The execution order is:
# detector -> property -> every epoch/learning-rate combination.
DETECTORS=(
  fasterrcnn_resnet50_fpn_v2
  yolo26n
  yolo26s
  yolo26m
  yolo26l
  yolo26x
  yoloe-26s-seg
)
PROPERTIES=(POSITION ROTATION POSE LIGHTING TEXTURE)
EPOCHS=(100 1000)
LEARNING_RATES=(0.001 0.01 0.05)

if [[ "${1:-}" != "--worker" ]]; then
  mkdir -p "$OUTPUT_DIR/sweeps"
  STAMP="$(date +%Y%m%d-%H%M%S)"
  SWEEP_DIR="$OUTPUT_DIR/sweeps/background-$STAMP"
  mkdir -p "$SWEEP_DIR"
  nohup bash "$0" --worker "$SWEEP_DIR" \
    >"$SWEEP_DIR/sweep.log" 2>&1 < /dev/null &
  PID=$!
  echo "$PID" > "$SWEEP_DIR/sweep.pid"
  RUNS=$((${#DETECTORS[@]} * ${#PROPERTIES[@]} * ${#EPOCHS[@]} * ${#LEARNING_RATES[@]}))
  echo "Started detector/property/epoch/rate sweep with $RUNS runs (PID $PID)."
  echo "Log: $SWEEP_DIR/sweep.log"
  echo "Summary: $SWEEP_DIR/summary.csv"
  echo "Follow progress: tail -f '$SWEEP_DIR/sweep.log'"
  echo "Stop it: kill $PID"
  exit 0
fi

SWEEP_DIR="${2:?worker mode requires a sweep directory}"
mkdir -p "$SWEEP_DIR"
SUMMARY_FILE="$SWEEP_DIR/summary.csv"
printf 'detector,property,epochs,learning_rate,exit_status,log_file\n' > "$SUMMARY_FILE"

cd "$SCRIPT_DIR"
TOTAL=$((${#DETECTORS[@]} * ${#PROPERTIES[@]} * ${#EPOCHS[@]} * ${#LEARNING_RATES[@]}))
RUN=0
CURRENT_PID=""

stop_current_run() {
  if [[ -n "$CURRENT_PID" ]]; then
    kill "$CURRENT_PID" 2>/dev/null || true
  fi
  exit 143
}
trap stop_current_run TERM INT

for detector in "${DETECTORS[@]}"; do
  for property in "${PROPERTIES[@]}"; do
    for epoch_count in "${EPOCHS[@]}"; do
      for learning_rate in "${LEARNING_RATES[@]}"; do
        RUN=$((RUN + 1))
        rate_label="${learning_rate//./p}"
        LOG_FILE="$SWEEP_DIR/${detector}_${property}_e${epoch_count}_lr${rate_label}.log"
        echo "[$RUN/$TOTAL] detector=$detector property=$property epochs=$epoch_count learning_rate=$learning_rate"

        python -u main.py \
          --detector "$detector" \
          --property "$property" \
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

        printf '%s,%s,%s,%s,%s,%s\n' \
          "$detector" "$property" "$epoch_count" "$learning_rate" \
          "$STATUS" "$(basename -- "$LOG_FILE")" >> "$SUMMARY_FILE"
      done
    done
  done
done

echo "Finished $TOTAL runs. Summary: $SUMMARY_FILE"
