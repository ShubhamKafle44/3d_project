#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS=10000
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  EPOCHS="$1"
  shift
fi
OUTPUT_DIR="$SCRIPT_DIR/outputs"

if [[ ! "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 [positive-epoch-count] [main.py options...]" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$OUTPUT_DIR/search-$STAMP.log"
PID_FILE="$OUTPUT_DIR/search-$STAMP.pid"

cd "$SCRIPT_DIR"
nohup python -u main.py --epochs "$EPOCHS" "$@" >"$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"

echo "Started pipeline search (up to $EPOCHS epochs, PID $PID)"
echo "Log: $LOG_FILE"
echo "Follow progress: tail -f '$LOG_FILE'"
echo "Stop it: kill $PID"
