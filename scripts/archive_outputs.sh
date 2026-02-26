#!/usr/bin/env bash
set -euo pipefail

# Non-destructive cleanup: move generated outputs into an archive folder.
# This does not delete anything; it just relocates folders/files.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_DIR="$ROOT_DIR/archive"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ARCHIVE_DIR/run_$TIMESTAMP"

mkdir -p "$RUN_DIR"

# List of output folders/files to archive if they exist
TARGETS=(
  "logs"
  "results"
  "saved_checkpoints"
  "training_logs"
  "videos"
  "MIXTAPE_Results"
  "results.zip"
  "__pycache__"
)

moved_any=false

for target in "${TARGETS[@]}"; do
  src="$ROOT_DIR/$target"
  if [[ -e "$src" ]]; then
    echo "Moving $src -> $RUN_DIR/"
    mv "$src" "$RUN_DIR/"
    moved_any=true
  else
    echo "Skipping missing: $src"
  fi
done

if [[ "$moved_any" == false ]]; then
  echo "No targets were found to move."
else
  echo "Archive complete: $RUN_DIR"
fi
