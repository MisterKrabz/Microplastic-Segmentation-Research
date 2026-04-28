#!/bin/bash
set -euo pipefail

export USER=condor
export LOGNAME=condor
export HOME="$PWD"
export XDG_CACHE_HOME="$PWD/.cache"
export TORCH_HOME="$PWD/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/torchinductor"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export ULTRALYTICS_SETTINGS_DIR="$PWD/.ultralytics"
export YOLO_CONFIG_DIR="$PWD/.ultralytics"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" "$ULTRALYTICS_SETTINGS_DIR"

echo "🚀 JOB START"
date

: > checkpoint.pt
: > final_model.pt

on_term () {
  echo "⚠️ Received termination signal. Attempting checkpoint save..."
  LAST=$(find "$PWD/yolo_results" -path "*/weights/last.pt" -print -quit 2>/dev/null || true)
  if [ -n "${LAST}" ] && [ -f "${LAST}" ]; then
    cp -f "${LAST}" checkpoint.pt || true
  fi
  exit 0
}
trap on_term TERM INT QUIT

echo "📦 Extracting datasets..."
rm -rf data_root
mkdir -p data_root

for zip_file in *.zip; do
  if [ -f "$zip_file" ]; then
    dir_name=$(basename "$zip_file" .zip)
    mkdir -p "data_root/$dir_name"
    unzip -o -q "$zip_file" -d "data_root/$dir_name"
  fi
done

# --- NEW: Install dependencies and run offline augmentation ---
echo "🛠️ Installing offline augmentation dependencies..."
pip3 install --user opencv-python-headless shapely numpy

echo "🧬 Running strict 0-overlap copy-paste augmentation..."
python3 -u augment_data.py
# ------------------------------------------------------------

echo "🏋️ Start training..."
python3 -u train_yolo.py

echo "📦 Collect outputs..."
BEST=$(find "$PWD/yolo_results" -path "*/weights/best.pt" -print -quit 2>/dev/null || true)
LAST=$(find "$PWD/yolo_results" -path "*/weights/last.pt" -print -quit 2>/dev/null || true)

if [ -n "${LAST}" ] && [ -f "${LAST}" ]; then cp -f "${LAST}" checkpoint.pt; fi
if [ -n "${BEST}" ] && [ -f "${BEST}" ]; then cp -f "${BEST}" final_model.pt; else exit 3; fi

echo "✅ JOB END"