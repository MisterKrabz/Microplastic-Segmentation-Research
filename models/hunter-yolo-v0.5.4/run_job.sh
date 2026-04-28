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

# --- FIXED: PREVENT CONDOR HOLD ON EARLY EVICTION ---
touch final_model.pt checkpoint.pt 

echo "🚀 JOB START"
date

# --- THE BULLETPROOF TERMINATION TRAP ---
on_term () {
  echo "⚠️ Received termination/eviction signal. Attempting safe checkpoint save..."
  
  LAST_P2=$(find "$PWD/yolo_results/phase2_cooldown" -path "*/weights/last.pt" -print -quit 2>/dev/null || true)
  LAST_P1=$(find "$PWD/yolo_results/phase1_mosaic" -path "*/weights/last.pt" -print -quit 2>/dev/null || true)
  
  if [ -s "${LAST_P2}" ]; then
    cp -f "${LAST_P2}" checkpoint.pt
    echo "Saved Phase 2 checkpoint."
  elif [ -s "${LAST_P1}" ]; then
    cp -f "${LAST_P1}" checkpoint.pt
    echo "Saved Phase 1 checkpoint."
  fi
  exit 143 
}
trap on_term TERM INT QUIT

# ==============================================================
# STAGING RECOVERY 
# ==============================================================
echo "🔄 Checking for staged checkpoints from previous evictions..."
if [ -s "/staging/pwang384/checkpoint_${CLUSTER}.pt" ]; then
    cp "/staging/pwang384/checkpoint_${CLUSTER}.pt" ./checkpoint.pt
    echo "✅ Successfully recovered checkpoint from staging!"
else
    echo "ℹ️ No staging checkpoint found (normal for a brand new job)."
fi
# ==============================================================

echo "📦 Extracting datasets..."
rm -rf data_root
mkdir -p data_root

for zip_file in *.zip; do
  if [ -s "$zip_file" ]; then
    dir_name=$(basename "$zip_file" .zip)
    mkdir -p "data_root/$dir_name"
    unzip -o -q "$zip_file" -d "data_root/$dir_name"
  fi
done

find data_root -name "__MACOSX" -type d -exec rm -rf {} +

echo "🛠️ Installing offline augmentation dependencies..."
pip3 install opencv-python-headless shapely numpy

echo "🧬 Running strict 0-overlap copy-paste augmentation..."
python3 -u augment_data.py

echo "🏋️ Start training..."
python3 -u train_yolo.py

echo "📦 Collect outputs..."
BEST_P2="$PWD/yolo_results/phase2_cooldown/weights/best.pt"
BEST_P1="$PWD/yolo_results/phase1_mosaic/weights/best.pt"
LAST_P2="$PWD/yolo_results/phase2_cooldown/weights/last.pt"
LAST_P1="$PWD/yolo_results/phase1_mosaic/weights/last.pt"

if [ -s "${BEST_P2}" ]; then 
    cp -f "${BEST_P2}" final_model.pt
    echo "Grabbed Phase 2 Real-World Model."
elif [ -s "${BEST_P1}" ]; then 
    cp -f "${BEST_P1}" final_model.pt
    echo "Grabbed Phase 1 Mosaic Model (Phase 2 did not complete)."
else 
    echo "❌ CRITICAL: No valid best.pt found!"
    exit 3
fi

if [ -s "${LAST_P2}" ]; then 
    cp -f "${LAST_P2}" checkpoint.pt
elif [ -s "${LAST_P1}" ]; then 
    cp -f "${LAST_P1}" checkpoint.pt
fi

echo "✅ JOB END"