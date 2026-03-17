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

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$MPLCONFIGDIR" \
  "$ULTRALYTICS_SETTINGS_DIR"

echo "🚀 JOB START"
date
pwd
ls -lah

: > checkpoint.pt
: > final_model.pt

on_term () {
  echo "⚠️ Received termination signal. Attempting checkpoint save..."
  LAST=$(find "$PWD/yolo_results" -path "*/weights/last.pt" -print -quit 2>/dev/null || true)
  if [ -n "${LAST}" ] && [ -f "${LAST}" ]; then
    cp -f "${LAST}" checkpoint.pt || true
    echo "✅ Saved checkpoint.pt from ${LAST}"
  else
    echo "ℹ️ No last.pt found."
  fi
  exit 0
}
trap on_term TERM INT QUIT

echo "🔍 GPU smoke test..."
python3 - <<'PY'
import os, torch
print("USER =", os.environ.get("USER"))
print("HOME =", os.environ.get("HOME"))
print("PWD  =", os.getcwd())
print("CUDA Available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA NOT AVAILABLE")
x = torch.tensor([1.0], device="cuda")
print("Allocation successful:", x)
PY

echo "📦 Extracting datasets..."
rm -rf data_root
mkdir -p data_root

# Loop through all zip files and extract them into their own subdirectories
count=0
for zip_file in *.zip; do
  if [ -f "$zip_file" ]; then
    echo "Extracting $zip_file..."
    dir_name=$(basename "$zip_file" .zip)
    mkdir -p "data_root/$dir_name"
    unzip -q "$zip_file" -d "data_root/$dir_name"
    count=$((count+1))
  fi
done

if [ $count -eq 0 ]; then
  echo "❌ No zip files found in scratch."
  exit 2
fi

echo "✅ Extracted tree:"
find data_root -maxdepth 3 -type d | sort

echo "🔎 Sanity check dataset..."
python3 - <<'PY'
import os, glob

root = "data_root"
train_imgs = glob.glob(os.path.join(root, "**", "train", "images"), recursive=True)
yamls = glob.glob(os.path.join(root, "**", "*.yaml"), recursive=True)

print(f"Found {len(train_imgs)} train/images directories.")
print(f"Found {len(yamls)} yaml files.")

if not train_imgs:
    raise SystemExit("❌ Could not find any train/images directories")
if not yamls:
    raise SystemExit("❌ Could not find any yaml configuration files")
PY

echo "🏋️ Start training..."
python3 -u train_yolo.py

echo "📦 Collect outputs..."
BEST=$(find "$PWD/yolo_results" -path "*/weights/best.pt" -print -quit 2>/dev/null || true)
LAST=$(find "$PWD/yolo_results" -path "*/weights/last.pt" -print -quit 2>/dev/null || true)

if [ -n "${LAST}" ] && [ -f "${LAST}" ]; then
  cp -f "${LAST}" checkpoint.pt
  echo "✅ checkpoint.pt ready"
fi

if [ -n "${BEST}" ] && [ -f "${BEST}" ]; then
  cp -f "${BEST}" final_model.pt
  echo "✅ final_model.pt ready"
else
  echo "❌ best.pt not found"
  find "$PWD/yolo_results" -maxdepth 8 -type f | sort || true
  exit 3
fi

ls -lh checkpoint.pt final_model.pt
date
echo "✅ JOB END"