#!/bin/bash
# run_segmentation.sh

# Exit immediately if a command fails
set -e

# --- Define CHTC Paths ---
INPUT_DIR="/staging/pwang384/URS_Project/Data"
OUTPUT_DIR="/staging/pwang384/URS_Project/output_segmentations"
CHECKPOINT="/staging/pwang384/URS_Project/sam2.1_hiera_large.pt"

# --- Define Local Job Paths ---
# (These files were transferred by HTCondor)
ENV_PACK="sam2_env.tar.gz"
PYTHON_SCRIPT="segment_sam2.py"
CONFIG_FILE="sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

# --- Setup ---
# Create the output directory on the shared filesystem
mkdir -p $OUTPUT_DIR
echo "Created output directory: $OUTPUT_DIR"

# Unpack the Conda environment
tar -xzf sam2_env_linux.tar.gz
echo "Unpacked Conda environment"

# Activate the Conda environment
source sam2_env_linux/bin/activate
echo "Activated Conda environment"

# --- Run the Python Script ---
echo "Starting Python segmentation..."
python $PYTHON_SCRIPT \
    $INPUT_DIR \
    $OUTPUT_DIR \
    $CHECKPOINT \
    $CONFIG_FILE

echo "Python script finished."

# --- Cleanup ---
rm -rf sam2_env
echo "Job complete. Cleaned up environment."
