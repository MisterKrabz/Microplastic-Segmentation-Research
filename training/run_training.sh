#!/bin/bash
set -e

# --- 1. Environment Setup ---
echo "Creating environment directory..."
mkdir sam_train_env

echo "Unpacking environment (this takes 10-15 min)..."
tar -xzf sam_train_env.tar.gz -C sam_train_env

echo "Fixing paths with conda-unpack..."
# FIX: We use the SYSTEM python to run the unpack script. 
# This avoids the "encodings" crash because we aren't using the broken env python yet.
/usr/bin/python3 sam_train_env/bin/conda-unpack

echo "Activating environment..."
source sam_train_env/bin/activate

# --- 2. Verification (Predict Future Errors) ---
echo "Verifying Python..."
# If this fails, we know immediately, rather than waiting for the script to crash
python -c "import sys; print('Python is working. Path:', sys.prefix)"

echo "--- 3. Configuration ---"
# We copy the config to the correct internal location so hydra finds it
mkdir -p sam2/sam2/configs
cp my_train_config.yaml sam2/sam2/configs/

echo "--- 4. Training ---"
# We use the config that points to the local scratch directory
python sam2/training/train.py --config configs/my_train_config.yaml

echo "--- 5. Saving Results ---"
# We zip the output so CHTC can transfer it back to you
echo "Zipping model outputs..."
tar -czf model_results.tar.gz sam2_model_output

echo "Job complete."
