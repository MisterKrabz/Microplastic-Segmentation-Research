#!/bin/bash
set -e
echo "--- Debug: Unpacking ---"
mkdir sam_train_env
tar -xzf sam_train_env.tar.gz -C sam_train_env

echo "--- Debug: Checking for conda-unpack ---"
ls -l sam_train_env/bin/conda-unpack

echo "--- Debug: Content of conda-unpack ---"
# This will show us if the script is what we expect
head -n 20 sam_train_env/bin/conda-unpack

echo "--- Debug: Python binary details ---"
ls -l sam_train_env/bin/python
# Check if python is a symlink or binary
file sam_train_env/bin/python

echo "--- Debug: Attempting to run conda-unpack ---"
# We try running it and capture any output
./sam_train_env/bin/conda-unpack || echo "conda-unpack FAILED"

echo "--- Debug: Python Paths AFTER unpack ---"
# We ask python where it thinks it is
source sam_train_env/bin/activate
python -c "import sys; print(sys.path)" || echo "Python Crash"
