# Dockerfile
# Base image: PyTorch 2.5.1 with CUDA 12.4 (Matches your code requirements)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Prevent python buffering (makes logs show up instantly)
ENV PYTHONUNBUFFERED=1

# Install system dependencies (git for cloning, libgl1 for opencv)
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the SAM2 source code into the container
# We copy the whole training folder, which includes sam2/
COPY training/sam2 /app/segment-anything-2

# Install SAM2
# We use --no-build-isolation to avoid the numpy conflict
WORKDIR /app/segment-anything-2
RUN pip install --no-cache-dir --no-build-isolation -e ".[notebooks]"

# Create the output directory
RUN mkdir -p /app/sam2_model_output

# Copy your specific config into the correct location
COPY training/my_train_config.yaml /app/segment-anything-2/sam2/configs/

# Set the default command to start a shell (useful for testing)
CMD ["/bin/bash"]
