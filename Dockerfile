# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHON_VERSION=3.10

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory in container
WORKDIR ./dense_direction

# Copy requirements
COPY requirements.txt .

# Install PyTorch with CUDA support and other dependencies
RUN pip3 install --no-cache-dir torch==2.1.2 torchvision==0.16.2 xformers==0.0.23.post1 \
    --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install openmim==0.3.9
RUN mim install mmengine==0.10.3 mmcv==2.1.0 mmdet==3.3.0 mmsegmentation==1.2.2
RUN pip3 install --no-cache-dir -r requirements.txt