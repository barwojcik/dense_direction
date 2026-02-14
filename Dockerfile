# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS runtime

# Set environment variables
ENV PYTHON_VERSION=3.10 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

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
WORKDIR /dense_direction

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA support and other dependencies
RUN pip3 install --no-cache-dir torch==2.1.2 torchvision==0.16.2 xformers==0.0.23.post1 \
    --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install openmim==0.3.9
RUN mim install mmengine==0.10.3 mmcv==2.1.0 mmdet==3.3.0 mmsegmentation==1.2.2

# Copy pyproject.toml and README (required for metadata)
COPY pyproject.toml README.md* ./

# Copy dense_direction
COPY src ./src/

# Install dense_direction
RUN pip3 install --no-cache-dir .

# Copy configs and tools
COPY configs ./configs
COPY tools ./tools

# Prepare data dir
RUN mkdir data

FROM runtime AS test
# Install test/dev dependencies into the same env
RUN pip3 install --no-cache-dir ".[dev]"

# Include tests in the image so pytest can run
COPY tests ./tests

# Default command for this stage (optional, but convenient)
CMD ["pytest", "-v"]
