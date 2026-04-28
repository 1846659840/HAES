# HAES Reproducible Docker Image
# Pinned image hash for reproducibility
# Docker image: haes:latest
# SHA256: will be generated at build time

FROM nvcr.io/nvidia/pytorch:23.12-py3

LABEL maintainer="HAES Authors"
LABEL description="Hierarchical Adaptive Expert System for Weakly-Supervised Incremental Violence Detection"
LABEL version="1.0.0"
LABEL gpu="Tesla A100 40GB"
LABEL cuda="12.1"
LABEL driver="535.104.05"

# Set working directory
WORKDIR /workspace/HAES

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables for reproducibility
ENV PYTHONHASHSEED=42
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV CUDA_VISIBLE_DEVICES=0

# Default command
ENTRYPOINT ["python", "main_train.py"]
CMD ["--dataset", "xd_violence", "--config", "configs/default.yaml"]
