# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

# Set environment variables for GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model during build
# Set environment variable for model cache
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

# Download the model using huggingface-cli
RUN mkdir -p /models && \
    pip install --no-cache-dir huggingface-hub && \
    huggingface-cli download openai/gpt-oss-20b --local-dir /models/openai/gpt-oss-20b --quiet && \
    echo "Model download complete"

# Copy application code
COPY app ./app

# Create directory for model downloads (kept for compatibility)
RUN mkdir -p /tmp/models

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]