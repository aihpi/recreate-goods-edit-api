FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=/app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install latest diffusers from git (as recommended by Qwen)
RUN pip3 install --no-cache-dir git+https://github.com/huggingface/diffusers

# Copy application code and scripts
COPY app app/
COPY download_model.py /app/

# Create directory for model (will be mounted as volume or downloaded to)
RUN mkdir -p /app/model

# Expose port
EXPOSE 8000

# Run the application from root directory
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]