# Stage 1: Download the model
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS model-downloader

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and necessary packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install just what's needed to download the model
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    diffusers \
    transformers \
    accelerate

# Install latest diffusers from git
RUN pip3 install --no-cache-dir git+https://github.com/huggingface/diffusers

# Download the model during build
RUN python3 -c "from diffusers import DiffusionPipeline; \
    import torch; \
    print('Downloading Qwen-Image-Edit model...'); \
    pipeline = DiffusionPipeline.from_pretrained( \
        'Qwen/Qwen-Image-Edit', \
        torch_dtype=torch.bfloat16); \
    pipeline.save_pretrained('/model'); \
    print('Model saved to /model')"

# Stage 2: Final application image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the downloaded model from stage 1
COPY --from=model-downloader /model /app/model

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install latest diffusers from git (as recommended by Qwen)
RUN pip3 install --no-cache-dir git+https://github.com/huggingface/diffusers

# Copy application code
COPY app app/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]