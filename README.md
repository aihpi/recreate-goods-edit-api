# Qwen Image Edit API

OpenAI-compatible API for image editing using Qwen-Image-Edit model.

## Features

- OpenAI-compatible endpoints for seamless integration with OpenWebUI/OpenRouter
- GPU-accelerated image editing using Qwen-Image-Edit
- Simple REST API with multipart form data support
- Docker support with CUDA

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.10+
- 16GB+ GPU memory recommended

## Installation

### Local Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers
```

2. Run the server:

```bash
python run.py
```

The API will be available at `http://localhost:8000`

**Note**: On first run, the model (~15GB) will be downloaded from Hugging Face.

### Docker Setup

1. Build and run with Docker Compose:

```bash
docker-compose up -d
```

The API will be accessible at `http://localhost:200`

**Note**: On first run, the model (~15GB) will be downloaded to `./model-cache/` directory. Subsequent restarts will use the cached model.

### Kubernetes Deployment

#### Option 1: Use Pre-built Image from GitHub

The image is automatically built and published to GitHub Container Registry when code is pushed.

1. Deploy directly to Kubernetes:

```bash
kubectl apply -f k8s-deployment.yaml
```

#### Option 2: Build and Push Manually

1. Login to GitHub Container Registry:

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

2. Build and push the image:

```bash
docker build -t ghcr.io/aihpi/recreate-goods-edit-api:latest .
docker push ghcr.io/aihpi/recreate-goods-edit-api:latest
```

3. Deploy to Kubernetes:

```bash
kubectl apply -f k8s-deployment.yaml
```

The service will be accessible on port 200 within the cluster (ClusterIP). Use kubectl port-forward or an Ingress for external access.

**Note**: The init container will download the model to a persistent volume on first deployment. The model is shared across pod restarts.

## API Endpoints

### Edit Image

`POST /v1/images/edits`

Send a multipart form with:

- `image`: Image file to edit
- `prompt`: Edit instruction (e.g., "Change the background to sunset")
- `model`: (optional) Model ID, defaults to "qwen-image-edit"

Returns base64-encoded edited image in OpenAI format.

### List Models

`GET /v1/models`

Returns available models.

### Health Check

`GET /v1/health`

Returns server status and model loading state.

## Usage with OpenWebUI

1. Start the API server
2. In OpenWebUI settings, add custom OpenAI API endpoint:
   - URL: `http://localhost:200/v1` (Docker) or `http://localhost:8000/v1` (Local)
   - Model: `qwen-image-edit`

## Usage with curl

```bash
# Docker/Kubernetes (port 200)
curl -X POST "http://localhost:200/v1/images/edits" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@input.jpg" \
  -F "prompt=Make the sky purple" \
  -F "model=qwen-image-edit"
```

## Environment Variables

Create a `.env` file if you want to change the default config (see `.env.example`):

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `DEVICE`: cuda or cpu (auto-detected)

## Model Information

Qwen-Image-Edit supports:

- Semantic image editing
- Text rendering and modification
- Style transfer
- Object manipulation
- Background changes

## Troubleshooting

### Out of Memory

Reduce image size or ensure you have 16GB+ GPU memory.

### Model Loading Issues

Ensure you have latest diffusers:

```bash
pip install git+https://github.com/huggingface/diffusers
```

### CUDA Not Available

Check NVIDIA drivers and PyTorch CUDA installation:

```python
import torch
print(torch.cuda.is_available())
```
