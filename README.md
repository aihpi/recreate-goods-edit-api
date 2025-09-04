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

### Docker Setup (Recommended)

The Docker image includes the model pre-downloaded for instant startup:

1. Build and run with Docker Compose:

```bash
docker-compose up -d
```

**Note**: The Docker build will take time initially (~20-30 minutes) as it downloads the model during build. The final image will be ~20GB but starts instantly without needing to download the model.

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
   - URL: `http://localhost:8000/v1`
   - Model: `qwen-image-edit`

## Usage with curl

```bash
curl -X POST "http://localhost:8000/v1/images/edits" \
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
