from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from PIL import Image
import io
import base64
from app.models.schemas import (
    ImageResponse,
    ImageData,
    ModelsResponse,
    ModelInfo,
)
from app.services.model_service import model_service
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/images/edits", response_model=ImageResponse)
async def create_image_edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model: str = Form(default="qwen-image-edit"),
):
    """
    Edit an image using Qwen-Image-Edit model.
    """
    # Validate model parameter for OpenAI compatibility
    if model and model != "qwen-image-edit":
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{model}' not found. Available models: qwen-image-edit"
        )
    
    try:
        # Read and validate image
        image_bytes = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid image format: {str(e)}"
            )

        # Perform the edit
        edited_image = model_service.edit_image(pil_image, prompt)

        # Convert to base64
        buffered = io.BytesIO()
        edited_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Return OpenAI-compatible response
        return ImageResponse(data=[ImageData(b64_json=img_base64)])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image edit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available models"""
    return ModelsResponse(data=[ModelInfo(id="qwen-image-edit")])


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_service.pipeline is not None,
        "device": settings.device,
    }
