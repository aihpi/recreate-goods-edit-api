import torch
from diffusers import DiffusionPipeline
from PIL import Image
import logging
import os
from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self):
        self.pipeline = None
        
    def load_model(self):
        """Load Qwen-Image-Edit model on startup"""
        try:
            # Check if local model exists (from Docker build)
            local_model_path = "/app/model"
            if os.path.exists(local_model_path):
                logger.info(f"Loading model from local path: {local_model_path}")
                model_path = local_model_path
            else:
                logger.info(f"Loading model from Hugging Face: {settings.model_name}")
                model_path = settings.model_name
            
            # Load the pipeline
            self.pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if settings.device == "cuda" else torch.float32
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(settings.device)
            
            # Disable progress bar for cleaner logs
            self.pipeline.set_progress_bar_config(disable=True)
            
            logger.info(f"Model loaded successfully on {settings.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def edit_image(self, image: Image.Image, prompt: str, seed: int = None) -> Image.Image:
        """Edit an image using the prompt"""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded")
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Set up generation parameters
        generator = torch.manual_seed(seed) if seed is not None else None
        
        # Run inference
        result = self.pipeline(
            image=image,
            prompt=prompt,
            true_cfg_scale=settings.true_cfg_scale,
            num_inference_steps=settings.num_inference_steps,
            negative_prompt=settings.negative_prompt,
            generator=generator
        )
        
        return result.images[0]


# Global instance
model_service = ModelService()