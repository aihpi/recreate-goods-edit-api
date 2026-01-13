import torch
from diffusers import DiffusionPipeline
from PIL import Image
import logging
import os
import time
from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self):
        self.pipeline = None
        
    def load_model(self):
        """Load Qwen-Image-Edit model on startup"""
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("Deterministic CUDA settings enabled")

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

            # Load LoRA if configured
            lora_path = os.environ.get("LORA_PATH")
            lora_weight_name = os.environ.get("LORA_WEIGHT_NAME")
            hf_token = os.environ.get("HF_TOKEN")
            max_lora_rank = int(os.environ.get("LORA_RANK", "16"))

            if lora_path:
                logger.info(f"Loading LoRA from: {lora_path}")
                self.pipeline.enable_lora_hotswap(target_rank=max_lora_rank)
                self.pipeline.load_lora_weights(
                    lora_path,
                    weight_name=lora_weight_name,
                    adapter_name="default",
                    token=hf_token
                )
                logger.info("LoRA loaded successfully")
            
            # Regional compilation - 7x faster cold start than full compile
            if settings.device == "cuda" and hasattr(self.pipeline, 'transformer'):
                logger.info("Compiling transformer (regional)...")
                self.pipeline.transformer.compile_repeated_blocks(fullgraph=True)
                logger.info("Compilation complete")
            logger.info(f"Model loaded successfully on {settings.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def edit_image(
        self,
        image: Image.Image,
        prompt: str,
        seed: int = None,
        true_cfg_scale: float = settings.true_cfg_scale,
        num_inference_steps: int = settings.num_inference_steps,
        negative_prompt: str = settings.negative_prompt,
    ) -> Image.Image:
        """Edit an image using the prompt"""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded")
        
        start_time = time.perf_counter()

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        logger.info(
            "edit_image start: size=%sx%s device=%s steps=%s",
            image.width,
            image.height,
            settings.device,
            num_inference_steps,
        )
        
        # Set up generation parameters
        generator = None
        if seed is not None:
            generator = torch.Generator(device=settings.device).manual_seed(seed)
        
        # Run inference
        inference_start = time.perf_counter()
        result = self.pipeline(
            image=image,
            prompt=prompt,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            generator=generator
        )
        inference_end = time.perf_counter()

        logger.info(
            "edit_image done: total=%.3fs inference=%.3fs",
            inference_end - start_time,
            inference_end - inference_start,
        )
        
        return result.images[0]


# Global instance
model_service = ModelService()
