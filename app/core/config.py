from pydantic_settings import BaseSettings
import torch


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model
    model_name: str = "Qwen/Qwen-Image-Edit"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Qwen-Image-Edit required parameters
    true_cfg_scale: float = 4.0
    num_inference_steps: int = 50
    negative_prompt: str = " "
    
    class Config:
        env_file = ".env"


settings = Settings()