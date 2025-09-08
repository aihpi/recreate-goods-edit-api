#!/usr/bin/env python3
"""
Script to download Qwen-Image-Edit model to local directory.
Used by Kubernetes init containers.
"""
import os
import sys
from diffusers import DiffusionPipeline
import torch

def download_model():
    model_path = "/app/model"
    
    # Check if model already exists
    if os.path.exists(f"{model_path}/model_index.json"):
        print(f"Model already exists at {model_path}")
        sys.exit(0)
    
    print("Downloading Qwen-Image-Edit model...")
    try:
        pipeline = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit",
            torch_dtype=torch.bfloat16
        )
        pipeline.save_pretrained(model_path)
        print(f"Model successfully saved to {model_path}")
    except Exception as e:
        print(f"Failed to download model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()