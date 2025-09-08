from pydantic import BaseModel, Field
from typing import List
import time


class ImageData(BaseModel):
    b64_json: str


class ImageResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[ImageData]  # Will contain just 1 image


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "qwen"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]
