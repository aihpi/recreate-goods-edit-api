from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging
from app.api.endpoints import router
from app.services.model_service import model_service
from app.core.config import settings
from app.core.exceptions import http_exception_handler, general_exception_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    logger.info("Starting up...")
    model_service.load_model()
    yield
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Qwen Image Edit API",
    description="OpenAI-compatible API for image editing using Qwen-Image-Edit",
    version="1.0.0",
    lifespan=lifespan
)

# Include API routes
app.include_router(router, prefix="/v1")

# Register exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "Qwen Image Edit API",
        "version": "1.0.0",
        "endpoints": [
            "/v1/images/edits",
            "/v1/models",
            "/v1/health"
        ]
    }