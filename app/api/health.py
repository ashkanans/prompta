from fastapi import APIRouter

from app.core.config import settings
from app.services.inference import is_model_ready


router = APIRouter(tags=["health"])


@router.get("/health/live")
async def health_live():
    return {"status": "ok"}


@router.get("/health/ready")
async def health_ready():
    ready = is_model_ready()
    return {
        "status": "ready" if ready else "initializing",
        "model_id": settings.MODEL_ID,
    }
