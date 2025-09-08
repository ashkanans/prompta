from fastapi import APIRouter

from app.core.config import settings


router = APIRouter(tags=["health"])


@router.get("/health/live")
async def health_live():
    return {"status": "ok"}


@router.get("/health/ready")
async def health_ready():
    ready = True
    return {
        "status": "ready" if ready else "initializing",
        "model_id": settings.MODEL_ID,
    }
