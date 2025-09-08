from typing import Optional

from app.core.config import settings


# Minimal readiness flag; set to True once the model is loaded.
_MODEL_READY: bool = False


def is_model_ready() -> bool:
    return _MODEL_READY


def mark_model_ready(value: bool = True) -> None:
    global _MODEL_READY
    _MODEL_READY = bool(value)


def current_model_id() -> str:
    return settings.MODEL_ID
