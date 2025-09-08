import hashlib
import os
from functools import lru_cache
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Prompta"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    JSON_LOGS: bool = True

    # Optional allowlist; empty disables auth
    AUTH_BEARER_TOKENS: List[str] = []

    # CORS origins; set specific origins in prod
    CORS_ORIGINS: List[str] = ["*"]

    # Inference/model configuration
    MODEL_ID: str = "openai/gpt-oss-20b"
    DEVICE_MAP: str = "auto"  # "cuda" | "cpu" | "auto"
    TORCH_DTYPE: str = "auto"  # "auto" | "float16" | "bfloat16" | "float32"
    HF_HOME: Optional[str] = None
    HF_TOKEN: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=(".env",), env_prefix="PROMPTA_", case_sensitive=False
    )

    @field_validator("AUTH_BEARER_TOKENS", mode="before")
    @classmethod
    def _parse_csv_tokens(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, str):
            # Split comma-separated tokens, strip whitespace, drop empties
            return [t.strip() for t in v.split(",") if t.strip()]
        return v

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_csv_origins(cls, v):
        if v is None or v == "":
            return ["*"]
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def _normalize_log_level(cls, v):
        if isinstance(v, str):
            return v.upper()
        return v


def apply_environment(settings: Settings) -> None:
    """Apply environment variables side-effects based on settings.

    - Set HF_HOME/HF_TOKEN so that downstream libraries respect project-local cache and auth.
    """
    if settings.HF_HOME:
        os.environ.setdefault("HF_HOME", settings.HF_HOME)
    if settings.HF_TOKEN:
        os.environ.setdefault("HF_TOKEN", settings.HF_TOKEN)


def _lib_versions() -> dict:
    vers = {}
    try:
        import torch  # type: ignore

        vers["torch"] = getattr(torch, "__version__", "na")
    except Exception:
        vers["torch"] = "na"
    try:
        import transformers  # type: ignore

        vers["transformers"] = getattr(transformers, "__version__", "na")
    except Exception:
        vers["transformers"] = "na"
    try:
        import accelerate  # type: ignore

        vers["accelerate"] = getattr(accelerate, "__version__", "na")
    except Exception:
        vers["accelerate"] = "na"
    try:
        import numpy as np  # type: ignore

        vers["numpy"] = getattr(np, "__version__", "na")
    except Exception:
        vers["numpy"] = "na"
    return vers


@lru_cache
def system_fingerprint(prefix: str = "fp_", length: int = 12) -> str:
    """Deterministic fingerprint derived from model id and key libs.

    The value is stable for a given combination of: MODEL_ID, DEVICE_MAP,
    TORCH_DTYPE, and versions of torch/transformers/accelerate/numpy.
    """
    s = get_settings()
    parts = {
        "model": s.MODEL_ID,
        "device": s.DEVICE_MAP,
        "dtype": s.TORCH_DTYPE,
        **_lib_versions(),
    }
    blob = "|".join(f"{k}={v}" for k, v in sorted(parts.items()))
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return f"{prefix}{h[:length]}"

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
