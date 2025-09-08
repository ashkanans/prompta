from time import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from app.core.config import settings, system_fingerprint


router = APIRouter()


@router.post("/v1/completions", tags=["completions"])
async def create_completion(request: Request, payload: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI-compatible legacy Completions endpoint (scaffold).

    This is a stub that returns a syntactically compatible response. Full
    inference, streaming, logprobs, and best_of behavior will be implemented
    in subsequent steps.
    """
    model = payload.get("model") or settings.MODEL_ID
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    created = int(time())
    resp = {
        "id": f"cmpl-stub-{created}",
        "object": "text_completion",
        "created": created,
        "model": model,
        "system_fingerprint": system_fingerprint(),
        "choices": [
            {
                "text": "",  # filled by inference in later step
                "index": 0,
                "logprobs": None,
                "finish_reason": "not_implemented",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return resp

