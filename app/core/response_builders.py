from time import time
from typing import Dict, Any

from app.core.config import system_fingerprint

def build_completions_response_dict(model: str, gen: Dict[str, Any]) -> Dict[str, Any]:
    created = int(time())
    prompt_tokens = int(gen.get("prompt_tokens", 0))
    completion_tokens = int(gen.get("completion_tokens", 0))
    return {
        "id": f"cmpl-{created}",
        "object": "text_completion",
        "created": created,
        "model": model,
        "system_fingerprint": system_fingerprint(),
        "choices": gen.get("choices", []),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
