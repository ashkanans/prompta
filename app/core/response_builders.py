# app/core/response_builders.py
from time import time
from typing import Iterable, Optional

from app.core.config import system_fingerprint
from app.schemas import (
    CompletionsResponse,
    CompletionChoice,
    Usage,
    LogProbs,
)

def build_completions_response(model: str, created: int, gen) -> CompletionsResponse:
    choices = []
    for i, c in enumerate(gen.choices):
        lp = None
        # Only include logprobs if they exist on the choice
        if getattr(c, "tokens", None) or getattr(c, "token_logprobs", None) or getattr(c, "top_logprobs", None):
            lp = LogProbs(
                tokens=c.tokens,
                token_logprobs=c.token_logprobs,
                top_logprobs=c.top_logprobs,  # type: ignore
                text_offset=None,
            )
        choices.append(
            CompletionChoice(
                text=c.text,
                index=i,
                logprobs=lp,
                finish_reason=c.finish_reason,
            )
        )

    usage = Usage(
        prompt_tokens=gen.prompt_tokens,
        completion_tokens=gen.completion_tokens,
        total_tokens=gen.prompt_tokens + gen.completion_tokens,
    )

    return CompletionsResponse(
        id=f"cmpl-{created}",
        created=created,
        model=model,
        system_fingerprint=system_fingerprint(),
        choices=choices,
        usage=usage,
    )
