from time import time

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import require_bearer_token
from app.core.config import system_fingerprint
from app.schemas import (
    BatchCompletionsRequest,
    BatchCompletionsResponse,
    CompletionsResponse,
    CompletionChoice,
    Usage,
    LogProbs,
)
from app.services.inference import generate_completions


router = APIRouter()


@router.post("/completions", response_model=BatchCompletionsResponse)
async def batch_completions(
    payload: BatchCompletionsRequest, _auth: str | None = Depends(require_bearer_token)
) -> BatchCompletionsResponse:
    created = int(time())
    results: list[CompletionsResponse] = []

    for i, req in enumerate(payload.requests):
        try:
            gen = generate_completions(req)
        except ValueError as e:
            # Surface which item failed
            raise HTTPException(status_code=400, detail=f"item {i}: {e}")

        choices = []
        for j, c in enumerate(gen.choices):
            lp = None
            # Use req.logprobs (per-item setting) to include logprobs if requested
            if req.logprobs:
                lp = LogProbs(
                    tokens=c.tokens,
                    token_logprobs=c.token_logprobs,
                    top_logprobs=c.top_logprobs,  # type: ignore
                    text_offset=None,
                )
            choices.append(
                CompletionChoice(
                    text=c.text,
                    index=j,
                    logprobs=lp,
                    finish_reason=c.finish_reason,
                )
            )

        usage = Usage(
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            total_tokens=gen.prompt_tokens + gen.completion_tokens,
        )

        results.append(
            CompletionsResponse(
                id=f"cmpl-batch-{created}-{i}",
                created=created,
                model=req.model or "",
                system_fingerprint=system_fingerprint(),
                choices=choices,
                usage=usage,
            )
        )

    return BatchCompletionsResponse(results=results)
