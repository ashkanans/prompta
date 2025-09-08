from time import time

from fastapi import APIRouter, HTTPException, Depends

from app.core.config import system_fingerprint
from app.schemas import CompletionsRequest, CompletionsResponse, CompletionChoice, Usage
from app.core.auth import require_bearer_token
from app.services.inference import generate_completions


router = APIRouter()


@router.post("/v1/completions", tags=["completions"], response_model=CompletionsResponse)
async def create_completion(
    payload: CompletionsRequest, _auth: str | None = Depends(require_bearer_token)
) -> CompletionsResponse:
    """OpenAI-compatible legacy Completions endpoint (scaffold).

    This is a stub that returns a syntactically compatible response. Full
    inference, streaming, logprobs, and best_of behavior will be implemented
    in subsequent steps.
    """
    model = payload.model
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    created = int(time())
    try:
        gen = generate_completions(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    choices = [
        CompletionChoice(
            text=t,
            index=i,
            logprobs=None,
            finish_reason=("length" if gen.reached_max else "stop"),
        )
        for i, t in enumerate(gen.texts)
    ]

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
