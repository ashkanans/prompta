from time import time

from fastapi import APIRouter, HTTPException, Depends

from app.core.config import system_fingerprint
from app.schemas import CompletionsRequest, CompletionsResponse, CompletionChoice, Usage
from app.core.auth import require_bearer_token


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
    return CompletionsResponse(
        id=f"cmpl-stub-{created}",
        created=created,
        model=model,
        system_fingerprint=system_fingerprint(),
        choices=[
            CompletionChoice(
                text="",
                index=0,
                logprobs=None,
                finish_reason="not_implemented",
            )
        ],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
