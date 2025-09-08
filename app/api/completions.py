from time import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Response, status
from app.core.config import system_fingerprint
from app.schemas import (
    CompletionsRequest,
    CompletionsResponse,
    CompletionChoice,
    Usage,
    LogProbs,
    CompletionJob,
)
from app.core.auth import require_bearer_token
from app.services.inference import generate_completions
from app.core.jobs import job_store, run_completion_job  # NEW

router = APIRouter()


def _build_response(model: str, created: int, gen) -> CompletionsResponse:
    choices = []
    for i, c in enumerate(gen.choices):
        lp = None
        if c.tokens or c.token_logprobs or c.top_logprobs:
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


@router.post("/v1/completions", tags=["completions"], response_model=CompletionsResponse)
async def create_completion(
    payload: CompletionsRequest,
    _auth: Optional[str] = Depends(require_bearer_token),
) -> CompletionsResponse:
    """
    OpenAI-compatible *synchronous* Completions endpoint.
    Suitable when you expect generation to finish quickly.
    """
    model = payload.model
    created = int(time())
    try:
        gen = generate_completions(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _build_response(model, created, gen)


# ----------------------------
# Long-running jobs (async)
# ----------------------------

@router.post(
    "/v1/completions/jobs",
    tags=["completions"],
    response_model=CompletionJob,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_completion_job(
    payload: CompletionsRequest,
    background: BackgroundTasks,
    _auth: Optional[str] = Depends(require_bearer_token),
) -> CompletionJob:
    """
    Submit a long-running batch completion job.
    Returns 202 with a job id you can poll.
    """
    # Persist the request and queue the job
    job = job_store.create(payload.model_dump())
    background.add_task(run_completion_job, job.id)

    return CompletionJob(
        id=job.id,
        created=job.created,
        status=job.status,
        result=None,
        error=None,
    )


@router.get(
    "/v1/completions/jobs/{job_id}",
    tags=["completions"],
    response_model=CompletionJob,
)
async def get_completion_job(job_id: str, response: Response) -> CompletionJob:
    """
    Poll a previously-submitted job. When finished, `result` contains
    the exact `CompletionsResponse` object you would get from /v1/completions.
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    # Surface a 202 while queued/running (handy for proxies/load balancers)
    if job.status in ("queued", "running"):
        response.status_code = status.HTTP_202_ACCEPTED

    return CompletionJob(
        id=job.id,
        created=job.created,
        status=job.status,
        result=job.result,  # already shaped as CompletionsResponse payload
        error=job.error,
    )
