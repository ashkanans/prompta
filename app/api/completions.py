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
from app.core.response_builders import build_completions_response as _build_response

router = APIRouter()


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
