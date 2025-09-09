from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, BackgroundTasks, Response, status, Body, HTTPException

from app.core.auth import require_bearer_token
from app.core.jobs import job_store, run_completion_job

router = APIRouter()

@router.post("/v1/completions", tags=["completions"], status_code=status.HTTP_202_ACCEPTED)
async def create_completion_job(
    payload: Dict[str, Any] = Body(...),
    background: BackgroundTasks = None,
    _auth: Optional[str] = Depends(require_bearer_token),
):
    """
    Submit a BATCH completion job.

    Body must include:
      - prompt: list[str]
      - system_prompt: list[str]    (MUST have same length as 'prompt')
      - optionally: max_tokens, temperature, reasoning

    We enqueue the job and immediately return a job descriptor.
    """
    # Store the raw payload and run in the background
    job = job_store.create(dict(payload))
    if background is not None:
        background.add_task(run_completion_job, job.id)
    else:
        # Fallback if BackgroundTasks not provided (e.g., tests)
        run_completion_job(job.id)

    return {
        "id": job.id,
        "object": "completion.job",
        "created": job.created,
        "status": job.status,   # "queued" | "running" | ...
        "result": None,
        "error": None,
    }


@router.get("/v1/completions/jobs/{job_id}", tags=["completions"])
async def get_completion_job(job_id: str, response: Response):
    """
    Poll a job. While running we return HTTP 202;
    when finished, HTTP 200 with result or error.
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    if job.status in ("queued", "running"):
        response.status_code = status.HTTP_202_ACCEPTED

    return {
        "id": job.id,
        "object": "completion.job",
        "created": job.created,
        "status": job.status,     # queued | running | succeeded | failed
        "result": job.result,     # EXACTLY what the model methods returned (JSON-safe)
        "error": job.error,
    }
