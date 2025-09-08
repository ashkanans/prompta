from time import time
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Response, status, Body

from app.core.auth import require_bearer_token
from app.services.inference import generate_completions
from app.core.jobs import job_store, run_completion_job
from app.core.response_builders import build_completions_response_dict

router = APIRouter()

@router.post("/v1/completions", tags=["completions"])
async def create_completion(
    payload: Dict[str, Any] = Body(...),
    _auth: Optional[str] = Depends(require_bearer_token),
):
    """
    OpenAI-compatible synchronous Completions endpoint (no Pydantic).
    """
    try:
        model = payload.get("model", "openai/gpt-oss-20b")
        gen = generate_completions(payload)
        return build_completions_response_dict(model, gen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # surface unexpected errors
        raise HTTPException(status_code=500, detail=str(e))

# Long-running jobs (async)

@router.post(
    "/v1/completions/jobs",
    tags=["completions"],
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_completion_job(
    payload: Dict[str, Any] = Body(...),
    background: BackgroundTasks = None,
    _auth: Optional[str] = Depends(require_bearer_token),
):
    """
    Submit a long-running batch completion job.
    Returns 202 with a job id you can poll.
    """
    job = job_store.create(dict(payload))
    if background is not None:
        background.add_task(run_completion_job, job.id)
    else:
        # Fallback: synchronous run (only used if BackgroundTasks not provided)
        run_completion_job(job.id)

    return {
        "id": job.id,
        "object": "completion.job",
        "created": job.created,
        "status": job.status,
        "result": None,
        "error": None,
    }

@router.get(
    "/v1/completions/jobs/{job_id}",
    tags=["completions"],
)
async def get_completion_job(job_id: str, response: Response):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    if job.status in ("queued", "running"):
        response.status_code = status.HTTP_202_ACCEPTED

    return {
        "id": job.id,
        "object": "completion.job",
        "created": job.created,
        "status": job.status,
        "result": job.result,
        "error": job.error,
    }
