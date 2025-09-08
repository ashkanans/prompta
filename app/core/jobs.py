import threading
import time
import uuid
from typing import Any, Dict, Optional

from app.services.inference import generate_completions
from app.core.response_builders import build_completions_response_dict

class _Job:
    def __init__(self, payload: Dict[str, Any]):
        self.id = f"job-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.updated = self.created
        self.status = "queued"  # queued | running | succeeded | failed
        self.payload = payload
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

class _JobStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, _Job] = {}

    def create(self, payload: Dict[str, Any]) -> _Job:
        job = _Job(payload)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[_Job]:
        return self._jobs.get(job_id)

    def set_running(self, job_id: str):
        with self._lock:
            j = self._jobs[job_id]
            j.status = "running"
            j.updated = int(time.time())

    def set_succeeded(self, job_id: str, result: Dict[str, Any]):
        with self._lock:
            j = self._jobs[job_id]
            j.status = "succeeded"
            j.result = result
            j.updated = int(time.time())

    def set_failed(self, job_id: str, error: str):
        with self._lock:
            j = self._jobs[job_id]
            j.status = "failed"
            j.error = error
            j.updated = int(time.time())

job_store = _JobStore()

def run_completion_job(job_id: str):
    job = job_store.get(job_id)
    if not job:
        return
    try:
        job_store.set_running(job_id)
        req = dict(job.payload)
        model = req.get("model", "openai/gpt-oss-20b")

        gen = generate_completions(req)
        response = build_completions_response_dict(model, gen)

        job_store.set_succeeded(job_id, response)
    except Exception as e:
        job_store.set_failed(job_id, str(e))
