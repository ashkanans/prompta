import threading
import time
import uuid
import json
from typing import Any, Dict, Optional, List, Tuple

from app.services.inference import complete, complete_batch

class _Job:
    def __init__(self, payload: Dict[str, Any]):
        self.id = f"job-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.updated = self.created
        self.status = "queued"   # queued | running | succeeded | failed
        self.payload = payload   # raw request dict
        self.result: Optional[Any] = None   # raw result (JSON-safe)
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

    def _update(self, job_id: str, **kwargs):
        with self._lock:
            j = self._jobs[job_id]
            for k, v in kwargs.items():
                setattr(j, k, v)
            j.updated = int(time.time())

    def set_running(self, job_id: str):
        self._update(job_id, status="running")

    def set_succeeded(self, job_id: str, result: Any):
        self._update(job_id, status="succeeded", result=result)

    def set_failed(self, job_id: str, error: str):
        self._update(job_id, status="failed", error=error)

job_store = _JobStore()


def _jsonify_like_notebook(obj: Any) -> Any:
    """
    Make sure 'result' is JSON serializable while preserving the "what you get"
    feel. If an object isn't JSON-serializable (e.g., Harmony TextContent),
    we return its repr(). Lists/dicts are processed recursively.
    """
    # Fast-path for simple JSON types
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        pass

    # Recurse common containers
    if isinstance(obj, list):
        return [_jsonify_like_notebook(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_jsonify_like_notebook(x) for x in obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify_like_notebook(v) for k, v in obj.items()}

    # Last resort: string representation (shows up like [TextContent(text='...')])
    try:
        return repr(obj)
    except Exception:
        return str(obj)


def run_completion_job(job_id: str):
    job = job_store.get(job_id)
    if not job:
        return

    try:
        job_store.set_running(job_id)

        req = dict(job.payload)
        prompt = req.get("prompt")
        system_prompt = req.get("system_prompt", "")
        max_tokens = req.get("max_tokens", 256)
        temperature = req.get("temperature", 0.7)
        reasoning = req.get("reasoning", "medium")

        # Decide single vs batch and call the exact methods user provided
        if isinstance(prompt, list) and isinstance(system_prompt, list):
            pairs: List[Tuple[str, str]] = [(system_prompt[i], prompt[i]) for i in range(len(prompt))]
            result_raw = complete_batch(
                pairs,
                max_new_tokens=int(max_tokens),
                reasoning=str(reasoning),
                temperature=float(temperature),
            )
        elif isinstance(prompt, str):
            result_raw = complete(
                (system_prompt, prompt),
                max_new_tokens=int(max_tokens),
                reasoning=str(reasoning),
                temperature=float(temperature),
            )
        else:
            raise ValueError("prompt must be a string or a list of strings")

        # Store EXACTLY what we got, JSON-ified minimally via repr for non-JSON types
        result_payload = _jsonify_like_notebook(result_raw)
        job_store.set_succeeded(job_id, result_payload)

    except Exception as e:
        job_store.set_failed(job_id, str(e))
