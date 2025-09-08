import time
import uuid
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        status_code = 500
        response = None
        try:
            response = await call_next(request)
            status_code = getattr(response, "status_code", 200)
            return response
        except Exception:
            # Maintain status_code=500 for error path; let exception bubble
            raise
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            route = getattr(request.scope.get("route", None), "path", request.url.path)
            if response is not None:
                response.headers.setdefault("X-Request-ID", request_id)
            # Structured access log
            import logging

            access_logger = logging.getLogger("access")
            access_logger.info(
                {
                    "request_id": request_id,
                    "route": route,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                }
            )
