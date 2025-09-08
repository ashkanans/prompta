from typing import Any, Dict

import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from app.core.config import settings


def _request_id(request: Request) -> str:
    rid = getattr(getattr(request, "state", object()), "request_id", None)
    return rid or "unknown"


def _safe_detail(detail: Any) -> Any:
    # Trust HTTPException detail as provided; otherwise avoid leaking internals
    return detail


async def http_exception_handler(request: Request, exc: HTTPException):
    rid = _request_id(request)
    logger = logging.getLogger("app.errors")
    level = logging.WARNING if 400 <= exc.status_code < 500 else logging.ERROR
    logger.log(level, {"request_id": rid, "status_code": exc.status_code, "detail": exc.detail})
    return JSONResponse(status_code=exc.status_code, content={"detail": _safe_detail(exc.detail), "request_id": rid})


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    rid = _request_id(request)
    logger = logging.getLogger("app.errors")
    logger.warning({"request_id": rid, "status_code": 422, "errors": exc.errors()})
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "request_id": rid})


async def unhandled_exception_handler(request: Request, exc: Exception):
    rid = _request_id(request)
    logger = logging.getLogger("app.errors")
    logger.error({"request_id": rid, "status_code": 500, "error": str(exc)}, exc_info=True)
    detail = str(exc) if settings.DEBUG else "Internal Server Error"
    return JSONResponse(status_code=500, content={"detail": detail, "request_id": rid})


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

