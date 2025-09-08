from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings


_bearer_scheme = HTTPBearer(auto_error=False)


def require_bearer_token(
    request: Request, creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme)
) -> Optional[str]:
    """Optional bearer auth allowlist.

    - If `AUTH_BEARER_TOKENS` is empty, authentication is disabled (returns None).
    - Otherwise, requires a `Bearer <token>` Authorization header with a token
      present in the allowlist.
    """
    allowlist = settings.AUTH_BEARER_TOKENS
    if not allowlist:
        return None

    token = creds.credentials if (creds and creds.scheme.lower() == "bearer") else None
    if token and token in allowlist:
        return token

    # Log a redacted auth failure
    request.app.logger = getattr(request.app, "logger", None)
    return _unauthorized()


def _unauthorized():
    raise HTTPException(status_code=401, detail="Unauthorized")

