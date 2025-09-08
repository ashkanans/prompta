from __future__ import annotations

from typing import List

from pydantic import BaseModel

from .completions import CompletionsRequest, CompletionsResponse


class BatchCompletionsRequest(BaseModel):
    requests: List[CompletionsRequest]


class BatchCompletionsResponse(BaseModel):
    results: List[CompletionsResponse]
