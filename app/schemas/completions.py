from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import settings


PromptType = Union[str, List[str]]
StopType = Union[str, List[str]]


class StreamOptions(BaseModel):
    class Config:
        extra = "allow"


class CompletionsRequest(BaseModel):
    model: str = Field(default_factory=lambda: settings.MODEL_ID)
    prompt: PromptType

    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    best_of: Optional[int] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[StopType] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    user: Optional[str] = None
    logit_bias: Optional[Dict[Union[str, int], float]] = None

    # GPT-OSS/Harmony-specific optional inputs
    use_harmony: Optional[bool] = None
    reasoning: Optional[str] = "medium"
    system_prompt: Optional[str] = None

    @field_validator("prompt", mode="before")
    @classmethod
    def coerce_prompt(cls, v: PromptType):
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            if len(v) == 1:
                return v[0]
            if not all(isinstance(x, str) for x in v):
                raise ValueError("prompt list must contain only strings")
            return v
        raise ValueError("prompt must be a string or list of strings")

    @field_validator("stop", mode="before")
    @classmethod
    def coerce_stop(cls, v: Optional[StopType]):
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            if len(v) > 4:
                raise ValueError("stop may contain at most 4 strings")
            if not all(isinstance(x, str) for x in v):
                raise ValueError("stop list must contain only strings")
            return v
        raise ValueError("stop must be a string or list of strings")

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        if v is None:
            return None
        if v < 0:
            raise ValueError("max_tokens must be >= 0")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if v is None:
            return None
        if not (0 <= v <= 2):
            raise ValueError("temperature must be between 0 and 2")
        return v

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v):
        if v is None:
            return None
        if not (0 <= v <= 1):
            raise ValueError("top_p must be between 0 and 1")
        return v

    @field_validator("n")
    @classmethod
    def validate_n(cls, v):
        if v is None:
            return None
        if v < 1:
            raise ValueError("n must be >= 1")
        return v

    @field_validator("best_of")
    @classmethod
    def validate_best_of(cls, v):
        if v is None:
            return None
        if v < 1:
            raise ValueError("best_of must be >= 1")
        return v

    @field_validator("logprobs")
    @classmethod
    def validate_logprobs(cls, v):
        if v is None:
            return None
        if not (0 <= v <= 5):
            raise ValueError("logprobs must be between 0 and 5")
        return v

    @field_validator("presence_penalty", "frequency_penalty")
    @classmethod
    def validate_penalties(cls, v):
        if v is None:
            return None
        if not (-2 <= v <= 2):
            raise ValueError("penalties must be between -2 and 2")
        return v

    @field_validator("logit_bias", mode="before")
    @classmethod
    def coerce_logit_bias(cls, v):
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("logit_bias must be a mapping")
        norm: Dict[str, float] = {}
        for k, val in v.items():
            key = str(k)
            try:
                fval = float(val)
            except Exception as e:
                raise ValueError(f"invalid logit_bias value for {k}: {val}") from e
            if not (-100 <= fval <= 100):
                raise ValueError("logit_bias values must be between -100 and 100")
            norm[key] = fval
        return norm

    @model_validator(mode="after")
    def cross_validate(self):
        n = self.n or 1
        if self.best_of is not None and self.best_of < n:
            raise ValueError("best_of must be greater than or equal to n")
        if self.stream and self.best_of not in (None, 1):
            raise ValueError("streaming is not supported with best_of > 1")
        if isinstance(self.prompt, list) and self.n not in (None, 1):
            pass
        if self.use_harmony is None:
            self.use_harmony = "gpt-oss" in (self.model or "").lower()
        return self


class LogProbs(BaseModel):
    tokens: List[str]
    token_logprobs: List[Optional[float]]
    top_logprobs: Optional[List[Dict[str, float]]] = None
    text_offset: Optional[List[int]] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length", "content_filter", "not_implemented"]] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionsResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = Field(default="text_completion")
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    choices: List[CompletionChoice]
    usage: Usage


# --------- Job (async) envelope ---------

class CompletionJob(BaseModel):
    id: str
    object: Literal["completion.job"] = Field(default="completion.job")
    created: int
    status: Literal["queued", "running", "succeeded", "failed"]
    # When finished, the exact CompletionsResponse payload goes here
    result: Optional[CompletionsResponse] = None
    error: Optional[str] = None
