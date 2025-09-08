from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import (  # type: ignore
    StoppingCriteria,
    StoppingCriteriaList,
)

from app.core.config import settings
from app.schemas import CompletionsRequest


# Readiness flag; set True once the model is loaded.
_MODEL_READY: bool = False
_LOAD_LOCK = threading.Lock()
_TOKENIZER = None
_MODEL = None


def is_model_ready() -> bool:
    return _MODEL_READY


def current_model_id() -> str:
    return settings.MODEL_ID


def _parse_dtype(name: str | None):
    if not name or name.lower() == "auto":
        return None
    name = name.lower()
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    return None


def _choose_device() -> str:
    dm = (settings.DEVICE_MAP or "auto").lower()
    if dm == "cuda" and torch.cuda.is_available():
        return "cuda"
    if dm == "cpu" or not torch.cuda.is_available():
        return "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def _lazy_load():
    global _MODEL_READY, _TOKENIZER, _MODEL
    if _MODEL is not None and _TOKENIZER is not None:
        return _TOKENIZER, _MODEL
    with _LOAD_LOCK:
        if _MODEL is not None and _TOKENIZER is not None:
            return _TOKENIZER, _MODEL
        device = _choose_device()
        dtype = _parse_dtype(settings.TORCH_DTYPE)

        _TOKENIZER = AutoTokenizer.from_pretrained(settings.MODEL_ID, use_fast=True)
        if (settings.DEVICE_MAP or "auto").lower() == "auto":
            _MODEL = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_ID,
                torch_dtype=dtype,  # None -> auto
                device_map="auto",
            )
        else:
            _MODEL = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_ID,
                torch_dtype=dtype,  # None -> auto
            )
            _MODEL.to(device)
        _MODEL_READY = True
        return _TOKENIZER, _MODEL


class StopOnSequences(StoppingCriteria):
    def __init__(self, tokenizer, stops: Sequence[str]):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops = list(stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:  # type: ignore
        # Check last sequence only
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        for s in self.stops:
            if text.endswith(s):
                return True
        return False


@dataclass
class GenerationResult:
    texts: List[str]
    prompt_tokens: int
    completion_tokens: int
    reached_max: bool


def _strip_stop(text: str, stops: Optional[Sequence[str]]) -> Tuple[str, bool]:
    if not stops:
        return text, False
    for s in stops:
        if text.endswith(s):
            return text[: -len(s)], True
    return text, False


def generate_completions(req: CompletionsRequest) -> GenerationResult:
    if req.best_of is not None and req.best_of > 1:
        raise ValueError("best_of > 1 not supported in this backend")

    tokenizer, model = _lazy_load()
    device = _choose_device()

    prompts: List[str]
    if isinstance(req.prompt, list):
        prompts = req.prompt
    else:
        prompts = [req.prompt]

    # Prepare generation settings
    max_new_tokens = req.max_tokens or 16
    temperature = req.temperature if req.temperature is not None else 1.0
    top_p = req.top_p if req.top_p is not None else 1.0
    n = req.n or 1
    do_sample = (temperature and temperature > 0) or (top_p and top_p < 1.0) or (n > 1)

    # Build stopping criteria
    stopping = StoppingCriteriaList()
    stops = None
    if req.stop is not None:
        stops = req.stop if isinstance(req.stop, list) else [req.stop]
        if stops:
            stopping.append(StopOnSequences(tokenizer, stops))

    # Penalties: approximate via repetition penalty; frequency/presence processors
    logits_processor = None
    try:
        from transformers.generation.logits_process import (  # type: ignore
            LogitsProcessorList,
            RepetitionPenaltyLogitsProcessor,
            FrequencyAndPresencePenaltyLogitsProcessor,
        )

        lp = LogitsProcessorList()
        if req.frequency_penalty or req.presence_penalty:
            freq = float(req.frequency_penalty or 0.0)
            pres = float(req.presence_penalty or 0.0)
            lp.append(FrequencyAndPresencePenaltyLogitsProcessor(
                presence_penalty=pres, frequency_penalty=freq, eos_token_id=tokenizer.eos_token_id
            ))
        # A mild repetition penalty if penalties suggest discouraging repeats
        if (req.frequency_penalty and req.frequency_penalty > 0) or (req.presence_penalty and req.presence_penalty > 0):
            lp.append(RepetitionPenaltyLogitsProcessor(penalty=1.05))
        logits_processor = lp
    except Exception:
        # Fallback: no special processors
        logits_processor = None

    # Seeding
    generator = None
    if req.seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(req.seed))
        generator = g

    all_texts: List[str] = []
    prompt_token_total = 0
    completion_token_total = 0

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        prompt_len = input_ids.shape[-1]

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
            logits_processor=logits_processor,
            generator=generator,
        )

        # outputs shape: (n, seq_len)
        for seq in outputs:
            text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            text, _ = _strip_stop(text, stops)
            if req.echo:
                all_texts.append(prompt + text)
            else:
                all_texts.append(text)

        prompt_token_total += int(prompt_len)
        # approximate completion token count per sequence
        completion_token_total += int(outputs.shape[-1] - prompt_len) * n

    reached_max = False  # heuristic; we cannot easily know per HF without return_dict
    return GenerationResult(
        texts=all_texts,
        prompt_tokens=prompt_token_total,
        completion_tokens=completion_token_total,
        reached_max=reached_max,
    )
