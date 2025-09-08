from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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

        _TOKENIZER = AutoTokenizer.from_pretrained(
            settings.MODEL_ID,
            use_fast=True,
            trust_remote_code=True,
        )
        if (settings.DEVICE_MAP or "auto").lower() == "auto":
            _MODEL = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_ID,
                torch_dtype=dtype,  # None -> auto
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            _MODEL = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_ID,
                torch_dtype=dtype,  # None -> auto
                trust_remote_code=True,
            )
            _MODEL.to(device)
        # Ensure pad token exists for generation APIs
        if _TOKENIZER.pad_token_id is None and _TOKENIZER.eos_token_id is not None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token
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
class ChoiceOut:
    text: str
    tokens: List[str]
    token_logprobs: List[Optional[float]]
    top_logprobs: Optional[List[Optional[Dict[str, float]]]]
    finish_reason: str
    completion_tokens: int


@dataclass
class GenerationResult:
    choices: List[ChoiceOut]
    prompt_tokens: int
    completion_tokens: int


def _strip_stop(text: str, stops: Optional[Sequence[str]]) -> Tuple[str, bool]:
    if not stops:
        return text, False
    for s in stops:
        if text.endswith(s):
            return text[: -len(s)], True
    return text, False


def generate_completions(req: CompletionsRequest) -> GenerationResult:
    tokenizer, model = _lazy_load()
    device = _choose_device()

    prompts: List[str] = req.prompt if isinstance(req.prompt, list) else [req.prompt]

    # Prepare generation settings
    max_new_tokens = req.max_tokens or 16
    temperature = req.temperature if req.temperature is not None else 1.0
    top_p = req.top_p if req.top_p is not None else 1.0
    n = req.n or 1
    best_of = req.best_of or n
    if best_of < n:
        raise ValueError("best_of must be >= n")
    if req.stream and best_of > 1:
        raise ValueError("streaming is not supported with best_of > 1")
    do_sample = (temperature and temperature > 0) or (top_p and top_p < 1.0) or (n > 1)

    # Build stopping criteria
    stopping = StoppingCriteriaList()
    stops = None
    if req.stop is not None:
        stops = req.stop if isinstance(req.stop, list) else [req.stop]
        if stops:
            stopping.append(StopOnSequences(tokenizer, stops))

    # Penalties
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
            lp.append(
                FrequencyAndPresencePenaltyLogitsProcessor(
                    presence_penalty=pres,
                    frequency_penalty=freq,
                    eos_token_id=tokenizer.eos_token_id,
                )
            )
        if (req.frequency_penalty and req.frequency_penalty > 0) or (
            req.presence_penalty and req.presence_penalty > 0
        ):
            lp.append(RepetitionPenaltyLogitsProcessor(penalty=1.05))
        logits_processor = lp
    except Exception:
        logits_processor = None

    # Seeding
    generator = None
    if req.seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(req.seed))
        generator = g

    # Tokenize batch
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    prompt_lens = enc.attention_mask.sum(dim=1).tolist()

    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=best_of,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping,
        logits_processor=logits_processor,
        generator=generator,
        output_scores=True,
        return_dict_in_generate=True,
    )

    sequences = gen.sequences  # (B*best_of, total_len)
    scores_per_step = gen.scores  # list length max_gen_steps
    logprobs_steps = [torch.log_softmax(s, dim=-1) for s in scores_per_step]

    B = len(prompts)
    top_k = min(5, int(req.logprobs or 0)) if req.logprobs else 0

    def row_index(b: int, k: int) -> int:
        return b * best_of + k

    all_choices: List[ChoiceOut] = []
    prompt_token_total = sum(int(l) for l in prompt_lens)
    completion_token_total = 0

    for b in range(B):
        p_len = int(prompt_lens[b])
        prompt_ids = enc.input_ids[b].tolist()

        cand_lp: List[List[float]] = []
        cand_top: List[Optional[List[Optional[Dict[str, float]]]]] = []
        cand_texts: List[str] = []
        cand_gen_ids: List[List[int]] = []

        for k in range(best_of):
            r = row_index(b, k)
            seq = sequences[r]
            gen_ids = seq[p_len:]
            token_ids = gen_ids.tolist()
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            stripped_text, _stopped = _strip_stop(text, stops)

            # token logprobs for this row
            row_lp = [float(logprobs_steps[t][r, token_ids[t]].item()) for t in range(len(token_ids))]

            # top-k
            if top_k > 0:
                per_step: List[Dict[str, float]] = []
                for t in range(len(token_ids)):
                    lpv = logprobs_steps[t][r]
                    vals, idx = torch.topk(lpv, k=top_k, dim=-1)
                    toks = tokenizer.convert_ids_to_tokens(idx.tolist())
                    per_step.append({toks[j]: float(vals[j].item()) for j in range(len(toks))})
                cand_top.append(per_step)
            else:
                cand_top.append(None)

            cand_lp.append(row_lp)
            cand_texts.append(stripped_text)
            cand_gen_ids.append(token_ids)

        # Select top-n by mean logprob
        means = [float(torch.tensor(lp).mean().item()) if len(lp) else float("-inf") for lp in cand_lp]
        selected = sorted(range(len(means)), key=lambda i: means[i], reverse=True)[: n]

        for k in selected:
            gen_ids_list = cand_gen_ids[k]
            tokens = tokenizer.convert_ids_to_tokens((prompt_ids + gen_ids_list) if req.echo else gen_ids_list)

            if req.echo:
                token_lp = [None] * len(prompt_ids) + [float(v) for v in cand_lp[k]]
                top_lp = ([None] * len(prompt_ids) + cand_top[k]) if top_k > 0 and isinstance(cand_top[k], list) else None
                text_out = prompts[b] + cand_texts[k]
            else:
                token_lp = [float(v) for v in cand_lp[k]]
                top_lp = cand_top[k]
                text_out = cand_texts[k]

            finish_reason = "stop"
            if len(gen_ids_list) >= max_new_tokens and (not stops):
                finish_reason = "length"

            all_choices.append(
                ChoiceOut(
                    text=text_out,
                    tokens=tokens,
                    token_logprobs=token_lp,
                    top_logprobs=top_lp,
                    finish_reason=finish_reason,
                    completion_tokens=len(gen_ids_list),
                )
            )
            completion_token_total += len(gen_ids_list)

    return GenerationResult(
        choices=all_choices,
        prompt_tokens=prompt_token_total,
        completion_tokens=completion_token_total,
    )
