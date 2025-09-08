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

    all_choices: List[ChoiceOut] = []
    prompt_token_total = 0
    completion_token_total = 0

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(model.device)
        prompt_len = input_ids.shape[-1]
        gen = model.generate(
            input_ids=input_ids,
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

        # gen.sequences: (best_of, total_len), gen.scores: List[Tensor] length gen_len, each (best_of, vocab)
        sequences = gen.sequences
        gen_len = sequences.shape[-1] - prompt_len
        scores_per_step = gen.scores  # list length gen_len

        # Compute logprobs matrices [gen_len, best_of, vocab]
        logprobs_steps = [torch.log_softmax(s, dim=-1) for s in scores_per_step]

        # For each sequence, collect sampled token ids and their logprobs
        seq_token_ids = [sequences[i, prompt_len:].tolist() for i in range(sequences.shape[0])]
        seq_token_logprobs = [
            [float(logprobs_steps[t][i, tok_id].item()) for t, tok_id in enumerate(token_ids)]
            for i, token_ids in enumerate(seq_token_ids)
        ]

        # Prepare top-k per step if requested
        top_k = min(5, int(req.logprobs or 0)) if req.logprobs else 0
        seq_top_logprobs: List[Optional[List[Optional[Dict[str, float]]]]] = []
        if top_k > 0:
            for i in range(sequences.shape[0]):
                per_step: List[Dict[str, float]] = []
                for t in range(gen_len):
                    lp = logprobs_steps[t][i]
                    vals, idx = torch.topk(lp, k=top_k, dim=-1)
                    toks = tokenizer.convert_ids_to_tokens(idx.tolist())
                    per_step.append({toks[j]: float(vals[j].item()) for j in range(len(toks))})
                seq_top_logprobs.append(per_step)
        else:
            seq_top_logprobs = [None for _ in range(sequences.shape[0])]

        # Score sequences by mean token logprob for best_of selection
        means = [float(torch.tensor(lp).mean().item()) if len(lp) > 0 else float("-inf") for lp in seq_token_logprobs]
        # Select indices of top n sequences
        selected_idx = sorted(range(len(means)), key=lambda i: means[i], reverse=True)[: n]

        # Build choices for selected sequences
        for rank, i_sel in enumerate(selected_idx):
            seq = sequences[i_sel]
            gen_ids = seq[prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            stripped_text, stopped = _strip_stop(text, stops)

            # Tokens array for echo or non-echo
            prompt_ids = enc.input_ids[0].tolist()
            combined_ids = (prompt_ids + gen_ids.tolist()) if req.echo else gen_ids.tolist()
            tokens = tokenizer.convert_ids_to_tokens(combined_ids)

            # Logprobs alignment: pad prompt part with None
            token_lp: List[Optional[float]]
            top_lp: Optional[List[Optional[Dict[str, float]]]]
            if req.echo:
                pad = [None] * len(prompt_ids)
                token_lp = pad + [float(v) for v in seq_token_logprobs[i_sel]]
                if top_k > 0 and isinstance(seq_top_logprobs[i_sel], list):
                    top_lp = [None] * len(prompt_ids) + seq_top_logprobs[i_sel]  # type: ignore
                else:
                    top_lp = None
            else:
                token_lp = [float(v) for v in seq_token_logprobs[i_sel]]
                top_lp = seq_top_logprobs[i_sel]

            finish_reason = "length"
            if stopped or (gen_ids.shape[0] < max_new_tokens) or (
                gen_ids.shape[0] == max_new_tokens and int(gen_ids[-1].item()) == tokenizer.eos_token_id
            ):
                finish_reason = "stop"

            choice = ChoiceOut(
                text=(prompt + stripped_text) if req.echo else stripped_text,
                tokens=tokens,
                token_logprobs=token_lp,
                top_logprobs=top_lp,
                finish_reason=finish_reason,
                completion_tokens=int(gen_ids.shape[0]),
            )
            all_choices.append(choice)

        prompt_token_total += int(prompt_len)
        # Count completion tokens only for selected sequences
        completion_token_total += sum(c.completion_tokens for c in all_choices[-len(selected_idx) :])

    return GenerationResult(
        choices=all_choices,
        prompt_tokens=prompt_token_total,
        completion_tokens=completion_token_total,
    )
