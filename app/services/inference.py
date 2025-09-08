from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.config import settings
from app.schemas import CompletionsRequest


# Minimal, single-path inference service using HF + Harmony

_tokenizer = None
_model = None


def _load_model_once():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        model_id = settings.MODEL_ID
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="cuda",
        )
        try:
            print("Loaded:", model_id, "on", torch.cuda.get_device_name(0))
        except Exception:
            print("Loaded:", model_id)
    return _tokenizer, _model


def is_model_ready() -> bool:
    tok, mdl = _tokenizer, _model
    return tok is not None and mdl is not None


# Harmony encoding
from openai_harmony import load_harmony_encoding, HarmonyEncodingName, Role  # type: ignore

enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def complete(prompt: str, system_prompt: str = "", max_new_tokens: int = 256, reasoning: str = "medium", temperature: float = 0.7) -> str:
    tokenizer, model = _load_model_once()

    messages = [
        {"role": "system", "content": f"Reasoning: {reasoning}\n{system_prompt}".strip()},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0][inputs["input_ids"].shape[-1]:].tolist()

    msgs = enc.parse_messages_from_completion_tokens(new_tokens, role=Role.ASSISTANT)
    finals = [m for m in msgs if getattr(m, "channel", "final") == "final"]
    return finals[-1].content if finals else tokenizer.decode(new_tokens)


def complete_batch(
    prompts: List[Tuple[str, str]],
    max_new_tokens: int = 256,
    reasoning: str = "medium",
    temperature: float = 0.7,
) -> List[str]:
    """
    prompts: list of (system_prompt, user_prompt)
    returns: list[str] model outputs, one per input pair
    """
    tokenizer, model = _load_model_once()

    conversations = [
        [
            {"role": "system", "content": f"Reasoning: {reasoning}\n{system_prompt}".strip()},
            {"role": "user", "content": user_prompt},
        ]
        for system_prompt, user_prompt in prompts
    ]

    inputs = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    outputs: List[str] = []
    for i, in_len in enumerate(input_lengths):
        new_tokens = generated[i, in_len:].tolist()
        msgs = enc.parse_messages_from_completion_tokens(new_tokens, role=Role.ASSISTANT)
        finals = [m for m in msgs if getattr(m, "channel", "final") == "final"]
        text = finals[-1].content if finals else tokenizer.decode(new_tokens)
        outputs.append(text)

    return outputs


@dataclass
class ChoiceOut:
    text: str
    tokens: List[str]
    token_logprobs: List[Optional[float]]
    top_logprobs: Optional[List[dict]]
    finish_reason: str
    completion_tokens: int


@dataclass
class GenerationResult:
    choices: List[ChoiceOut]
    prompt_tokens: int
    completion_tokens: int


def generate_completions(req: CompletionsRequest) -> GenerationResult:
    # Normalize input to a list of prompts
    prompts: List[str] = req.prompt if isinstance(req.prompt, list) else [req.prompt]

    # Build (system, user) pairs
    sys = req.system_prompt or ""
    pairs: List[Tuple[str, str]] = [(sys, p) for p in prompts]

    # Default params to match the requested flow
    max_new = req.max_tokens or 600
    reasoning = req.reasoning or "medium"
    temperature = 0.1 if req.temperature is None else req.temperature

    t0 = time.time()
    texts = complete_batch(pairs, max_new_tokens=max_new, reasoning=reasoning, temperature=temperature)
    t1 = time.time()
    _ = (t0, t1)  # kept for clarity; times can be logged if needed

    tokenizer, _model_ref = _load_model_once()

    choices: List[ChoiceOut] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Rough token accounting: chat template for prompt; plain encode for output
    for (system_prompt, user_prompt), text in zip(pairs, texts):
        conv = [
            {"role": "system", "content": f"Reasoning: {reasoning}\n{system_prompt}".strip()},
            {"role": "user", "content": user_prompt},
        ]
        enc_in = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
        prompt_tok = int(enc_in.shape[-1]) if hasattr(enc_in, "shape") else len(enc_in[0])  # type: ignore
        comp_tok = len(tokenizer.encode(text))
        total_prompt_tokens += prompt_tok
        total_completion_tokens += comp_tok

        choices.append(
            ChoiceOut(
                text=text,
                tokens=[],
                token_logprobs=[],
                top_logprobs=None,
                finish_reason="stop",
                completion_tokens=comp_tok,
            )
        )

    return GenerationResult(
        choices=choices,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
    )
