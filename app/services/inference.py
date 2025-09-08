from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.schemas import CompletionsRequest

# Load model exactly as requested
model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cuda",
)
print("Loaded:", model_id, "on", torch.cuda.get_device_name(0))

# Harmony
from openai_harmony import load_harmony_encoding, HarmonyEncodingName, Role  # type: ignore

enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def _safe_pad_id():
    # pad_token_id may be None for some LLMs; fall back to eos
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def complete(prompt: tuple[str, str], max_new_tokens=256, reasoning="medium", temperature=0.7):
    sys_prompt, user_prompt = prompt
    messages = [
        {"role": "system", "content": f"Reasoning: {reasoning}\n{sys_prompt}".strip()},
        {"role": "user", "content": user_prompt},
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
):
    """
    prompts: list of (system_prompt, user_prompt)
    returns: list[str] model outputs, one per input pair
    """
    conversations = [
        [
            {"role": "system", "content": f"Reasoning: {reasoning}\n{sys_prompt}".strip()},
            {"role": "user", "content": user_prompt},
        ]
        for sys_prompt, user_prompt in prompts
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
            pad_token_id=_safe_pad_id(),
        )

    # Slice out each sample’s newly generated tokens
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    outputs = []
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
    # Normalize prompts: allow str or list[str]
    prompts: List[str] = req.prompt if isinstance(req.prompt, list) else [req.prompt]

    # Build (system, user) pairs
    sys_prompt = req.system_prompt or ""
    pairs: List[Tuple[str, str]] = [(sys_prompt, p) for p in prompts]

    # Default params to match the requested flow
    max_new = req.max_tokens or 600
    reasoning = req.reasoning or "medium"
    temperature = 0.1 if req.temperature is None else req.temperature

    # Run batch generation
    t0 = time.time()
    texts = complete_batch(
        pairs,
        max_new_tokens=max_new,
        reasoning=reasoning,
        temperature=temperature,
    )
    t1 = time.time()
    _ = (t0, t1)  # keep for optional logging

    choices: List[ChoiceOut] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Rough token accounting
    for (sys_p, user_p), text in zip(pairs, texts):
        conv = [
            {"role": "system", "content": f"Reasoning: {reasoning}\n{sys_p}".strip()},
            {"role": "user", "content": user_p},
        ]
        enc_in = tokenizer.apply_chat_template(
            conv,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        try:
            prompt_tok = int(enc_in["input_ids"].shape[1])
        except Exception:
            prompt_tok = 0
        try:
            comp_tok = len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            comp_tok = 0

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


def is_model_ready() -> bool:
    # Model and tokenizer are loaded at import time in this flow
    return True
