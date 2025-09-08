from __future__ import annotations

import time
from typing import List, Tuple, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

# ---------- DO NOT CHANGE THESE (per your request) ----------

def complete(prompt: tuple[str, str], max_new_tokens=256, reasoning="medium", temperature=0.7):
    system_prompt, user_prompt = prompt
    messages = [
        {"role": "system", "content": f"Reasoning: {reasoning}\n{system_prompt}".strip()},
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
    prompts: list[tuple[str, str]],
    max_new_tokens: int = 256,
    reasoning: str = "medium",
    temperature: float = 0.7,
):
    """
    prompts: list of (system_prompt, user_prompt)
    returns: list[str] model outputs, one per input pair
    """
    # Build a batch of conversations
    conversations = [
        [
            {"role": "system", "content": f"Reasoning: {reasoning}\n{system_prompt}".strip()},
            {"role": "user", "content": user_prompt},
        ]
        for system_prompt, user_prompt in prompts
    ]

    # Tokenize as a batch
    inputs = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
        truncation=True,  # optional but safer
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
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

# ---------- Helpers (safe to add) ----------

def normalize_text_content(x: Any) -> str:
    """
    Convert Harmony content (str | TextContent | list[TextContent] | mixed) to a plain str.
    We do NOT touch/modify complete()/complete_batch(); we only normalize their returns here.
    """
    # already a string
    if isinstance(x, str):
        return x

    # one TextContent-like object with .text
    if hasattr(x, "text") and isinstance(getattr(x, "text"), str):
        return x.text

    # list of segments (e.g., [TextContent(...), ...] or mixed)
    if isinstance(x, list):
        parts = []
        for item in x:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text") and isinstance(getattr(item, "text"), str):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "".join(parts)

    # anything else: stringify
    return str(x)


def generate_completions(req: dict) -> dict:
    """
    Returns a plain dict:
    {
      "choices": [{"text": str, "index": int, "logprobs": None, "finish_reason": "stop"}, ...],
      "prompt_tokens": int,
      "completion_tokens": int
    }
    """
    # prompt can be str or list[str]; keep it loose (no Pydantic)
    raw_prompts = req.get("prompt")
    if isinstance(raw_prompts, str):
        prompts: List[str] = [raw_prompts]
    elif isinstance(raw_prompts, list):
        prompts = [str(p) for p in raw_prompts]
    else:
        raise ValueError("prompt must be a string or list of strings")

    sys_prompt = req.get("system_prompt") or ""
    pairs: List[Tuple[str, str]] = [(sys_prompt, p) for p in prompts]

    # defaults
    max_new = req.get("max_tokens") or 600
    reasoning = req.get("reasoning") or "medium"
    temperature = req.get("temperature")
    if temperature is None:
        # your example uses 0.1 in batch; but keep flexible
        temperature = 0.7

    # Run batch
    t0 = time.time()
    texts = complete_batch(
        pairs,
        max_new_tokens=max_new,
        reasoning=reasoning,
        temperature=float(temperature),
    )
    t1 = time.time()
    _ = (t0, t1)

    # Normalize to strings
    norm_texts = [normalize_text_content(t) for t in texts]

    # Rough token accounting
    total_prompt_tokens = 0
    total_completion_tokens = 0
    choices = []
    for idx, ((sp, up), text) in enumerate(zip(pairs, norm_texts)):
        conv = [
            {"role": "system", "content": f"Reasoning: {reasoning}\n{sp}".strip()},
            {"role": "user", "content": up},
        ]
        enc_in = tokenizer.apply_chat_template(
            conv,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        prompt_tok = int(enc_in["input_ids"].shape[1]) if "input_ids" in enc_in else 0
        comp_tok = len(tokenizer.encode(text, add_special_tokens=False)) if text else 0
        total_prompt_tokens += prompt_tok
        total_completion_tokens += comp_tok

        choices.append({
            "text": text,                # guaranteed str
            "index": idx,
            "logprobs": None,
            "finish_reason": "stop",
        })

    return {
        "choices": choices,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }


def is_model_ready() -> bool:
    return True
