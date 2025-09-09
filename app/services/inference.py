from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Model & Tokenizer loading
# =========================
model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",         # torch_dtype is deprecated; let HF choose best dtype
    device_map="cuda",
)
model.eval()
print("Loaded:", model_id, "on", torch.cuda.get_device_name(0))

# Optional: slightly more aggressive math kernels on Ampere+
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# =========
# Harmony IO
# =========
from openai_harmony import load_harmony_encoding, HarmonyEncodingName, Role  # type: ignore

enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

def _harmony_or_decode(new_tokens_ids: List[int]) -> str:
    """
    Try Harmony parsing; if it fails (token mismatch etc.), fall back to plain decode.
    Returns the text content of the final assistant message when available.
    """
    try:
        msgs = enc.parse_messages_from_completion_tokens(new_tokens_ids, role=Role.ASSISTANT)
        finals = [m for m in msgs if getattr(m, "channel", "final") == "final"]
        if finals:
            return finals[-1].content
    except Exception:
        pass
    # Fallback: basic decode
    return tokenizer.decode(new_tokens_ids, skip_special_tokens=True)

# ==================
# EOS tokens utility
# ==================
def _collect_eos_token_ids() -> List[int]:
    """
    Build a list of token ids that should terminate generation.
    Includes the tokenizer's eos plus common chat end markers if present.
    """
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))

    # Add extra known end markers if they exist in this tokenizer vocab
    maybe_tokens = ["<|return|>", "<|assistant_end|>", "<|eot_id|>"]
    vocab = tokenizer.get_vocab()
    for tok in maybe_tokens:
        if tok in vocab:
            tid = vocab[tok]
            if tid is not None:
                eos_ids.add(int(tid))
    return list(eos_ids)

_EOS_TOKEN_IDS = _collect_eos_token_ids()

# =================
# Batch Inference API
# =================
def complete_batch(
    prompts: List[Tuple[str, str]],
    max_new_tokens: int = 200,     # meta tags seldom need more than ~200 tokens
    reasoning: str = "medium",
    temperature: float = 0.01,     # near-deterministic for stable formatting
) -> List[str]:
    """
    Run a batch of conversations.

    Args:
        prompts: list of (system_prompt, user_prompt)
        max_new_tokens: cap on new tokens (per sample)
        reasoning: string injected into the system content
        temperature: sampling temperature (set low or use do_sample=False)

    Returns:
        list[str]: one output per input pair
    """
    if not isinstance(prompts, list) or not all(isinstance(p, tuple) and len(p) == 2 for p in prompts):
        raise ValueError("prompts must be List[Tuple[str, str]]")

    # Build chat conversations
    conversations = [
        [
            {"role": "system", "content": sys_prompt.strip()},
            {"role": "user", "content": user_prompt},
        ]
        for sys_prompt, user_prompt in prompts
    ]

    # Tokenize as a batch
    inputs = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,     # batch padding
        truncation=True,  # safety for very long inputs
    ).to(model.device)

    # Generate
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=False if temperature <= 0.01 else True,
            temperature=max(temperature, 0.01),
            max_new_tokens=int(max_new_tokens),
            eos_token_id=_EOS_TOKEN_IDS,             # stop as soon as any end marker appears
            pad_token_id=tokenizer.eos_token_id,     # consistent padding
            # use_cache=True (default for decoder-only models)
        )

    # Slice out each sample’s newly generated tokens and parse/decode
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    outputs: List[str] = []
    for i, in_len in enumerate(input_lengths):
        new_tokens = generated[i, in_len:].tolist()
        text = _harmony_or_decode(new_tokens).strip()
        outputs.append(text)

    return outputs
