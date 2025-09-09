from __future__ import annotations

import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model exactly as requested
model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",          # torch_dtype is deprecated
    device_map="cuda",
)
print("Loaded:", model_id, "on", torch.cuda.get_device_name(0))

# Harmony
from openai_harmony import load_harmony_encoding, HarmonyEncodingName, Role  # type: ignore
enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


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
        truncation=True,  # safer for very long inputs
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,      # let model decide when to stop
            pad_token_id=tokenizer.eos_token_id,      # consistent padding
        )

    # Slice out each sample’s newly generated tokens
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    outputs: List[str] = []
    for i, in_len in enumerate(input_lengths):
        new_tokens = generated[i, in_len:].tolist()
        # Try Harmony parse; fallback to plain decode if parsing fails
        try:
            msgs = enc.parse_messages_from_completion_tokens(new_tokens, role=Role.ASSISTANT)
            finals = [m for m in msgs if getattr(m, "channel", "final") == "final"]
            text = finals[-1].content if finals else tokenizer.decode(new_tokens)
        except Exception:
            text = tokenizer.decode(new_tokens)
        outputs.append(text)

    return outputs
