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
