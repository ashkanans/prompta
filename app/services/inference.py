from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import pipeline
pipe = pipeline("text-generation", model="openai/gpt-oss-20b", torch_dtype="auto", device_map="auto")

# =================
# Batch Inference API
# =================
def complete_batch(
    prompts: List[Tuple[str, str]],
    max_new_tokens: int = 200,     # meta tags seldom need more than ~200 tokens
    reasoning: str = "medium",
    temperature: float = 0.01,     # near-deterministic for stable formatting
) -> List[str]:

    if not isinstance(prompts, list) or not all(isinstance(p, tuple) and len(p) == 2 for p in prompts):
        raise ValueError("prompts must be List[Tuple[str, str]]")

    # Build chat conversations
    batch = [
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for sys_prompt, user_prompt in prompts
    ]

    outs = pipe(batch, max_new_tokens=max_new_tokens)
    outputs = [out[0]['generated_text'][-1]['content'].split('final')[-1] for out in outs]
    return outputs
