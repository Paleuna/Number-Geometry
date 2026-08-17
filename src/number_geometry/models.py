"""Small model-loading helper for the extraction notebook."""
from __future__ import annotations

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

MODEL_SPECS = {
    "bert": ("bert-base-uncased", False),
    "gpt2": ("gpt2", False),
    "qwen2.5": ("Qwen/Qwen2.5-7B", True),
    "qwen2.5-math": ("Qwen/Qwen2.5-Math-7B", True),
}


def load_model(name: str):
    """Load one model at a time so Colab does not need to hold all four in memory."""
    if name not in MODEL_SPECS:
        raise KeyError(f"Unknown model {name!r}. Choose from {list(MODEL_SPECS)}")
    model_id, causal = MODEL_SPECS[name]
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    cls = AutoModelForCausalLM if causal else AutoModel
    kwargs = {"output_hidden_states": True}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        if causal:
            kwargs["torch_dtype"] = torch.float16
    model = cls.from_pretrained(model_id, **kwargs)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token
    model.eval()
    return model, tokenizer
