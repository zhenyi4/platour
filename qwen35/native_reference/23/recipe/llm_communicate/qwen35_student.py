"""Text-only view of the official multimodal checkpoint, preserving its weights."""
import torch
from transformers import AutoModelForImageTextToText, Qwen3_5ForCausalLM


def load_student(path):
    full = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True)
    with torch.device('meta'):
        model = Qwen3_5ForCausalLM(full.config.text_config)
    model.model = full.model.language_model
    model.lm_head = full.lm_head
    model.generation_config = full.generation_config
    assert not any(p.is_meta for p in model.parameters())
    return model.to('cuda').eval().requires_grad_(False)
