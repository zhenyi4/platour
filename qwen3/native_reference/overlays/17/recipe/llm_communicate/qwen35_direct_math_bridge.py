"""Native Qwen3.5 text models and the existing zero-initialized linear bridge."""
import importlib.metadata

import torch

from direct_residual_chat_bridge import DirectResidualChatBridge
from qwen35_student import load_student


def accelerated_models():
    # Bind the installed, pinned FLA kernels directly. This does not require
    # remote kernel downloads and affects only this isolated training process.
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    modeling.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    modeling.torch_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
    return dict(transformers=importlib.metadata.version('transformers'), torch=torch.__version__,
                fla=importlib.metadata.version('fla-core'), linear_prefill='FLA chunk_gated_delta_rule',
                linear_decode='FLA fused_recurrent_gated_delta_rule', convolution='PyTorch CUDA conv1d',
                attention='PyTorch SDPA', dtype='bfloat16', quantization=None)


def load_bridge(root):
    runtime = accelerated_models()
    student = load_student(root / 'data/models/Qwen3.5-4B')
    teacher = load_student(root / 'data/models/Qwen3.5-9B')
    assert student.config.num_hidden_layers == teacher.config.num_hidden_layers == 32
    assert student.config.layer_types == teacher.config.layer_types
    assert student.config.hidden_size == 2560 and teacher.config.hidden_size == 4096
    bridge = DirectResidualChatBridge(student, teacher, variant='teacher_residual',
                                     initialization='zero', lm_chunk_tokens=128).eval()
    assert bridge.layer_map == list(range(32))
    assert sum(p.numel() for p in bridge.bridge_parameters()) == 335544320
    assert all(torch.count_nonzero(p) == 0 for p in bridge.bridge_parameters())
    return bridge, runtime


@torch.no_grad()
def cache_check(bridge, prompt):
    """Check trained bridge prefill versus teacher forcing and native generation."""
    ids = torch.tensor([prompt], device=bridge.student.device)
    mask = torch.ones_like(ids)
    teacher = bridge.teacher.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                  output_hidden_states=True).hidden_states
    features = bridge.teacher_features(teacher)
    inject = mask.bool()
    inject[:, -1] = False

    def prefill(tokens):
        with bridge.inject(features, inject):
            return bridge.student(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                  use_cache=True, logits_to_keep=1)

    a = prefill(ids)
    nxt = a.logits[:, -1].argmax(-1, keepdim=True)
    extended = torch.cat([ids, nxt], dim=1)
    cached = bridge.student(input_ids=nxt, attention_mask=torch.ones_like(extended),
                            past_key_values=a.past_key_values, use_cache=True, logits_to_keep=1)
    full = prefill(extended)
    error = (cached.logits.float() - full.logits.float()).abs()
    # Native generate and our decoder must use exactly the same cache semantics.
    fresh = prefill(ids)
    native = bridge.student.generate(input_ids=extended, attention_mask=torch.ones_like(extended),
        past_key_values=fresh.past_key_values, max_new_tokens=1, do_sample=False,
        return_dict_in_generate=True, output_logits=True)
    native_error = (native.logits[0].float() - cached.logits[:, -1].float()).abs()
    report = dict(native_hybrid_cache=True, cache_type=type(cached.past_key_values).__name__,
        response_injection=False, last_prompt_token_injected=False,
        mean_abs_logit_error=float(error.mean()), max_abs_logit_error=float(error.max()),
        native_generate_max_abs_logit_error=float(native_error.max()),
        native_generate_greedy_match=bool(torch.equal(native.sequences[:, -1], cached.logits[:, -1].argmax(-1))))
    assert report['mean_abs_logit_error'] < .05 and report['max_abs_logit_error'] <= .5, report
    assert report['native_generate_greedy_match'] and report['native_generate_max_abs_logit_error'] < 1e-5, report
    return report
