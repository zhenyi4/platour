"""Linear residual KV transfer for full-attention Qwen3 models.

K/V are the actual attention cache tensors: K is after K-norm and RoPE,
V is after v_proj. Q, attention weights and backbone parameters stay native.
Two timings share the identical projections and prompt-only transfer rule.
"""
from contextlib import nullcontext

import torch
from torch import nn
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.modeling_outputs import CausalLMOutputWithPast

from hidden_bridge import bridge_layer_map


def _store(cache, index, key, value):
    # Replace rather than append: activation-checkpoint replay must be idempotent.
    layer = DynamicLayer()
    layer.update(key, value)
    cache.layers[index] = layer
    return layer.keys, layer.values


class _TransferPrefixCache(DynamicCache):
    def __init__(self, bridge, features, mask):
        super().__init__(config=bridge.student.config)
        self.bridge, self.features, self.mask = bridge, features, mask

    def update(self, key, value, layer_idx, *args, **kwargs):
        key, value = self.bridge.fuse(layer_idx, key, value,
                                     self.features[layer_idx], self.mask)
        return _store(self, layer_idx, key, value)


class _FunctionalTailCache(DynamicCache):
    """One teacher-forced tail forward; safe to replay blocks in backward."""
    def __init__(self, config, prefix):
        super().__init__(config=config)
        self.prefix = tuple((layer.keys, layer.values) for layer in prefix.layers)
        for i, (key, value) in enumerate(self.prefix):
            _store(self, i, key, value)

    def update(self, key, value, layer_idx, *args, **kwargs):
        past_key, past_value = self.prefix[layer_idx]
        return _store(self, layer_idx, torch.cat((past_key, key), -2),
                      torch.cat((past_value, value), -2))


class Qwen3KVResidualBridge(nn.Module):
    MODES = ('sequential', 'post_prefill')

    def __init__(self, student, teacher, mode='sequential', initialization='zero'):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f'Unknown KV transfer mode: {mode}')
        for model in (student, teacher):
            if model.config.model_type not in ('qwen3', 'qwen3_moe'):
                raise ValueError('KV residual bridge supports full-attention Qwen3 only')
            if getattr(model.config, 'use_sliding_window', False):
                raise ValueError('Sliding-window attention is not supported')
            model.eval().requires_grad_(False)
        self.student, self.teacher, self.mode = student, teacher, mode
        self.initialization = initialization
        self.student_activation_checkpointing = False
        self.layer_map = bridge_layer_map(student.config.num_hidden_layers,
                                          teacher.config.num_hidden_layers, 'uniform_endpoints')
        self.teacher_kv_width = teacher.config.num_key_value_heads * teacher.config.head_dim
        self.student_kv_width = student.config.num_key_value_heads * student.config.head_dim
        self.maps = nn.ModuleList([
            nn.ModuleDict({kind: nn.Linear(self.teacher_kv_width, self.student_kv_width,
                                          bias=False, device=student.device, dtype=torch.float32)
                           for kind in ('key', 'value')}) for _ in self.layer_map])
        if initialization != 'zero':
            raise ValueError('Construct with zero, then use the shared GRPO initializer')
        with torch.no_grad():
            for parameter in self.bridge_parameters():
                parameter.zero_()

    def bridge_parameters(self):
        return tuple(self.maps.parameters())

    def metadata(self):
        return dict(bridge_type='qwen3_kv_linear_residual', transfer_mode=self.mode,
                    formula='K_s + P_K(K_t), V_s + P_V(V_t)', layer_map=self.layer_map,
                    teacher_kv_width=self.teacher_kv_width, student_kv_width=self.student_kv_width,
                    key_space='post K-norm and post RoPE', value_space='post v_proj',
                    injection='non-padding prompt[:-1] only', query_transfer=False,
                    frozen_backbones=True, initialization=self.initialization,
                    trainable_parameters=sum(p.numel() for p in self.bridge_parameters()))

    def fuse(self, index, key, value, features, mask):
        result = []
        for kind, native, source in zip(('key', 'value'), (key, value), features, strict=True):
            # Mix KV heads and head channels using one bias-free linear per K/V.
            batch, _, length, _ = source.shape
            flat = source.transpose(1, 2).reshape(batch, length, self.teacher_kv_width)
            delta = self.maps[index][kind](flat.float()).to(native.dtype)
            delta = delta.view(batch, length, native.shape[1], native.shape[3]).transpose(1, 2)
            result.append(native + delta * mask[:, None, :, None].to(native.dtype))
        return tuple(result)

    def _checkpoint_context(self):
        if torch.is_grad_enabled() and self.student_activation_checkpointing:
            from qwen35_bridge_grpo_core import checkpoint_student_blocks
            return checkpoint_student_blocks(self)
        return nullcontext()

    def _prefix(self, ids, mask):
        if ids.ndim != 2 or ids.shape != mask.shape or ids.shape[1] < 2:
            raise ValueError('Expected batched, left-padded prompts of at least two tokens')
        if not bool(mask[:, -1].all()) or bool((mask[:, 1:] < mask[:, :-1]).any()):
            raise ValueError('Prompts must use left padding with a real final token')
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        with torch.no_grad():
            output = self.teacher.model(input_ids=ids, attention_mask=mask,
                position_ids=positions, use_cache=True)
            features = tuple((output.past_key_values.layers[j].keys[:, :, :-1].contiguous(),
                              output.past_key_values.layers[j].values[:, :, :-1].contiguous())
                             for j in self.layer_map)
            del output
        prefix_ids, prefix_mask = ids[:, :-1], mask[:, :-1]
        options = dict(input_ids=prefix_ids, attention_mask=prefix_mask,
                       position_ids=positions[:, :-1], use_cache=True)
        if self.mode == 'sequential':
            cache = _TransferPrefixCache(self, features, prefix_mask)
            with self._checkpoint_context():
                output = self.student.model(**options, past_key_values=cache)
            # Return a plain, appendable cache. Transfer ends with this prefix.
            native = DynamicCache(config=self.student.config)
            for i, layer in enumerate(output.past_key_values.layers):
                _store(native, i, layer.keys, layer.values)
        else:
            with torch.no_grad():
                native = self.student.model(**options).past_key_values
            for i, layer in enumerate(native.layers):
                key, value = self.fuse(i, layer.keys, layer.values, features[i], prefix_mask)
                _store(native, i, key, value)
        return native

    def prefill(self, ids, mask, extra=None):
        cache = self._prefix(ids, mask)
        tail = ids[:, -1:]
        if extra is not None:
            tail = torch.cat((tail, extra), 1)
            mask = torch.cat((mask, torch.ones_like(extra)), 1)
        positions = (mask.cumsum(-1) - 1).clamp_min(0)[:, -tail.shape[1]:]
        output = self.student(input_ids=tail, attention_mask=mask, position_ids=positions,
                              past_key_values=cache, use_cache=True, logits_to_keep=1)
        # All subsequent decode uses the untouched receiver, including vLLM.
        return CausalLMOutputWithPast(logits=output.logits, past_key_values=output.past_key_values)

    def response_hidden(self, prompt, response):
        """Differentiable teacher-forced response, matching rollout prefill exactly."""
        ids = torch.tensor([prompt], device=self.student.device)
        cache = self._prefix(ids, torch.ones_like(ids))
        tail = torch.tensor([[prompt[-1]] + response[:-1]], device=ids.device)
        mask = torch.ones((1, len(prompt) - 1 + tail.shape[1]), device=ids.device, dtype=torch.long)
        positions = torch.arange(len(prompt) - 1, mask.shape[1], device=ids.device)[None]
        cache = _FunctionalTailCache(self.student.config, cache)
        with self._checkpoint_context():
            return self.student.model(input_ids=tail, attention_mask=mask,
                position_ids=positions, past_key_values=cache, use_cache=True).last_hidden_state
