"""Isolated prompt hidden residual -> native receiver K/V heads.

Both backbones prefill natively. A layer's projected sender hidden changes
only its cached prompt K/V, never the receiver's prompt residual/skip stream.
Final prompt token and response decode retain ordinary cross-layer propagation.
"""
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from qwen3_kv_residual_bridge import Qwen3KVResidualBridge,_store


class Qwen3LocalHiddenKVBridge(Qwen3KVResidualBridge):
    def __init__(self,student,teacher):
        super().__init__(student,teacher,mode='post_prefill',initialization='zero')
        self.maps=nn.ModuleList([nn.Linear(teacher.config.hidden_size,student.config.hidden_size,
            bias=False,device=student.device,dtype=torch.float32) for _ in self.layer_map])
        with torch.no_grad():
            for parameter in self.bridge_parameters():parameter.zero_()

    def _prefix(self,ids,mask):
        if ids.ndim!=2 or ids.shape!=mask.shape or ids.shape[1]<2:
            raise ValueError('Expected batched left-padded prompts of at least two tokens')
        if not bool(mask[:,-1].all()) or bool((mask[:,1:]<mask[:,:-1]).any()):
            raise ValueError('Prompts must use left padding with a real final token')
        positions=(mask.cumsum(-1)-1).clamp_min(0)
        prefix_ids,prefix_mask=ids[:,:-1],mask[:,:-1]
        with torch.no_grad():
            teacher=self.teacher.model(input_ids=ids,attention_mask=mask,
                position_ids=positions,use_cache=False,output_hidden_states=True)
            features=tuple(teacher.hidden_states[j][:,:-1].contiguous() for j in self.layer_map)
            del teacher
            # No injection hooks: the complete native residual/skip stream is
            # computed once and saved before any per-layer cache reconstruction.
            student=self.student.model(input_ids=prefix_ids,attention_mask=prefix_mask,
                position_ids=positions[:,:-1],use_cache=True,output_hidden_states=True)
            native_hidden=student.hidden_states[:-1]
            cache=student.past_key_values
            cos,sin=self.student.model.rotary_emb(native_hidden[0],positions[:,:-1])
        assert len(native_hidden)==len(self.maps)
        for i,layer in enumerate(self.student.model.layers):
            original=native_hidden[i]
            delta=self.maps[i](features[i].float()).to(original.dtype)
            local=original+delta*prefix_mask[:,:,None].to(original.dtype)
            normalized=layer.input_layernorm(local)
            attention=layer.self_attn
            shape=(*normalized.shape[:-1],-1,attention.head_dim)
            key=attention.k_norm(attention.k_proj(normalized).view(shape)).transpose(1,2)
            value=attention.v_proj(normalized).view(shape).transpose(1,2)
            # Use the exact native rotary implementation, including any kernel
            # substitution. The unused Q result does not affect the residual path.
            _,key=apply_rotary_pos_emb(torch.zeros_like(key),key,cos,sin)
            valid=prefix_mask[:,None,:,None].bool()
            native=cache.layers[i]
            key=torch.where(valid,key,native.keys)
            value=torch.where(valid,value,native.values)
            _store(cache,i,key,value)
        return cache

    def metadata(self):
        metadata=super().metadata()
        metadata.update(bridge_type='qwen3_local_hidden_to_kv',transfer_mode='hidden_local_kv',
            formula='local_h = h_s_native + P(h_t_native); K/V = native_student_KV_heads(input_RMSNorm(local_h))',
            hidden_space='native decoder block input before RMSNorm',
            projection=dict(type='linear',input_dim=self.teacher.config.hidden_size,
                output_dim=self.student.config.hidden_size,intermediate_dim=None,activation=None,bias=False),
            kv_reconstruction='native student input RMSNorm, k_proj/v_proj, K-norm and RoPE',
            native_prompt_residual_stream=True,prefix_propagation=False,decode_propagation=True,
            extra_trainable_kv_projection=False,constant_gate=1.)
        return metadata
