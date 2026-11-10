"""Read-only HF bridge memory handoff for vLLM 0.17.1 (TP=1, GDN + KV).

The full original HF prompt memory and its first answer token are retained;
vLLM computes subsequent answer tokens. Prefix caching/preemption are not
supported: a resumed request must not silently fall back to native prefill.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import safetensors
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, SupportsHMA,
)
from vllm.v1.kv_cache_interface import MambaSpec

FORMAT = 'qwen35-hf-bridge-hybrid-memory-v1'


def memory_key(ids, checkpoint_sha, mode):
    return hashlib.sha256(json.dumps([ids, checkpoint_sha, mode],
        separators=(',', ':')).encode()).hexdigest()


@dataclass
class MemoryRequest:
    key: str
    tokens: int
    blocks: tuple[list[int], ...]
    checkpoint: str | None = None


@dataclass
class HybridMemoryMetadata(KVConnectorMetadata):
    requests: list[MemoryRequest] = field(default_factory=list)


class Qwen35HybridMemoryConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError('This memory format requires tensor_parallel_size=1')
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError('Disable prefix caching for bridge memory handoff')
        if vllm_config.cache_config.mamba_cache_mode != 'none':
            raise ValueError('Use mamba_cache_mode=none for one state per request')
        self.directory = Path(self._kv_transfer_config.get_from_extra_config('storage_path',None))
        self.checkpoint = self._kv_transfer_config.get_from_extra_config('checkpoint_sha256',None)
        self.mode = self._kv_transfer_config.get_from_extra_config('bridge_mode',None)
        self.dynamic = self._kv_transfer_config.get_from_extra_config('allow_dynamic_checkpoint',False)
        self.groups = kv_cache_config.kv_cache_groups if kv_cache_config else []
        self.pending = {}
        self.caches = {}

    def register_kv_caches(self, kv_caches):
        self.caches = kv_caches

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params or {}
        if not params.get('qwen35_bridge_memory'):
            raise ValueError('Bridge engine requires memory parameters on every request')
        ids = list(request.prompt_token_ids)
        n = int(params['prefix_tokens'])
        if not (0 < n < len(ids)) or num_computed_tokens != 0:
            raise ValueError('Memory prefix must precede uncached tokens, without prefix hits')
        checkpoint = params.get('checkpoint_sha256', self.checkpoint)
        if getattr(self, 'dynamic', False):
            if not isinstance(checkpoint, str) or not re.fullmatch(r'[0-9a-f]{64}', checkpoint):
                raise ValueError('Dynamic bridge requests require a SHA256 weight identity')
        elif checkpoint != self.checkpoint:
            raise ValueError('Bridge request checkpoint differs from the fixed engine')
        key = memory_key(ids[:n], checkpoint, self.mode)
        if key != params['memory_key'] or not (self.directory / (key+'.safetensors')).is_file():
            raise FileNotFoundError(f'Missing or mismatched bridge memory: {key}')
        return n, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens:
            self.pending[request.request_id] = request.kv_transfer_params

    def build_connector_meta(self, scheduler_output):
        result = HybridMemoryMetadata()
        for row in scheduler_output.scheduled_new_reqs:
            params = self.pending.pop(row.req_id, None)
            if params:
                result.requests.append(MemoryRequest(params['memory_key'],
                    int(params['prefix_tokens']), row.block_ids,
                    params.get('checkpoint_sha256', self.checkpoint)))
        if self.pending:
            raise RuntimeError('Bridge memory allocated but request was not scheduled')
        if scheduler_output.scheduled_cached_reqs.resumed_req_ids:
            raise RuntimeError('Bridge memory preemption is unsupported; reduce max_num_seqs')
        return result

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, HybridMemoryMetadata):
            raise TypeError(type(metadata))
        for request in metadata.requests:
            with safetensors.safe_open(self.directory / (request.key+'.safetensors'),
                    framework='pt', device='cpu') as handle:
                info = handle.metadata()
                expected = dict(format=FORMAT, checkpoint_sha256=request.checkpoint or self.checkpoint,
                    bridge_mode=self.mode, prefix_tokens=str(request.tokens), memory_key=request.key)
                if any(info.get(k) != v for k,v in expected.items()):
                    raise ValueError(f'Bridge memory provenance mismatch: {info}')
                for group_index, group in enumerate(self.groups):
                    spec = group.kv_cache_spec
                    blocks = request.blocks[group_index]
                    for name in group.layer_names:
                        index = int(re.search(r'\.layers\.(\d+)\.', name).group(1))
                        destination = self.caches[name]
                        if isinstance(spec, MambaSpec):
                            if len(blocks) != 1:
                                raise ValueError(f'Expected one recurrent memory block, got {blocks}')
                            for suffix, target in zip(('conv','recurrent'), destination):
                                source = handle.get_tensor(f'layer.{index}.{suffix}')
                                if source.shape != target[blocks[0]].shape or source.dtype != target.dtype:
                                    raise ValueError(f'{name}/{suffix}: incompatible state shape/dtype')
                                target[blocks[0]].copy_(source.to(target.device))
                        else:
                            # vLLM keeps logical [2, blocks, block_size, heads, dim]
                            # even when its physical hybrid-cache strides interleave K/V.
                            if destination.ndim != 5 or destination.shape[0] != 2:
                                raise ValueError(f'Unsupported KV layout: {destination.shape}')
                            source = handle.get_tensor(f'layer.{index}.kv').to(destination.device)
                            block_size = destination.shape[2]
                            if source.shape != (2,request.tokens,*destination.shape[3:]) or source.dtype!=destination.dtype:
                                raise ValueError(f'{name}: incompatible KV shape/dtype')
                            # Do not reshape a strided cache: it could produce a copy.
                            logical_size = spec.block_size
                            if len(blocks)*logical_size < request.tokens:
                                raise ValueError('Insufficient full-attention blocks for imported prefix')
                            offsets = torch.arange(request.tokens,device=destination.device)
                            block_ids = torch.tensor(blocks,device=destination.device)
                            slots = block_ids[offsets//logical_size]*logical_size+offsets%logical_size
                            destination[:,slots//block_size,slots%block_size] = source

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def request_finished_all_groups(self, request, block_ids):
        return False, None
