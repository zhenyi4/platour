"""Read-only HF bridge memory handoff for vLLM 0.17.1 (TP=1, GDN + KV).

The full original HF prompt memory and its first answer token are retained;
vLLM computes subsequent answer tokens. Prefix caching is unsupported.
Evicted requests reload their original HF prefix into fresh blocks before
vLLM recomputes the generated-token history; native prefill fallback is forbidden.
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
        restored = []
        def add(request_id, blocks, computed, resumed):
            if request_id not in self.pending:
                raise RuntimeError(f'Scheduled bridge request lacks allocated prefix: {request_id}, resumed={resumed}')
            params = self.pending[request_id]
            prefix = int(params['prefix_tokens'])
            if computed != prefix or not blocks or any(not ids for ids in blocks):
                raise RuntimeError(f'Invalid prefix restoration layout: {request_id}, computed={computed}, prefix={prefix}')
            if scheduler_output.num_scheduled_tokens.get(request_id, 0) <= 0:
                raise RuntimeError(f'Prefix restoration is not backed by scheduled compute: {request_id}')
            result.requests.append(MemoryRequest(params['memory_key'],prefix,blocks,
                                                 params.get('checkpoint_sha256',self.checkpoint)))
            restored.append(dict(id=request_id,resumed=resumed,prefix=prefix,blocks=blocks,
                                 memory_key=params['memory_key']))
            del self.pending[request_id]
        for row in scheduler_output.scheduled_new_reqs:
            add(row.req_id,row.block_ids,row.num_computed_tokens,False)
        cached = scheduler_output.scheduled_cached_reqs
        if len(cached.req_ids)!=len(cached.new_block_ids) or len(cached.req_ids)!=len(cached.num_computed_tokens):
            raise RuntimeError('Cached request layout arrays disagree')
        for index,request_id in enumerate(cached.req_ids):
            if request_id in cached.resumed_req_ids:
                params = self.pending.get(request_id)
                history = cached.all_token_ids.get(request_id)
                if params is None or history is None or len(history)<=int(params['prefix_tokens']):
                    raise RuntimeError(f'Resumed request lacks complete prefix/token history: {request_id}')
                add(request_id,cached.new_block_ids[index],cached.num_computed_tokens[index],True)
        if set(cached.resumed_req_ids)-set(cached.req_ids):
            raise RuntimeError('Resumed request IDs are not in cached schedule')
        if self.pending:
            raise RuntimeError(f'Bridge memory allocated but request was not scheduled: {list(self.pending)}')
        if any(event['resumed'] for event in restored):
            path = self.directory.parent/'prefix_restore_events.jsonl'
            with path.open('a') as stream:
                stream.write(json.dumps(dict(restored=restored,pending=len(self.pending)))+'\n')
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
