"""Qwen3.5 HF hybrid GDN/KV memory handoff to native vLLM 0.17.1."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re

import safetensors
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, SupportsHMA,
)
from vllm.v1.kv_cache_interface import MambaSpec

FORMAT = 'qwen35-hf-bridge-hybrid-memory-v1'


def memory_key(ids, checkpoint, mode):
    return hashlib.sha256(json.dumps([ids, checkpoint, mode],
        separators=(',', ':')).encode()).hexdigest()


@dataclass
class MemoryRequest:
    key: str
    tokens: int
    blocks: tuple[list[int], ...]


@dataclass
class HybridMemoryMetadata(KVConnectorMetadata):
    requests: list[MemoryRequest] = field(default_factory=list)


class Qwen35HybridMemoryConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError('Qwen3.5 memory handoff requires TP=1')
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError('Disable prefix caching for Qwen3.5 handoff')
        if vllm_config.cache_config.mamba_cache_mode != 'none':
            raise ValueError('Use one recurrent state per request')
        self.directory = Path(self._kv_transfer_config.get_from_extra_config('storage_path', None))
        self.checkpoint = self._kv_transfer_config.get_from_extra_config('checkpoint_sha256', None)
        self.checkpoint_state = self._kv_transfer_config.get_from_extra_config('checkpoint_state_path', None)
        self.mode = self._kv_transfer_config.get_from_extra_config('bridge_mode', None)
        self.groups = kv_cache_config.kv_cache_groups if kv_cache_config else []
        self.pending = {}
        self.caches = {}

    def current_checkpoint(self):
        if self.checkpoint_state is None:
            return self.checkpoint
        return json.loads(Path(self.checkpoint_state).read_text())['checkpoint_sha256']

    def register_kv_caches(self, kv_caches):
        self.caches = kv_caches

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params or {}
        if not params.get('qwen35_bridge_memory'):
            raise ValueError('Every Qwen3.5 bridge request needs hybrid memory')
        ids = list(request.prompt_token_ids)
        count = int(params['prefix_tokens'])
        if not 0 < count == len(ids) - 1 or num_computed_tokens != 0:
            raise ValueError('Import full original prompt memory before the HF handoff token')
        checkpoint = self.current_checkpoint()
        if params.get('checkpoint_sha256') != checkpoint:
            raise ValueError('Qwen3.5 bridge checkpoint identity differs')
        key = memory_key(ids[:count], checkpoint, self.mode)
        if key != params['memory_key'] or not (self.directory / (key + '.safetensors')).is_file():
            raise FileNotFoundError('Missing Qwen3.5 hybrid prefix memory')
        return count, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens:
            self.pending[request.request_id] = request.kv_transfer_params

    def build_connector_meta(self, scheduler_output):
        result = HybridMemoryMetadata()
        for row in scheduler_output.scheduled_new_reqs:
            params = self.pending.pop(row.req_id, None)
            if params:
                result.requests.append(MemoryRequest(params['memory_key'],
                    int(params['prefix_tokens']), row.block_ids))
        if self.pending:
            raise RuntimeError('Hybrid memory allocated without scheduling its request')
        if scheduler_output.scheduled_cached_reqs.resumed_req_ids:
            raise RuntimeError('Hybrid preemption unsupported; lower max_num_seqs')
        return result

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, HybridMemoryMetadata):
            raise TypeError('Unexpected Qwen3.5 memory metadata')
        for request in metadata.requests:
            with safetensors.safe_open(self.directory / (request.key + '.safetensors'),
                                       framework='pt', device='cpu') as handle:
                info = handle.metadata() or {}
                expected = dict(format=FORMAT, checkpoint_sha256=self.current_checkpoint(),
                                bridge_mode=self.mode, prefix_tokens=str(request.tokens),
                                memory_key=request.key)
                if any(info.get(k) != v for k, v in expected.items()):
                    raise ValueError('Qwen3.5 memory provenance mismatch')
                for group_index, group in enumerate(self.groups):
                    spec = group.kv_cache_spec
                    blocks = request.blocks[group_index]
                    for name in group.layer_names:
                        match = re.search(r'\.layers\.(\d+)\.', name)
                        if match is None:
                            raise ValueError('Cannot identify receiver hybrid layer')
                        index = int(match.group(1))
                        destination = self.caches[name]
                        if isinstance(spec, MambaSpec):
                            if len(blocks) != 1:
                                raise ValueError('Expected one recurrent memory block')
                            for suffix, target in zip(('conv', 'recurrent'), destination):
                                source = handle.get_tensor(f'layer.{index}.{suffix}')
                                if source.shape != target[blocks[0]].shape or source.dtype != target.dtype:
                                    raise ValueError(f'Hybrid layer {index}/{suffix} state differs: '
                                        f'{tuple(source.shape)}/{source.dtype} vs '
                                        f'{tuple(target[blocks[0]].shape)}/{target.dtype}')
                                target[blocks[0]].copy_(source.to(target.device))
                        else:
                            if destination.ndim != 5 or destination.shape[0] != 2:
                                raise ValueError('Unsupported Qwen3.5 KV layout')
                            source = handle.get_tensor(f'layer.{index}.kv').to(destination.device)
                            block_size = destination.shape[2]
                            if (source.shape != (2, request.tokens, *destination.shape[3:]) or
                                    source.dtype != destination.dtype):
                                raise ValueError('Hybrid KV state shape/dtype differs')
                            logical = spec.block_size
                            if len(blocks) * logical < request.tokens:
                                raise ValueError('Insufficient Qwen3.5 KV blocks')
                            offsets = torch.arange(request.tokens, device=destination.device)
                            block_ids = torch.tensor(blocks, device=destination.device)
                            slots = block_ids[offsets // logical] * logical + offsets % logical
                            destination[:, slots // block_size, slots % block_size] = source

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def request_finished_all_groups(self, request, block_ids):
        return False, None
