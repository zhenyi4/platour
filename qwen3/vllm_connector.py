"""Qwen3 dense prefix-KV import for native vLLM 0.17.1 decoding (TP=1)."""

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

FORMAT = 'platour-hf-prefix-v1'


def memory_key(tokens, checkpoint):
    return hashlib.sha256(json.dumps([tokens, checkpoint],
                                     separators=(',', ':')).encode()).hexdigest()


@dataclass
class PrefixRequest:
    key: str
    tokens: int
    blocks: tuple[list[int], ...]


@dataclass
class PrefixMetadata(KVConnectorMetadata):
    requests: list[PrefixRequest] = field(default_factory=list)


class PLatourPrefixConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError('Qwen3 prefix import requires receiver TP=1')
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError('Disable vLLM prefix caching for Qwen3 bridge memory')
        self.directory = Path(self._kv_transfer_config.get_from_extra_config('storage_path', None))
        self.checkpoint = self._kv_transfer_config.get_from_extra_config('checkpoint_sha256', None)
        self.groups = kv_cache_config.kv_cache_groups if kv_cache_config else []
        self.pending = {}
        self.caches = {}

    def register_kv_caches(self, kv_caches):
        self.caches = kv_caches

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params or {}
        if not params.get('platour_prefix_memory'):
            raise ValueError('Every Qwen3 bridge request must supply its prefix KV')
        ids = list(request.prompt_token_ids)
        n = int(params['prefix_tokens'])
        if not 0 < n == len(ids) - 1 or num_computed_tokens != 0:
            raise ValueError('Import prompt[:-1] exactly; final prompt token remains native')
        if params.get('checkpoint_sha256') != self.checkpoint:
            raise ValueError('Checkpoint identity differs from the loaded Qwen3 bridge')
        key = memory_key(ids[:n], self.checkpoint)
        if key != params['memory_key'] or not (self.directory / (key + '.safetensors')).is_file():
            raise FileNotFoundError('Missing or mismatched Qwen3 bridge prefix')
        return n, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens:
            self.pending[request.request_id] = request.kv_transfer_params

    def build_connector_meta(self, scheduler_output):
        result = PrefixMetadata()
        for row in scheduler_output.scheduled_new_reqs:
            params = self.pending.pop(row.req_id, None)
            if params:
                result.requests.append(PrefixRequest(params['memory_key'],
                    int(params['prefix_tokens']), row.block_ids))
        if self.pending:
            raise RuntimeError('Prefix allocated without scheduling its request')
        if scheduler_output.scheduled_cached_reqs.resumed_req_ids:
            raise RuntimeError('Prefix preemption unsupported; reduce batch size')
        return result

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, PrefixMetadata):
            raise TypeError('Unexpected Qwen3 prefix metadata')
        for request in metadata.requests:
            with safetensors.safe_open(self.directory / (request.key + '.safetensors'),
                                       framework='pt', device='cpu') as handle:
                info = handle.metadata() or {}
                expected = dict(format=FORMAT, checkpoint_sha256=self.checkpoint,
                                prefix_tokens=str(request.tokens), memory_key=request.key)
                if any(info.get(k) != value for k, value in expected.items()):
                    raise ValueError('Qwen3 prefix provenance mismatch')
                for group_index, group in enumerate(self.groups):
                    blocks = request.blocks[group_index]
                    for name in group.layer_names:
                        match = re.search(r'\.layers\.(\d+)\.', name)
                        if match is None:
                            raise ValueError('Cannot identify receiver layer')
                        destination = self.caches[name]
                        if (not isinstance(destination, torch.Tensor) or
                                destination.ndim != 5 or destination.shape[0] != 2):
                            raise ValueError('Only dense full-attention Qwen3 KV is supported')
                        source = handle.get_tensor(f'layer.{match.group(1)}.kv').to(destination.device)
                        if (source.shape != (2, request.tokens, *destination.shape[3:]) or
                                source.dtype != destination.dtype):
                            raise ValueError('Qwen3 prefix tensor shape/dtype mismatch')
                        logical_size = group.kv_cache_spec.block_size
                        if len(blocks) * logical_size < request.tokens:
                            raise ValueError('Insufficient cache blocks')
                        offsets = torch.arange(request.tokens, device=destination.device)
                        block_ids = torch.tensor(blocks, device=destination.device)
                        slots = block_ids[offsets // logical_size] * logical_size + offsets % logical_size
                        physical_size = destination.shape[2]
                        destination[:, slots // physical_size, slots % physical_size] = source

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def request_finished_all_groups(self, request, block_ids):
        return False, None
