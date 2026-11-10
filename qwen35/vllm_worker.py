"""Qwen3.5 vLLM 0.17.1 worker for bridge, virtual, full and LoRA evaluation."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import re
import time
import traceback

from .checkpoints import atomic_json


def register_gdn_lora():
    from vllm.model_executor.models.qwen3_5 import (Qwen3_5ForCausalLMBase,
                                                      Qwen3_5ForConditionalGeneration)
    for cls in (Qwen3_5ForCausalLMBase, Qwen3_5ForConditionalGeneration):
        cls.packed_modules_mapping = dict(cls.packed_modules_mapping,
            in_proj_qkvz=['in_proj_q', 'in_proj_k', 'in_proj_v', 'in_proj_z'])


if os.environ.get('PLATOUR_NATIVE_LORA_GDN_WORKER') == '1':
    register_gdn_lora()


class PublicWeights:
    def load_full(self, path, expected_sha):
        from safetensors import safe_open
        from hashlib import sha256
        digest = sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected_sha:
            raise ValueError('Full model SHA-256 differs from the released checkpoint')
        with safe_open(path, framework='pt', device='cpu') as source:
            names = list(source.keys())
            if len(names) < 200 or not all(name.startswith(('model.', 'lm_head.')) for name in names):
                raise ValueError('Full Qwen3.5 text model weight keys differ')
            weights = ((('model.language_model.' + name[6:]) if name.startswith('model.') else name,
                        source.get_tensor(name)) for name in names)
            model = self.model_runner.model
            loaded = model.load_weights(weights)
            if len(loaded) < 100:
                raise ValueError('Native vLLM model did not consume the published full weights')
            import torch
            language = getattr(model, 'language_model', model)
            actual = dict(language.named_parameters())
            fused = {'q_proj': ('qkv_proj', 0), 'k_proj': ('qkv_proj', 1),
                     'v_proj': ('qkv_proj', 2), 'gate_proj': ('gate_up_proj', 0),
                     'up_proj': ('gate_up_proj', 1), 'in_proj_qkv': ('in_proj_qkvz', 0),
                     'in_proj_z': ('in_proj_qkvz', 1), 'in_proj_b': ('in_proj_ba', 0),
                     'in_proj_a': ('in_proj_ba', 1)}
            parts = {'qkv_proj': ('q_proj', 'k_proj', 'v_proj'),
                     'gate_up_proj': ('gate_proj', 'up_proj'),
                     'in_proj_qkvz': ('in_proj_qkv', 'in_proj_z'),
                     'in_proj_ba': ('in_proj_b', 'in_proj_a')}
            for name in names:
                value = source.get_tensor(name)
                if name == 'lm_head.weight' and language.config.tie_word_embeddings:
                    target = language.model.embed_tokens.weight
                elif name in actual:
                    target = actual[name]
                else:
                    stem, leaf = name.rsplit('.', 1)
                    module = stem.rsplit('.', 1)[-1]
                    if module not in fused:
                        raise ValueError(f'Unexpected native Qwen3.5 tensor: {name}')
                    packed, slot = fused[module]
                    parent = stem.rsplit('.', 1)[0]
                    target = actual[parent + '.' + packed + '.' + leaf]
                    offsets = [source.get_slice(parent + '.' + part + '.' + leaf).get_shape()[0]
                               for part in parts[packed]]
                    target = target.narrow(0, sum(offsets[:slot]), offsets[slot])
                if target.shape != value.shape or not torch.equal(target.detach().cpu(), value):
                    raise ValueError(f'Native Qwen3.5 full tensor differs after fusion: {name}')
        return {'full_model_sha256': expected_sha, 'source_tensors': len(names),
                'native_loaded_tensors': len(loaded), 'tensor_equality_audited': True}

    def load_and_audit_lora(self, path, adapter_dir, expected_sha):
        import hashlib
        import torch
        from safetensors import safe_open
        from vllm.lora.request import LoRARequest
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected_sha:
            raise ValueError('LoRA release SHA differs')
        self.add_lora(LoRARequest('published', 1, adapter_dir))
        manager = self.model_runner.lora_manager._adapter_manager
        adapter = manager.get_adapter(1)
        if adapter is None:
            raise RuntimeError('Native vLLM did not load the LoRA adapter')
        mapping = {'q_proj': ('qkv_proj', 0), 'k_proj': ('qkv_proj', 1),
                   'v_proj': ('qkv_proj', 2), 'gate_proj': ('gate_up_proj', 0),
                   'up_proj': ('gate_up_proj', 1), 'in_proj_qkv': ('in_proj_qkvz', 0),
                   'in_proj_z': ('in_proj_qkvz', 3), 'in_proj_b': ('in_proj_ba', 0),
                   'in_proj_a': ('in_proj_ba', 1)}
        verified = 0
        with safe_open(path, framework='pt', device='cpu') as source:
            names = list(source.keys())
            if len(names) != 496:
                raise ValueError('Original Qwen3.5 LoRA matrix count differs')
            from .hf_export import custom_lora_name
            for key in names:
                name = key if key.startswith('model.layers.') else custom_lora_name(key)
                value = source.get_tensor(key)
                stem, factor = name.rsplit('.', 1)
                stem = 'language_model.' + stem
                if stem.endswith('.in_proj_qkv'):
                    layer = adapter.loras[stem[:-len('in_proj_qkv')] + 'in_proj_qkvz']
                    values = ([value] * 3 if factor == 'lora_A'
                              else list(value.split([2048, 2048, 4096], dim=0)))
                    loaded = layer.lora_a if factor == 'lora_A' else layer.lora_b
                    if not all(tensor is not None and tensor.shape == expected.shape and
                               torch.equal(tensor.cpu(), expected.to(tensor.dtype))
                               for tensor, expected in zip(loaded[:3], values, strict=True)):
                        raise ValueError(f'Fused GDN LoRA differs: {name}')
                else:
                    if stem in adapter.loras:
                        layer = adapter.loras[stem]
                        tensor = layer.lora_a if factor == 'lora_A' else layer.lora_b
                    else:
                        parent, module = stem.rsplit('.', 1)
                        packed, slot = mapping[module]
                        layer = adapter.loras[parent + '.' + packed]
                        tensor = (layer.lora_a if factor == 'lora_A' else layer.lora_b)[slot]
                    if tensor is None or tensor.shape != value.shape or not torch.equal(
                            tensor.cpu(), value.to(tensor.dtype)):
                        raise ValueError(f'Native LoRA factor differs: {name}')
                verified += 1
        return {'lora_sha256': expected_sha, 'verified_tensors': verified,
                'tensor_equality_audited': True}


def untied_full_config(config):
    """Preserve both independently trained tensors when building the decoder."""
    config.tie_word_embeddings = False
    if getattr(config, 'text_config', None) is not None:
        config.text_config.tie_word_embeddings = False
    return config


def response_row(row, output, method, cap, tokenizer, logprobs=False):
    """Preserve the hybrid first token and exactly one log-probability per response token."""
    if row.get('handoff_is_eos') or row.get('handoff_only'):
        ids = [row['first_token']]
        finish = 'stop' if row.get('handoff_is_eos') else 'length'
        values = [row.get('first_log_prob')] if logprobs else None
    else:
        expected = row['prompt_token_ids'] + ([row['first_token']] if method in ('bridge', 'random') else [])
        if list(output.prompt_token_ids) != expected:
            raise ValueError('vLLM changed the signed prompt token IDs')
        answer = output.outputs[0]
        ids = ([row['first_token']] if method in ('bridge', 'random') else []) + list(answer.token_ids)
        finish, values = answer.finish_reason, None
        if logprobs:
            if answer.logprobs is None or len(answer.logprobs) != len(answer.token_ids):
                raise RuntimeError('Missing vLLM rollout log-probabilities')
            values = ([row['first_log_prob']] if method in ('bridge', 'random') else []) + [
                float(scores[token].logprob) for token, scores in
                zip(answer.token_ids, answer.logprobs, strict=True)]
    if not ids or len(ids) > cap:
        raise ValueError('Invalid Qwen3.5 response length')
    result = dict(index=row['index'], sample_index=row['sample_index'], sampling_seed=row['sampling_seed'],
                  response_token_ids=ids, response=tokenizer.decode(ids, skip_special_tokens=True),
                  tokens=len(ids), finish_reason=finish)
    if logprobs:
        result['log_probs'] = values
        from training_protocol import rollout_log_probs
        rollout_log_probs(result, cap)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    cfg = json.loads((root / 'config.json').read_text())
    version = importlib.metadata.version('vllm')
    if version != '0.17.1':
        raise RuntimeError(f'Qwen3.5 hybrid memory requires vLLM 0.17.1, got {version}')
    method = cfg['method']
    persistent = cfg.get('persistent', False)
    if persistent and method not in ('bridge', 'random'):
        raise ValueError('Persistent decoding requires a frozen receiver')
    checkpoint_state = root / 'active-checkpoint.json'
    if persistent:
        atomic_json(checkpoint_state, dict(checkpoint_sha256=cfg['checkpoint_sha256']))
    if method == 'lora':
        os.environ['PLATOUR_NATIVE_LORA_GDN_WORKER'] = '1'
        register_gdn_lora()
    from vllm import LLM, SamplingParams
    shared = dict(model=cfg['model'], dtype='bfloat16', tensor_parallel_size=1,
        max_model_len=cfg['max_model_len'], max_num_seqs=cfg['max_num_seqs'],
        max_num_batched_tokens=4096, gpu_memory_utilization=cfg['memory_fraction'],
        generation_config='vllm', seed=42, disable_log_stats=True,
        language_model_only=True, limit_mm_per_prompt={'image': 0, 'video': 0},
        attention_config={'backend': 'FLASH_ATTN'}, enable_prefix_caching=False,
        mamba_cache_mode='none', mamba_ssm_cache_dtype='float32',
        disable_hybrid_kv_cache_manager=False)
    if method in ('bridge', 'random'):
        shared.update(logits_processors=['qwen35.vllm_logits:HandoffPresencePenalty'],
            kv_transfer_config=dict(kv_connector='Qwen35HybridMemoryConnector',
                kv_connector_module_path='qwen35.vllm_connector', kv_role='kv_consumer',
                kv_connector_extra_config=dict(storage_path=str(root / 'memory'),
                    checkpoint_sha256=cfg['checkpoint_sha256'], bridge_mode='full_hidden')))
    else:
        shared['worker_extension_cls'] = 'qwen35.vllm_worker.PublicWeights'
        if method == 'full':
            # Load every released tensor through the audited RPC below; the
            # tied upstream base may legitimately omit a separate output head.
            shared.update(hf_overrides=untied_full_config, load_format='dummy')
        if method == 'lora':
            shared.update(enable_lora=True, max_lora_rank=128, max_loras=1,
                          max_cpu_loras=1, lora_dtype='bfloat16')
    if persistent:
        shared['enable_sleep_mode'] = True
        shared['kv_transfer_config']['kv_connector_extra_config']['checkpoint_state_path'] = str(checkpoint_state)
    llm = LLM(**shared)
    sleeping = False
    from vllm.lora.request import LoRARequest
    lora_request = None
    audit = None
    if method == 'full':
        reports = llm.collective_rpc('load_full', args=(cfg['checkpoint'],
                                                        cfg['checkpoint_sha256']))
        if not reports or not all(report['full_model_sha256'] == cfg['checkpoint_sha256']
                                  for report in reports):
            raise RuntimeError('Full model did not load into every vLLM worker')
        audit = reports
    elif method == 'lora':
        reports = llm.collective_rpc('load_and_audit_lora', args=(
            cfg['checkpoint'], cfg['adapter_dir'], cfg['checkpoint_sha256']))
        if not reports or not all(report['verified_tensors'] == 496 for report in reports):
            raise RuntimeError('Native vLLM did not verify all 496 LoRA matrices')
        audit = reports
        lora_request = LoRARequest('published', 1, cfg['adapter_dir'])
    atomic_json(root / 'ready.json', {'complete': True, 'backend': 'vllm',
                                     'version': version, 'method': method,
                                     'weight_audit': audit, 'persistent': persistent, 'worker_pid': os.getpid()})
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('command') == 'stop':
                break
            destination = Path(request['result'])
            try:
                command = request.get('command')
                if command is not None:
                    started = time.monotonic()
                    if not persistent or llm.llm_engine.has_unfinished_requests():
                        raise RuntimeError('Worker control requires an idle persistent engine')
                    if command == 'set_checkpoint':
                        identity = request['checkpoint_sha256']
                        if not isinstance(identity, str) or re.fullmatch('[0-9a-f]{64}', identity) is None:
                            raise ValueError('Invalid bridge checkpoint digest')
                        cfg['checkpoint_sha256'] = identity
                        atomic_json(checkpoint_state, dict(checkpoint_sha256=identity))
                    elif command == 'sleep' and not sleeping:
                        llm.sleep(level=1)
                        sleeping = True
                    elif command == 'wake' and sleeping:
                        llm.wake_up()
                        sleeping = False
                    else:
                        raise ValueError('Invalid worker lifecycle transition')
                    atomic_json(destination, dict(complete=True, command=command, sleeping=sleeping,
                        checkpoint_sha256=cfg['checkpoint_sha256'], worker_pid=os.getpid(),
                        elapsed_seconds=time.monotonic()-started))
                    continue
                if sleeping:
                    raise RuntimeError('Cannot generate while receiver is asleep')
                batch = json.loads(Path(request['request']).read_text())
                rows, cap = batch['rows'], batch['cap']
                if persistent and any(row['transfer'].get('checkpoint_sha256') != cfg['checkpoint_sha256']
                                      for row in rows):
                    raise ValueError('Rollout uses a stale bridge checkpoint')
                if any(len(row['prompt_token_ids']) + cap > cfg['max_model_len']
                       for row in rows):
                    raise ValueError('Qwen3.5 prompt and response exceed context budget')
                sampling = dict(batch['sampling'])
                if batch.get('logprobs'):
                    sampling['logprobs'] = 1
                parameters = [SamplingParams(max_tokens=cap - (method in ('bridge', 'random')),
                    seed=row['sampling_seed'], **sampling,
                    **(dict(extra_args={'kv_transfer_params': row['transfer'],
                                        'handoff_response_token': row['first_token']})
                       if method in ('bridge', 'random') else {}))
                    for row in rows if not (row.get('handoff_is_eos') or row.get('handoff_only'))]
                live = [row for row in rows if not (row.get('handoff_is_eos') or row.get('handoff_only'))]
                outputs = llm.generate([dict(prompt_token_ids=row['prompt_token_ids'] +
                    ([row['first_token']] if method in ('bridge', 'random') else []))
                    for row in live], parameters, use_tqdm=False,
                    lora_request=lora_request) if live else []
                if len(outputs) != len(live):
                    raise RuntimeError('Qwen3.5 vLLM response coverage differs')
                generated = iter(outputs)
                tokenizer = llm.get_tokenizer()
                results = []
                for row in rows:
                    output = None if row.get('handoff_is_eos') or row.get('handoff_only') else next(generated)
                    results.append(response_row(row, output, method, cap, tokenizer, batch.get('logprobs', False)))
                for result in results:
                    result['checkpoint_sha256'] = cfg['checkpoint_sha256']
                atomic_json(destination, {'complete': True, 'rows': results})
            except BaseException as error:
                atomic_json(destination, {'complete': False, 'error': repr(error)})
                traceback.print_exc()
                raise
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
