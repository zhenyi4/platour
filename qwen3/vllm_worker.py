"""Native vLLM 0.17.1 Qwen3 decode worker; imported prefix KV is mandatory."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

from .checkpoints import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    directory = parser.parse_args().directory
    config = json.loads((directory / 'config.json').read_text())
    persistent = config.get('persistent', False)
    if persistent and config.get('receiver_only'):
        raise ValueError('Persistent prefix decoding requires a frozen receiver')
    checkpoint_state = directory / 'active-checkpoint.json'
    if persistent:
        atomic_json(dict(checkpoint_sha256=config['checkpoint_sha256']), checkpoint_state)
    version = importlib.metadata.version('vllm')
    if version != '0.17.1':
        raise RuntimeError(f'Prefix connector requires vLLM 0.17.1; installed {version}')
    if config.get('receiver_only'):
        import hashlib
        checkpoint = Path(config['model']) / 'model.safetensors'
        actual = hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                actual.update(block)
        if actual.hexdigest() != config['checkpoint_sha256']:
            raise RuntimeError('Receiver-only weights changed before native vLLM load')
    from vllm import LLM, SamplingParams
    options = dict(model=config['model'], dtype='bfloat16', tensor_parallel_size=1,
                   max_model_len=config['max_model_len'],
                   gpu_memory_utilization=config['memory_fraction'],
                   max_num_seqs=config['max_num_seqs'],
                   max_num_batched_tokens=config['max_num_batched_tokens'],
                   enable_prefix_caching=False,
                   attention_config={'backend': 'FLASH_ATTN', 'flash_attn_version': 2},
                   generation_config='vllm', mamba_cache_mode='none', disable_log_stats=True)
    if persistent:
        options['enable_sleep_mode'] = True
    if not config.get('receiver_only'):
        options['kv_transfer_config'] = dict(kv_connector='PLatourPrefixConnector',
            kv_role='kv_consumer', kv_connector_module_path='qwen3.vllm_connector',
            kv_connector_extra_config=dict(storage_path=str(directory / 'memory'),
                                           checkpoint_sha256=config['checkpoint_sha256']))
        if persistent:
            options['kv_transfer_config']['kv_connector_extra_config']['checkpoint_state_path'] = str(checkpoint_state)
    llm = LLM(**options)
    sleeping = False
    atomic_json(dict(complete=True, backend='vllm', version=version,
                     prefix_caching=False, max_model_len=config['max_model_len'],
                     persistent=persistent, worker_pid=os.getpid()),
                directory / 'ready.json')
    try:
        for line in sys.stdin:
            message = json.loads(line)
            if message.get('command') == 'stop':
                break
            destination = Path(message['result'])
            try:
                command = message.get('command')
                if command is not None:
                    started = time.monotonic()
                    if not persistent or llm.llm_engine.has_unfinished_requests():
                        raise RuntimeError('Worker control requires an idle persistent engine')
                    if command == 'set_checkpoint':
                        identity = message['checkpoint_sha256']
                        if not isinstance(identity, str) or re.fullmatch('[0-9a-f]{64}', identity) is None:
                            raise ValueError('Invalid bridge checkpoint digest')
                        config['checkpoint_sha256'] = identity
                        atomic_json(dict(checkpoint_sha256=identity), checkpoint_state)
                    elif command == 'sleep' and not sleeping:
                        llm.sleep(level=1)
                        sleeping = True
                    elif command == 'wake' and sleeping:
                        llm.wake_up()
                        sleeping = False
                    else:
                        raise ValueError('Invalid worker lifecycle transition')
                    atomic_json(dict(complete=True, command=command, sleeping=sleeping,
                        checkpoint_sha256=config['checkpoint_sha256'], worker_pid=os.getpid(),
                        elapsed_seconds=time.monotonic()-started), destination)
                    continue
                if sleeping:
                    raise RuntimeError('Cannot generate while receiver is asleep')
                request = json.loads(Path(message['request']).read_text())
                rows, cap = request['rows'], request['cap']
                if persistent and any(row['transfer'].get('checkpoint_sha256') !=
                                      config['checkpoint_sha256'] for row in rows):
                    raise ValueError('Rollout uses a stale bridge checkpoint')
                if any(len(row['prompt_token_ids']) + cap > config['max_model_len']
                       for row in rows):
                    raise ValueError('Prompt and response exceed declared context budget')
                sampling = dict(request['sampling'])
                if sampling.get('top_k') == 0:
                    sampling['top_k'] = -1
                parameters = [SamplingParams(n=1, max_tokens=cap, seed=row['sampling_seed'],
                    logprobs=1 if request.get('parity') else None, **sampling,
                    **({} if config.get('receiver_only') else
                       {'extra_args': {'kv_transfer_params': row['transfer']}})) for row in rows]
                outputs = llm.generate([dict(prompt_token_ids=row['prompt_token_ids'])
                    for row in rows], parameters, use_tqdm=False)
                if len(outputs) != len(rows):
                    raise RuntimeError('Incomplete vLLM response count')
                results = []
                for row, output in zip(rows, outputs, strict=True):
                    if list(output.prompt_token_ids) != row['prompt_token_ids']:
                        raise ValueError('vLLM changed prompt token IDs')
                    answer = output.outputs[0]
                    ids = list(answer.token_ids)
                    if not ids or len(ids) > cap:
                        raise ValueError('Invalid generated token count')
                    extra = {}
                    if request.get('parity'):
                        if answer.logprobs is None or len(answer.logprobs) != len(ids):
                            raise ValueError('Missing parity token log probabilities')
                        extra['log_probs'] = [float(values[token].logprob)
                            for token, values in zip(ids, answer.logprobs, strict=True)]
                    results.append(dict(**extra, index=row['index'], sample_index=row['sample_index'],
                        sampling_seed=row['sampling_seed'], response=answer.text,
                        response_token_ids=ids, tokens=len(ids), finish_reason=answer.finish_reason,
                        stop_reason=answer.stop_reason,
                        checkpoint_sha256=config['checkpoint_sha256']))
                atomic_json(dict(complete=True, rows=results), destination)
            except BaseException as error:
                atomic_json(dict(complete=False, error=repr(error)), destination)
                traceback.print_exc()
                raise
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
