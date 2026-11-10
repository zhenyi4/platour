"""Persistent native vLLM replica; file RPC keeps the two HF runtimes isolated."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ['PYTHONPATH'] = str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get('PYTHONPATH','')
from qwen35_bridge_opd_core import ROOT, save


def decode_manifest(llm, tokenizer, manifest, target, sampling_params, decode_batch_size=0):
    """Queue all local requests by default; max_num_seqs limits active work.

    A positive decode_batch_size reproduces the old synchronized chunk path
    for matched benchmarks. Request order and per-request RNG seeds are kept.
    """
    rows, cap = manifest['rows'], manifest['tokens']
    batch_size = manifest.get('decode_batch_size', decode_batch_size)
    if batch_size < 0:
        raise ValueError('decode_batch_size must be nonnegative')
    batch_size = batch_size or max(1, len(rows))
    responses, batches = [], []
    for offset in range(0, len(rows), batch_size):
        current = rows[offset:offset+batch_size]
        live = [r for r in current if not r['handoff_is_eos']]
        started = time.monotonic()
        parameters = [sampling_params(max_tokens=cap-1, seed=r['sampling_seed'],
            logprobs=1 if manifest.get('return_logprobs',False) else None,
            **manifest['sampling'], extra_args={'kv_transfer_params':r['transfer'],
                'handoff_response_token':r['hf_first_token']}) for r in live]
        output = llm.generate([dict(prompt_token_ids=r['prompt_token_ids']+[r['hf_first_token']])
            for r in live], parameters, use_tqdm=False) if live else []
        if len(output) != len(live):
            raise RuntimeError('vLLM returned an unexpected response count')
        decoded = iter(output)
        for row in current:
            if row['handoff_is_eos']:
                ids, finish, stop = [row['hf_first_token']], 'stop', row['hf_first_token']
            else:
                answer = next(decoded).outputs[0]
                ids = [row['hf_first_token'], *answer.token_ids]
                finish, stop = answer.finish_reason, answer.stop_reason
            assert 0 < len(ids) <= cap
            if finish == 'length':
                assert len(ids) == cap
            extra = {}
            if manifest.get('return_logprobs',False):
                log_probs = [row['hf_first_logprob']]
                if not row['handoff_is_eos']:
                    assert len(answer.logprobs) == len(answer.token_ids)
                    log_probs.extend(float(value[token].logprob)
                        for token,value in zip(answer.token_ids,answer.logprobs,strict=True))
                assert len(log_probs) == len(ids)
                extra['rollout_log_probs'] = log_probs
            responses.append(dict(**{k:v for k,v in row.items() if k not in
                ['transfer','handoff_is_eos','hf_first_token','hf_first_logprob']}, **extra,
                response_token_ids=ids, response=tokenizer.decode(ids,skip_special_tokens=True),
                finish_reason=finish, stop_reason=stop, tokens=len(ids),
                bridge_state_sha256=manifest['bridge_state_sha256']))
        batches.append(dict(responses=len(current), seconds=time.monotonic()-started))
        save(target.with_name(target.stem+'.live.json'), dict(completed=len(responses),
            total=len(rows), batches=batches, queued_together=batch_size))
    result = dict(complete=True, rows=responses, batches=batches,
        seconds=sum(r['seconds'] for r in batches), queued_together=batch_size,
        bridge_state_sha256=manifest['bridge_state_sha256'])
    save(target, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--memory-fraction', type=float, default=.12)
    parser.add_argument('--max-model-len', type=int, default=6144)
    parser.add_argument('--max-num-seqs', type=int, default=32)
    parser.add_argument('--decode-batch-size', type=int, default=0,
        help='0 queues all requests; positive values retain synchronized batches for benchmarking')
    args = parser.parse_args()
    if args.max_num_seqs < 1 or args.decode_batch_size < 0:
        parser.error('Need positive max-num-seqs and nonnegative decode-batch-size')
    from vllm import LLM, SamplingParams
    import torch
    assert importlib.metadata.version('vllm') == '0.17.1'
    args.directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    llm = LLM(model=str(ROOT/'data/models/Qwen3.5-4B'), dtype='bfloat16',
        tensor_parallel_size=1, max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=4096, gpu_memory_utilization=args.memory_fraction,
        enable_prefix_caching=False, seed=42, disable_log_stats=True,
        language_model_only=True, limit_mm_per_prompt={'image':0,'video':0},
        attention_config={'backend':'FLASH_ATTN'}, mamba_cache_mode='none',
        mamba_ssm_cache_dtype='float32', disable_hybrid_kv_cache_manager=False,
        logits_processors=['qwen35_bridge_logits_processors:HandoffPresencePenalty'],
        kv_transfer_config=dict(kv_connector='Qwen35HybridMemoryConnector',
            kv_connector_module_path='qwen35_hybrid_memory_connector', kv_role='kv_consumer',
            kv_connector_extra_config=dict(storage_path=str((args.directory/'memory').resolve()),
                bridge_mode='full_hidden', allow_dynamic_checkpoint=True)))
    tokenizer = llm.get_tokenizer()
    config = llm.llm_engine.vllm_config
    save(args.directory/'ready.json', dict(pid=os.getpid(), vllm=importlib.metadata.version('vllm'),
        torch=torch.__version__, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        tensor_parallel_size=1, cuda_graphs=str(config.compilation_config.cudagraph_mode),
        checkpoint_provenance='Per-request SHA256 of actual FP32 projection tensors',
        load_seconds=time.monotonic()-start, gpu_memory_fraction=args.memory_fraction,
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        decode_batch_size=args.decode_batch_size))
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('command') == 'stop':
            break
        target = Path(request['result'])
        try:
            manifest = json.loads(Path(request['manifest']).read_text())
            rows, cap = manifest['rows'], manifest['tokens']
            assert all(len(r['prompt_token_ids'])+cap<=args.max_model_len for r in rows)
            decode_manifest(llm,tokenizer,manifest,target,SamplingParams,args.decode_batch_size)
        except BaseException as error:
            traceback.print_exc()
            save(target, dict(complete=False, error=str(error), traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
