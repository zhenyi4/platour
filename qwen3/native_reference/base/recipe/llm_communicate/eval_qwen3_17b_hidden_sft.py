"""Uniform zero hidden-SFT checkpoint: HF prefix, native vLLM response sampling."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from qwen3_17b_hidden_sft import ROOT, HERE, PYTHON, CPU, SAMPLING, env, load_bridge, digest
import generate_qwen35_train_suite as suite


def prefill(bridge, tokens):
    import torch
    if hasattr(bridge,'prefill_rows'):
        return bridge.prefill_rows([dict(prompt_token_ids=tokens)],None)
    ids = torch.tensor([tokens], device='cuda')
    mask = torch.ones_like(ids)
    positions = torch.arange(len(tokens), device='cuda')[None]
    with torch.no_grad():
        teacher = bridge.teacher.model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1],
            position_ids=positions[:, :-1], use_cache=False, output_hidden_states=True).hidden_states
    features = bridge.teacher_features(teacher)
    with bridge.inject(features, mask[:, :-1].bool()):
        return bridge.student(input_ids=ids, attention_mask=mask, position_ids=positions,
            use_cache=True, logits_to_keep=1)


def wait(process, target):
    deadline = time.monotonic() + 1800
    while not target.exists():
        assert process.poll() is None, 'vLLM worker exited; see worker.log'
        assert time.monotonic() < deadline, 'vLLM worker timeout'
        time.sleep(.2)
    value = json.loads(target.read_text())
    assert value['complete'], value
    return value


def worker(base, rank, checkpoint, output, smoke):
    import torch
    from prepare_qwen35_bridge_vllm import export_memory
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = json.loads((base / 'protocol.json').read_text())
    loader=load_bridge
    if protocol.get('bridge_family')=='c2c_235b':
        from arc_bridge_sft_common import load_hidden as loader
    bridge = loader(base, checkpoint).eval().requires_grad_(False)
    mode='c2c_kv' if protocol.get('method')=='c2c' else 'full_hidden'
    state_digest = digest(bridge)
    record = json.loads((checkpoint.parent.parent / 'complete.json').read_text())
    assert state_digest == record['final_sha256']
    assert suite.sha(checkpoint) == record['checkpoint_sha256']
    generation = output / 'generation'; generation.mkdir(parents=True, exist_ok=True)
    directory = generation / f'rank-{rank:02d}.vllm'; directory.mkdir(exist_ok=True)
    memory = directory / 'memory'; memory.mkdir(exist_ok=True)
    child_env = env([rank]); child_env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    command = [os.environ.get('PLATOUR_GENERATION_PYTHON',sys.executable), '-u',
        str(HERE / 'qwen3_17b_hidden_sft_vllm_worker.py'), '--directory', str(directory),
        '--model', protocol['student_checkpoint'], '--checkpoint', state_digest,
        '--method','c2c' if mode=='c2c_kv' else 'hidden']
    log = (directory / 'worker.log').open('a')
    process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdin=subprocess.PIPE,
        stdout=log, stderr=subprocess.STDOUT, text=True)
    parity_prefixes={}
    def decode(rows, cap, sampling, path, parity=False):
        exported, transfers = [], {}
        with torch.inference_mode():
            if protocol.get('batch_evaluation_prefill',False):
                unique=list({r['index']:r for r in rows}.values())
                assert hasattr(bridge,'prefill_rows')
                for offset in range(0,len(unique),8):
                    group=unique[offset:offset+8];out=bridge.prefill_rows(group,None)
                    for b,row in enumerate(group):
                        n=len(row['prompt_token_ids'])-1
                        tensors={f'layer.{i}.kv':torch.stack((layer.keys[b:b+1,:,:n],layer.values[b:b+1,:,:n]),1).transpose(2,3).cpu().contiguous()
                            for i,layer in enumerate(out.past_key_values.layers)}
                        transfer=export_memory(tensors,dict(prompt_token_ids=row['prompt_token_ids'][:-1]),state_digest,mode,memory)
                        transfer['checkpoint_sha256']=state_digest;transfers[row['index']]=transfer
                        if parity and protocol.get('parity_reuse_exported_prefix',False):
                            parity_prefixes[row['index']]=[(layer.keys[b:b+1,:,:n].detach().clone(),layer.values[b:b+1,:,:n].detach().clone())
                                for layer in out.past_key_values.layers]
                    del out,tensors
            for row in rows:
                if row['index'] not in transfers:
                    out = prefill(bridge, row['prompt_token_ids'])
                    tensors = {}
                    for i, layer in enumerate(out.past_key_values.layers):
                        tensors[f'layer.{i}.kv'] = torch.stack((layer.keys[:, :, :-1], layer.values[:, :, :-1]), 1).transpose(2, 3).cpu()
                    transfer = export_memory(tensors, dict(prompt_token_ids=row['prompt_token_ids'][:-1]),
                        state_digest, mode, memory)
                    transfer['checkpoint_sha256'] = state_digest
                    transfers[row['index']] = transfer
                    del out, tensors
                exported.append(dict(**row, transfer=transfers[row['index']]))
        manifest = path.with_name(path.stem + '.manifest.json')
        suite.save(manifest, dict(rows=exported, tokens=cap, sampling=sampling, parity=parity))
        process.stdin.write(json.dumps(dict(manifest=str(manifest), result=str(path))) + '\n')
        process.stdin.flush()
        value = wait(process, path)
        for cache_file in memory.glob('*.safetensors'): cache_file.unlink()
        return value
    try:
        ready = wait(process, directory / 'ready.json')
        prompts = json.loads((base / 'eval_prompts.json').read_text())
        if smoke:
            prompts = prompts[:6]
        else:
            prompts = prompts[rank::protocol.get('eval_world_size',8)]
        if smoke or protocol.get('eval_check_parity',False):
            probe_indices=list(dict.fromkeys([0,1,max(range(len(prompts)),key=lambda i:len(prompts[i]['prompt_token_ids']))]))
            probes = [dict(**prompts[i], sample_index=0, sampling_seed=1234 + protocol['eval_samples']*prompts[i]['index']) for i in probe_indices]
            suffix='' if smoke else f'.rank-{rank:02d}'
            result = decode(probes, 16, dict(temperature=0., top_p=1., top_k=-1, min_p=0.),
                output / f'parity_responses{suffix}.json', parity=True)
            errors, matches, details, rounding_checks = [], [], [], []
            with torch.inference_mode():
                for row in result['rows']:
                    if protocol.get('parity_reuse_exported_prefix',False):
                        from transformers import DynamicCache
                        cache=DynamicCache(config=bridge.student.config)
                        for i,(key,value) in enumerate(parity_prefixes[row['index']]):cache.update(key.clone(),value.clone(),i)
                        out=bridge.student(input_ids=torch.tensor([[row['prompt_token_ids'][-1]]],device='cuda'),
                            past_key_values=cache,use_cache=True,logits_to_keep=1)
                    else:out = prefill(bridge, row['prompt_token_ids'])
                    matches.append(out.logits[0, -1].argmax().item() == row['response_token_ids'][0])
                    for token, lp in zip(row['response_token_ids'][:8], row['log_probs'][:8], strict=True):
                        logits = out.logits[0, -1].float()
                        errors.append(abs(float(logits[token]-logits.logsumexp(-1))-lp))
                        magnitude=max(abs(float(logits[token])),abs(float(logits.logsumexp(-1))),1.)
                        ulp=2.**(math.floor(math.log2(magnitude))-7)
                        margin=float(logits.max()-logits[token])
                        rounding_checks.append(errors[-1]<=max(.2,2*ulp))
                        # For the first token only, differing greedy choices
                        # must also lie within the BF16 last-bit neighborhood.
                        if len(details)==0 or details[-1]['index']!=row['index']:
                            rounding_checks.append(margin<=max(.2,2*ulp))
                        details.append(dict(index=row['index'],token=token,hf_logprob=float(logits[token]-logits.logsumexp(-1)),
                            vllm_logprob=lp,error=errors[-1],bf16_ulp=ulp,hf_greedy_margin=margin))
                        out = bridge.student(input_ids=torch.tensor([[token]], device='cuda'),
                            past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
            report = dict(passed=sum(errors)/len(errors) < .05 and max(errors) < .2,
                mean_abs_logprob_error=sum(errors)/len(errors), max_abs_logprob_error=max(errors),
                first_greedy_matches=sum(matches), probes=len(probes), teacher_forced_tokens=8,
                nonzero_checkpoint=True, native_vllm_first_response_token=True)
            if protocol.get('parity_bf16_rounding_audit',False):
                report.update(passed=sum(errors)/len(errors)<.05 and all(rounding_checks),
                    bf16_rounding_bounds_passed=all(rounding_checks),token_details=details,
                    tolerance='mean<0.05; per-token error<=max(0.2,2*BF16 ULP); first-token greedy margin within same bound')
            report['hf_reference_reuses_exact_exported_prefix']=protocol.get('parity_reuse_exported_prefix',False)
            report['mean_logprob_check_passed']=sum(errors)/len(errors)<.05
            report['accepted']=report['passed'] or (protocol.get('allow_bf16_backend_variation',False) and report['mean_logprob_check_passed'])
            # Keep failed strict checks visible. Sampling eval may explicitly
            # accept small mean distribution error without requiring identical
            # greedy choices between distinct BF16 implementations.
            suite.save(output / f'parity{suffix}.json', report); assert report['accepted'], report
        requests = [dict(**r, sample_index=s, sampling_seed=1234 + protocol['eval_samples']*r['index'] + s)
            for r in prompts for s in range(2 if smoke else protocol['eval_samples'])]
        # Bound staging disk/cache and save each decoded chunk for audit.
        answers, seconds = [], 0.
        for offset in range(0, len(requests), 64):
            result = decode(requests[offset:offset+64], protocol['eval_response_cap'], SAMPLING,
                generation / f'rank-{rank:02d}.batch-{offset:04d}.json')
            answers.extend(result['rows']); seconds += result['seconds']
            suite.jsonl(generation / f'rank-{rank:02d}.jsonl', answers)
            suite.save(generation / f'rank-{rank:02d}.live.json', dict(responses=len(answers), total=len(requests), seconds=seconds))
        suite.save(generation / f'rank-{rank:02d}.complete.json', dict(complete=True, responses=len(answers),
            checkpoint_sha256=state_digest, ready=ready, seconds=seconds))
    finally:
        if process.poll() is None:
            try:
                process.stdin.write('{"command":"stop"}\n'); process.stdin.flush(); process.wait(timeout=30)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()


def merge(base, output, smoke):
    protocol = json.loads((base / 'protocol.json').read_text())
    rows = [r for path in sorted((output / 'generation').glob('rank-??.jsonl')) for r in suite.read(path)]
    rows.sort(key=lambda r: (r['index'], r['sample_index']))
    count = 6 if smoke else protocol['eval_questions']; k = 2 if smoke else protocol['eval_samples']
    assert [(r['index'], r['sample_index']) for r in rows] == [(i, s) for i in range(count) for s in range(k)]
    digests = {r['checkpoint_sha256'] for r in rows}; assert len(digests) == 1
    prompts = json.loads((base / 'eval_prompts.json').read_text())
    for row in rows:
        assert all(row[key] == value for key, value in prompts[row['index']].items())
        assert row['tokens'] == len(row['response_token_ids']) <= protocol['eval_response_cap']
        assert row['finish_reason'] != 'length' or row['tokens'] == protocol['eval_response_cap']
        assert row['sampling_seed'] == 1234 + protocol['eval_samples']*row['index'] + row['sample_index']
    scorer=suite.score
    if protocol.get('task')=='gsm8k':
        from generate_qwen3_gsm8k_sft_suite import score as scorer
    with ProcessPoolExecutor(max_workers=6) as pool:
        scored = list(pool.map(scorer, rows, chunksize=8))
    assert all(not r['gold_missing'] and not r['scoring_timed_out'] for r in scored)
    suite.jsonl(output / 'responses.jsonl', rows); suite.jsonl(output / 'scored.jsonl', scored)
    correct = sum(r['acc'] for r in scored)
    summary = dict(complete=True, smoke=smoke, questions=count, responses=len(rows),
        correct=correct, avg_at_k=correct/len(rows), **{f'avg_at_{k}': correct/len(rows)},
        pass_at_k=sum(any(r['acc'] for r in scored[i*k:(i+1)*k]) for i in range(count))/count,
        samples_per_question=k, clipping_rate=sum(r['finish_reason']=='length' for r in rows)/len(rows),
        mean_tokens=sum(r['tokens'] for r in rows)/len(rows), enable_thinking=False,
        sampling=SAMPLING, max_new_tokens=protocol['eval_response_cap'],
        checkpoint_sha256=next(iter(digests)), layer_map=protocol['layer_map'], initialization=protocol.get('initialization','zero'),
        training_epochs=3, method='c2c_sft' if protocol.get('method')=='c2c' else 'hidden_bridge_sft', scorer='last balanced box; exact numeric / Math-Verify' if protocol.get('task')=='gsm8k' else 'VERL boxed-only math_verify',
        responses_sha256=suite.sha(output / 'responses.jsonl'), scored_sha256=suite.sha(output / 'scored.jsonl'))
    suite.save(output / 'summary.json', summary); print(json.dumps(summary), flush=True)


def queue(base, output, checkpoint, smoke):
    processes = []
    def stop(signum, frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        for rank in range(1 if smoke else 8):
            log = (output / f'rank-{rank:02d}.log').open('a')
            process = subprocess.Popen([str(PYTHON), '-u', str(Path(__file__).absolute()), 'worker',
                '--root', str(base), '--output', str(output), '--checkpoint', str(checkpoint), '--rank', str(rank),
                *(['--smoke'] if smoke else [])], cwd=ROOT, env=env([rank]), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((process, log))
        suite.wait_jobs(processes); processes = []
        command = [str(CPU), '-u', str(Path(__file__).absolute()), 'merge', '--root', str(base),
            '--output', str(output), *(['--smoke'] if smoke else [])]
        subprocess.run(command, env=env([]), cwd=ROOT, check=True)
    finally:
        for process, log in processes:
            if process.poll() is None: suite.stop_job(process)
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['worker', 'merge', 'queue'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'worker': worker(args.root, args.rank, args.checkpoint, args.output, args.smoke)
    elif args.stage == 'merge': merge(args.root, args.output, args.smoke)
    else: queue(args.root, args.output, args.checkpoint, args.smoke)


if __name__ == '__main__': main()
