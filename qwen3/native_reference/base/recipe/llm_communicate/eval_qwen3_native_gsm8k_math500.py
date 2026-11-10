"""Qwen3 native non-thinking GSM8K avg@2 and MATH500 avg@4, eight GPUs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import generate_qwen35_train_suite as suite

ROOT = suite.ROOT
SCRIPT = Path(__file__).absolute()
CPU = ROOT / '.runtime/readme-check/scoring/bin/python'
INFER = ROOT / '.runtime/readme-check/generation/bin/python'
ALLOCATIONS = {'Qwen3-1.7B': [0, 1, 2], 'Qwen3-4B': [3, 4, 5], 'Qwen3-8B': [6, 7]}
TASKS = {'gsm8k': (1319, 2), 'math500': (500, 4)}
SOURCES = {'gsm8k': ROOT / 'outputs/llm_communicate/qwen35_gsm8k_greedy_cap2048',
           'math500': ROOT / 'outputs/llm_communicate/qwen35_math500_official_sample4_cap4096'}
SAMPLING = dict(temperature=.7, top_p=.8, top_k=20, min_p=0.,
                presence_penalty=0., frequency_penalty=0., repetition_penalty=1.)


def prepare(root, math_cap, models=None):
    from transformers import AutoTokenizer
    root.mkdir(parents=True, exist_ok=True)
    sources = {}
    for task, (count, _) in TASKS.items():
        origin = SOURCES[task]
        original = json.loads((origin / 'Qwen3.5-4B/protocol.json').read_text())
        questions = json.loads((origin / 'questions.json').read_text())
        assert suite.sha(origin / 'questions.json') == original['canonical_questions_sha256']
        assert [r['index'] for r in questions] == list(range(count))
        questions = [{k: r[k] for k in ('index', 'question', 'input_text', 'gold')} for r in questions]
        assert all(r['input_text'] == r['question'] + suite.SUFFIX and r['gold'].strip() for r in questions)
        suite.immutable(root / task / 'questions.json', questions)
        sources[task] = dict(dataset=original['dataset'], revision=original['dataset_revision'],
            canonical_source=str(origin / 'questions.json'), canonical_source_sha256=suite.sha(origin / 'questions.json'),
            original_raw_content_sha256=original['source_raw_content_sha256'])
    allocations = ALLOCATIONS
    if models is not None:
        assert models and len(set(models)) == len(models) and len(models) <= 8
        allocations = {name: list(range(i * 8 // len(models), (i + 1) * 8 // len(models)))
                       for i, name in enumerate(models)}
    for name, gpus in allocations.items():
        model = ROOT / 'data/models' / name
        config = json.loads((model / 'config.json').read_text())
        index = model / 'model.safetensors.index.json'
        shards = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
        assert all((model / p).is_file() and (model / p).stat().st_size > 0 for p in shards)
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        for task, (count, samples) in TASKS.items():
            rows = json.loads((root / task / 'questions.json').read_text())
            prompts = []
            for row in rows:
                ids = tokenizer.apply_chat_template([dict(role='user', content=row['input_text'])],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
                if hasattr(ids, 'keys'): ids = ids['input_ids']
                assert tokenizer.decode(ids).endswith('<think>\n\n</think>\n\n')
                assert len(ids) <= 2048
                prompts.append(dict(**row, prompt_token_ids=ids))
            directory = root / task / name
            cap = 2048 if task == 'gsm8k' else math_cap
            model_len = max(2048, math_cap) + 2048
            assert model_len <= config['max_position_embeddings']
            suite.immutable(directory / 'prompts.json', prompts)
            suite.immutable(directory / 'protocol.json', dict(model='Qwen/' + name, model_path=str(model),
                task=task, source=sources[task], questions=count, samples_per_question=samples,
                responses=count * samples, max_new_tokens=cap, max_model_len=model_len,
                maximum_prompt_tokens=max(len(r['prompt_token_ids']) for r in prompts),
                enable_thinking=False, prompt_instruction=suite.SUFFIX, sampling=SAMPLING,
                seed=f'1234 + {samples} * index + sample_index', request_n=1,
                world_size=len(gpus), gpu_ids=gpus, tensor_parallel_size=1, dtype='bfloat16',
                attention_backend='FLASH_ATTN', flash_attn_version=2,
                quantization=None, checkpoint='Native model; no SFT or bridge',
                scorer='verl.utils.reward_score.math_verify.compute_score_dict', boxed_only=True,
                prompts_sha256=suite.sha(directory / 'prompts.json'),
                model_config_sha256=suite.sha(model / 'config.json'),
                tokenizer_config_sha256=suite.sha(model / 'tokenizer_config.json'),
                weights_index_sha256=suite.sha(index) if index.exists() else None,
                shard_sizes={p: (model / p).stat().st_size for p in shards},
                sample_filter='None; retain incorrect and clipped responses'))
    suite.immutable(root / 'plan.json', dict(models=allocations, tasks=TASKS,
        math_cap=math_cap, gsm8k_cap=2048, sampling=SAMPLING, enable_thinking=False,
        total_responses=sum(n * k for n, k in TASKS.values()) * len(allocations)))
    paths = [SCRIPT, Path(suite.__file__).absolute(),
             ROOT / 'third_party/verl/verl/utils/reward_score/math_verify.py',
             ROOT / 'third_party/verl/verl/utils/reward_score/math_reward.py']
    suite.immutable(root / 'implementation.json', {str(p.relative_to(ROOT)): suite.sha(p) for p in paths})
    for path in paths:
        target = root / 'source_snapshot' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)


def check_sources(root):
    assert all(suite.sha(ROOT / p) == h for p, h in json.loads((root / 'implementation.json').read_text()).items())


def requests(prompts, samples):
    return [dict(**row, sample_index=s, sampling_seed=1234 + samples * row['index'] + s)
            for row in prompts for s in range(samples)]


def worker(root, name, rank, bootstrap_task='math500'):
    import torch
    from vllm import LLM, SamplingParams
    check_sources(root)
    torch.set_num_threads(4)
    p = json.loads((root / bootstrap_task / name / 'protocol.json').read_text())
    assert 0 <= rank < p['world_size']
    start = time.monotonic()
    llm = LLM(model=p['model_path'], dtype='bfloat16', tensor_parallel_size=1,
        max_model_len=p['max_model_len'], max_num_seqs=64, gpu_memory_utilization=.65,
        enforce_eager=False, enable_prefix_caching=False, seed=1234, disable_log_stats=True,
        attention_config={'backend': 'FLASH_ATTN', 'flash_attn_version': 2}, generation_config='vllm')
    config = llm.llm_engine.vllm_config
    assert str(config.model_config.dtype) == 'torch.bfloat16' and config.model_config.quantization is None
    probe = SamplingParams(n=1, max_tokens=32, seed=1234, **SAMPLING)
    assert all(getattr(probe, k) == v for k, v in SAMPLING.items())
    warm = json.loads((root / 'gsm8k' / name / 'prompts.json').read_text())[rank]
    llm.generate([dict(prompt_token_ids=warm['prompt_token_ids'])], probe, use_tqdm=False)
    runtime = dict(vllm=importlib.metadata.version('vllm'), torch=torch.__version__,
        transformers=importlib.metadata.version('transformers'), sampling=SAMPLING,
        generation_config='vllm; explicitly override all sampling fields',
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        enforce_eager=config.model_config.enforce_eager,
        attention_config=str(config.attention_config), flash_attn_version=config.attention_config.flash_attn_version,
        cudagraph_mode=str(config.compilation_config.cudagraph_mode),
        max_model_len=config.model_config.max_model_len, load_and_warm_seconds=time.monotonic() - start,
        one_model_load_for_both_tasks=True)
    for task, (_, samples) in TASKS.items():
        directory = root / task / name
        protocol = json.loads((directory / 'protocol.json').read_text())
        assert suite.sha(directory / 'prompts.json') == protocol['prompts_sha256']
        rows = requests(json.loads((directory / 'prompts.json').read_text())[rank::protocol['world_size']], samples)
        path = directory / 'generation' / f'rank-{rank:02d}.jsonl'
        saved = suite.read(path)
        assert len(saved) <= len(rows)
        for actual, expected in zip(saved, rows):
            assert all(actual[k] == v for k, v in expected.items())
        suite.save(path.with_suffix('.runtime.json'), runtime)
        tick = time.monotonic()
        for offset in range(len(saved), len(rows), 64):
            batch = rows[offset:offset + 64]
            params = [SamplingParams(n=1, max_tokens=protocol['max_new_tokens'], seed=r['sampling_seed'], **SAMPLING)
                      for r in batch]
            outputs = llm.generate([dict(prompt_token_ids=r['prompt_token_ids']) for r in batch], params, use_tqdm=False)
            for row, output in zip(batch, outputs, strict=True):
                assert output.prompt_token_ids == row['prompt_token_ids'] and len(output.outputs) == 1
                response = output.outputs[0]
                tokens = list(response.token_ids)
                assert 0 < len(tokens) <= protocol['max_new_tokens']
                assert response.finish_reason != 'length' or len(tokens) == protocol['max_new_tokens']
                saved.append(dict(**row, response=response.text, response_token_ids=tokens, tokens=len(tokens),
                    max_new_tokens=protocol['max_new_tokens'], finish_reason=response.finish_reason,
                    stop_reason=response.stop_reason))
            suite.jsonl(path, saved)
            suite.save(path.with_suffix('.live.json'), dict(saved=len(saved), total=len(rows),
                seconds=time.monotonic() - tick, clipped=sum(r['finish_reason'] == 'length' for r in saved),
                generated_tokens=sum(r['tokens'] for r in saved)))
        suite.save(path.with_suffix('.complete.json'), dict(complete=True, responses=len(saved),
            protocol_sha256=suite.sha(directory / 'protocol.json'), output_sha256=suite.sha(path), runtime=runtime,
            seconds=time.monotonic() - tick))


def merge(root, allow_missing_gold=False):
    results = {}
    allocations = json.loads((root / 'plan.json').read_text())['models']
    with ProcessPoolExecutor(max_workers=8) as pool:
        for task, (count, samples) in TASKS.items():
            results[task] = {}
            for name, gpus in allocations.items():
                directory = root / task / name
                protocol = json.loads((directory / 'protocol.json').read_text())
                prompts = json.loads((directory / 'prompts.json').read_text())
                rows, runtimes = [], []
                for rank in range(len(gpus)):
                    path = directory / 'generation' / f'rank-{rank:02d}.jsonl'
                    done = json.loads(path.with_suffix('.complete.json').read_text())
                    assert done['complete'] and done['output_sha256'] == suite.sha(path)
                    assert done['protocol_sha256'] == suite.sha(directory / 'protocol.json')
                    shard = suite.read(path)
                    expected = requests(prompts[rank::len(gpus)], samples)
                    assert len(shard) == len(expected) == done['responses']
                    for actual, row in zip(shard, expected, strict=True):
                        assert all(actual[k] == v for k, v in row.items())
                        assert actual['tokens'] == len(actual['response_token_ids']) <= protocol['max_new_tokens']
                    rows.extend(shard)
                    runtimes.append(done['runtime'])
                rows.sort(key=lambda r: (r['index'], r['sample_index']))
                assert [(r['index'], r['sample_index']) for r in rows] == [(i, s) for i in range(count) for s in range(samples)]
                scored = list(pool.map(suite.score, rows, chunksize=8))
                assert all(not r['scoring_timed_out'] and (r['gold_missing'] or r['acc'] in (0, 1)) for r in scored)
                if not allow_missing_gold:
                    assert all(not r['gold_missing'] for r in scored)
                judged = [r for r in scored if not r['gold_missing']]
                assert judged
                suite.jsonl(directory / 'responses.jsonl', rows)
                suite.jsonl(directory / 'scored.jsonl', scored)
                per_question = []
                for i in range(count):
                    responses = scored[i * samples:(i + 1) * samples]
                    valid = [r for r in responses if not r['gold_missing']]
                    per_question.append(dict(index=i, correct=sum(r['acc'] for r in valid) if valid else None,
                                             gold_missing=not valid))
                suite.save(directory / 'per_question.json', per_question)
                judged_questions = [r for r in per_question if not r['gold_missing']]
                avg = sum(r['acc'] for r in judged) / len(judged)
                passed = sum(r['correct'] > 0 for r in judged_questions) / len(judged_questions)
                summary = dict(complete=True, model=name, task=task, questions=count, responses=len(rows),
                    samples_per_question=samples, avg_at_k=avg, pass_at_k=passed,
                    **{f'avg_at_{samples}': avg, f'pass_at_{samples}': passed},
                    sample_accuracies=[sum(r['acc'] for r in judged if r['sample_index'] == s) /
                                      sum(r['sample_index'] == s for r in judged) for s in range(samples)],
                    clipped=sum(r['finish_reason'] == 'length' for r in rows),
                    clipping_rate=sum(r['finish_reason'] == 'length' for r in rows) / len(rows),
                    mean_tokens=sum(r['tokens'] for r in rows) / len(rows),
                    scoring_timeouts=0, gold_missing=len(scored) - len(judged),
                    judged_responses=len(judged), judged_questions=len(judged_questions),
                    protocol=protocol, runtimes=runtimes,
                    responses_sha256=suite.sha(directory / 'responses.jsonl'), scored_sha256=suite.sha(directory / 'scored.jsonl'))
                suite.save(directory / 'summary.json', summary)
                results[task][name] = summary
                suite.save(root / 'summary.json', results)
    return results


def queue(root, allow_missing_gold=False):
    with (root / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_sources(root)
        jobs = []
        started = time.time()
        def stopped(sig, frame):
            raise SystemExit(128 + sig)
        signal.signal(signal.SIGTERM, stopped)
        signal.signal(signal.SIGINT, stopped)
        try:
            usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
            assert all(int(r.split(',')[1]) < 512 for r in usage.splitlines()), 'Eight GPUs must be idle'
            suite.save(root / 'phase.json', dict(phase='generating', time=time.time()))
            for name, gpus in json.loads((root / 'plan.json').read_text())['models'].items():
                for rank, gpu in enumerate(gpus):
                    path = root / 'logs' / f'{name}.rank-{rank:02d}.log'
                    path.parent.mkdir(exist_ok=True)
                    log = path.open('a')
                    proc = subprocess.Popen([str(INFER), '-u', str(SCRIPT), 'worker', '--root', str(root),
                        '--model', name, '--rank', str(rank)], cwd=ROOT, env=suite.environment([gpu]),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    jobs.append((proc, log))
            suite.save(root / 'workers.json', [dict(pid=p.pid) for p, _ in jobs])
            suite.wait_jobs(jobs)
            jobs = []
            suite.save(root / 'phase.json', dict(phase='scoring', time=time.time()))
            results = merge(root, allow_missing_gold=allow_missing_gold)
            suite.save(root / 'complete.json', dict(complete=True, results=results, seconds=time.time() - started))
            suite.save(root / 'phase.json', dict(phase='complete', time=time.time()))
        except BaseException as error:
            suite.save(root / 'phase.json', dict(phase='interrupted' if isinstance(error, SystemExit) else 'failed',
                error=str(error), time=time.time()))
            raise
        finally:
            for proc, log in jobs:
                if proc.poll() is None:
                    suite.stop_job(proc)
                if not log.closed:
                    log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'queue', 'worker'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--math-cap', type=int, choices=[2048, 4096], default=4096)
    parser.add_argument('--model', choices=['Qwen3-0.6B', *ALLOCATIONS])
    parser.add_argument('--models', nargs='+', choices=['Qwen3-0.6B', *ALLOCATIONS])
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    root = args.root.absolute()
    if args.stage == 'prepare':
        prepare(root, args.math_cap, models=args.models)
        print(json.dumps(dict(prepared=True, root=str(root))))
    elif args.stage == 'worker':
        worker(root, args.model, args.rank)
    elif args.detach:
        assert not (root / 'controller.json').exists(), 'Controller already exists; use a fresh root'
        with (root / 'pipeline.log').open('a') as log:
            proc = subprocess.Popen([str(CPU), '-u', str(SCRIPT), 'queue', '--root', str(root)],
                cwd=ROOT, env=suite.environment([]), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        suite.save(root / 'controller.json', dict(pid=proc.pid, time=time.time()))
        print(json.dumps(dict(pid=proc.pid, root=str(root))))
    else:
        queue(root)
