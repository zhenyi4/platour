"""Five native Qwen3.5 models: MATH-500 official non-thinking sample@4, <=6 GPUs."""
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
import sys
import time

import eval_qwen3_4b_aime135 as scorer_checks
import generate_qwen35_train_suite as suite

ROOT = suite.ROOT
SCRIPT = Path(__file__).resolve()
BASE = ROOT / 'outputs/llm_communicate/qwen35_math500_official_sample4_cap4096'
NAMES = ['Qwen3.5-4B', 'Qwen3.5-9B', 'Qwen3.5-27B', 'Qwen3.5-35B-A3B', 'Qwen3.5-122B-A10B']
TP = {name: 4 if '122B' in name else 1 for name in NAMES}
WORLD = {name: 1 if TP[name] == 4 else 6 for name in NAMES}
CAP = 4096
QUESTIONS = 500
SAMPLES = 4
SAMPLING = dict(temperature=1., top_p=1., top_k=40, min_p=0.,
                presence_penalty=2., repetition_penalty=1., frequency_penalty=0.)
ASSET = ROOT / 'outputs/llm_communicate/qwen3_large_encode_small_decode/evaluation/assets/math500.json'


def requests(prompts):
    return [dict(**row, sample_index=sample, sampling_seed=1234 + SAMPLES * row['index'] + sample)
            for row in prompts for sample in range(SAMPLES)]


def prepare(base):
    from transformers import AutoTokenizer
    base.mkdir(parents=True, exist_ok=True)
    original = json.loads(ASSET.read_text())
    rows = original['rows']
    assert original['repo'] == 'HuggingFaceH4/MATH-500'
    assert [r['index'] for r in rows] == list(range(QUESTIONS))
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'data/models/Qwen3-0.6B', local_files_only=True)
    head = '<|im_start|>user\n'
    tail = '<|im_end|>\n<|im_start|>assistant\n'
    questions = []
    for row in rows:
        rendered = tokenizer.decode(row['prompt_token_ids'], skip_special_tokens=False)
        assert rendered.count(head) == rendered.count(tail) == 1
        text = rendered.split(head, 1)[1].split(tail, 1)[0]
        assert text.endswith(suite.SUFFIX) and row['gold'].strip()
        check = tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        assert check == row['prompt_token_ids'], 'Original test questions must round-trip exactly'
        questions.append(dict(index=row['index'], gold=row['gold'],
                              question=text[:-len(suite.SUFFIX)], input_text=text))
    suite.immutable(base / 'questions.json', questions)
    downloads = json.loads((ROOT / 'outputs/model_downloads/qwen35_suite/manifest.json').read_text())
    for name in NAMES:
        spec = next(r for r in downloads['models'] if r['name'] == name)
        model = ROOT / 'data/models' / name
        assert all((model / p).is_file() and (model / p).stat().st_size == size
                   for p, size in spec['files'].items())
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        prompts = []
        for row in questions:
            ids = tokenizer.apply_chat_template([dict(role='user', content=row['input_text'])],
                tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
            assert len(ids) <= 2048 and tokenizer.decode(ids).endswith('<think>\n\n</think>\n\n')
            prompts.append(dict(**row, prompt_token_ids=ids))
        directory = base / name
        suite.immutable(directory / 'prompts.json', prompts)
        config = json.loads((model / 'config.json').read_text())
        assert CAP + 2048 <= config.get('text_config', config)['max_position_embeddings']
        protocol = dict(model=spec['repo'], revision=spec['revision'], model_path=str(model),
            dataset='HuggingFaceH4/MATH-500 test', dataset_revision=original['revision'],
            source_asset_sha256=suite.sha(ASSET), source_raw_content_sha256=original['raw_content_sha256'],
            canonical_questions_sha256=suite.sha(base / 'questions.json'),
            prompts_sha256=suite.sha(directory / 'prompts.json'),
            questions=QUESTIONS, responses=QUESTIONS * SAMPLES, samples_per_question=SAMPLES,
            enable_thinking=False, native_chat_template=True, prompt_instruction=suite.SUFFIX,
            maximum_prompt_tokens=max(len(r['prompt_token_ids']) for r in prompts),
            max_new_tokens=CAP, max_model_len=CAP + 2048, decoding='sampling', **SAMPLING,
            seed='1234 + 4 * index + sample_index', request_n=1,
            sampling_source=f'https://huggingface.co/Qwen/{name}#best-practices',
            sampling_source_section='Instruct (or non-thinking) mode for reasoning tasks',
            sampling_source_checked='official guidance', output_length_source='4096-token evaluation budget',
            world_size=WORLD[name], tensor_parallel_size=TP[name], gpu_limit=6,
            gpu_ids=[list(range(4))] if TP[name] == 4 else [[rank] for rank in range(6)],
            dtype='bfloat16', quantization=None, vllm='0.17.1', enforce_eager=False,
            attention_backend='FLASH_ATTN', language_model_only=True, max_num_seqs=32,
            request_batch_size=64, gpu_memory_utilization=.65, enable_prefix_caching=False,
            scorer='verl.utils.reward_score.math_verify.compute_score_dict', boxed_only=True,
            math_verify_version=importlib.metadata.version('math-verify'), score_timeout_seconds=10,
            metric='pass@4 = questions with >=1 correct / 500; avg@4 = correct responses / 2000',
            sample_filter='None; retain clipped, empty-text, and incorrect answers',
            checkpoint='Original downloaded BF16 model; no fine-tuning or quantization',
            model_config_sha256=suite.sha(model / 'config.json'),
            tokenizer_config_sha256=suite.sha(model / 'tokenizer_config.json'))
        suite.immutable(directory / 'protocol.json', protocol)
    suite.immutable(base / 'plan.json', dict(models=NAMES, questions_per_model=QUESTIONS,
        samples_per_question=SAMPLES, responses_per_model=QUESTIONS * SAMPLES,
        total_responses=len(NAMES) * QUESTIONS * SAMPLES, enable_thinking=False,
        decoding='sampling', sampling=SAMPLING, max_new_tokens=CAP, gpu_limit=6,
        order='Sequential; 4B first, 122B last', resume_paused_data_generation=False))
    files = [SCRIPT, Path(suite.__file__).resolve(), Path(scorer_checks.__file__).resolve(),
        ROOT / 'third_party/verl/verl/utils/reward_score/math_verify.py',
        ROOT / 'third_party/verl/verl/utils/reward_score/math_reward.py']
    suite.immutable(base / 'implementation.json', {str(p.relative_to(ROOT)): suite.sha(p) for p in files})
    for p in files:
        dest = base / 'source_snapshot' / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert suite.sha(dest) == suite.sha(p)
        else:
            shutil.copy2(p, dest)


def identity(directory, rank):
    protocol = json.loads((directory / 'protocol.json').read_text())
    return dict(protocol_sha256=suite.sha(directory / 'protocol.json'), rank=rank, world=protocol['world_size'])


def worker(base, name, rank):
    import torch
    from vllm import LLM, SamplingParams
    torch.set_num_threads(4)
    directory = base / name
    protocol = json.loads((directory / 'protocol.json').read_text())
    assert 0 <= rank < protocol['world_size']
    assert suite.sha(directory / 'prompts.json') == protocol['prompts_sha256']
    prompts = json.loads((directory / 'prompts.json').read_text())[rank::protocol['world_size']]
    rows = requests(prompts)
    gen = directory / 'generation'
    gen.mkdir(exist_ok=True)
    path = gen / f'rank-{rank:02d}.jsonl'
    marker = gen / f'rank-{rank:02d}.complete.json'
    if marker.exists():
        done = json.loads(marker.read_text())
        assert done['complete'] and done['identity'] == identity(directory, rank)
        assert done['responses'] == len(rows) and done['output_sha256'] == suite.sha(path)
        return
    saved = suite.read(path)
    assert len(saved) <= len(rows)
    for actual, expected in zip(saved, rows):
        assert all(actual[k] == v for k, v in expected.items())
        assert actual['max_new_tokens'] == CAP and actual['tokens'] == len(actual['response_token_ids'])
    started = time.monotonic()
    runtime_path = gen / f'rank-{rank:02d}.runtime.json'
    if len(saved) < len(rows):
        llm = LLM(model=protocol['model_path'], dtype='bfloat16', tensor_parallel_size=TP[name],
            max_model_len=protocol['max_model_len'], max_num_seqs=protocol['max_num_seqs'],
            gpu_memory_utilization=.65, enforce_eager=False, enable_prefix_caching=False,
            seed=1234, disable_log_stats=True, language_model_only=True,
            limit_mm_per_prompt={'image': 0, 'video': 0}, attention_config={'backend': 'FLASH_ATTN'})

        def parameters(seed, budget):
            params = SamplingParams(n=1, max_tokens=budget, seed=seed,
                                    **{key: protocol[key] for key in SAMPLING})
            assert all(getattr(params, key) == value for key, value in SAMPLING.items())
            return params

        llm.generate([dict(prompt_token_ids=r['prompt_token_ids']) for r in rows[len(saved):len(saved) + 4]],
            parameters(1234, 32), use_tqdm=False)
        config = llm.llm_engine.vllm_config
        runtime = dict(vllm=importlib.metadata.version('vllm'), torch=torch.__version__,
            transformers=importlib.metadata.version('transformers'), dtype=str(config.model_config.dtype),
            quantization=config.model_config.quantization, enforce_eager=config.model_config.enforce_eager,
            cudagraph_mode=str(config.compilation_config.cudagraph_mode), attention_config=str(config.attention_config),
            tensor_parallel_size=TP[name], max_model_len=config.model_config.max_model_len,
            sampling={key: getattr(parameters(1234, CAP), key) for key in SAMPLING},
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            load_and_warm_seconds=time.monotonic() - started)
        assert runtime['vllm'] == '0.17.1' and runtime['dtype'] == 'torch.bfloat16'
        assert runtime['quantization'] is None and runtime['enforce_eager'] is False
        assert 'NONE' not in runtime['cudagraph_mode']
        suite.save(runtime_path, runtime)
        opening = llm.get_tokenizer().convert_tokens_to_ids('<think>')
        for offset in range(len(saved), len(rows), 64):
            batch = rows[offset:offset + 64]
            batch_started = time.monotonic()
            outputs = llm.generate([dict(prompt_token_ids=r['prompt_token_ids']) for r in batch],
                [parameters(r['sampling_seed'], CAP) for r in batch], use_tqdm=False)
            seconds = time.monotonic() - batch_started
            for row, output in zip(batch, outputs, strict=True):
                assert output.prompt_token_ids == row['prompt_token_ids'] and len(output.outputs) == 1
                response = output.outputs[0]
                tokens = list(response.token_ids)
                assert 0 < len(tokens) <= CAP and opening not in tokens
                if response.finish_reason == 'length':
                    assert len(tokens) == CAP
                saved.append(dict(**row, response=response.text, response_token_ids=tokens,
                    tokens=len(tokens), max_new_tokens=CAP, finish_reason=response.finish_reason,
                    stop_reason=response.stop_reason))
            suite.jsonl(path, saved)
            live = dict(model=name, rank=rank, saved=len(saved), total=len(rows),
                seconds=time.monotonic() - started, generation_seconds=time.monotonic() - started - runtime['load_and_warm_seconds'],
                batch_seconds=seconds, batch_responses=len(batch),
                clipped=sum(r['finish_reason'] == 'length' for r in saved),
                generated_tokens=sum(r['tokens'] for r in saved))
            suite.save(gen / f'rank-{rank:02d}.live.json', live)
            print(json.dumps(live), flush=True)
    else:
        runtime = json.loads(runtime_path.read_text())
    assert len(saved) == len(rows)
    suite.save(marker, dict(complete=True, identity=identity(directory, rank), responses=len(rows),
        output_sha256=suite.sha(path), runtime=runtime, seconds=time.monotonic() - started))


def merge(base, name, started):
    directory = base / name
    protocol = json.loads((directory / 'protocol.json').read_text())
    assert suite.sha(directory / 'prompts.json') == protocol['prompts_sha256']
    prompts = json.loads((directory / 'prompts.json').read_text())
    rows, runtimes = [], []
    for rank in range(WORLD[name]):
        gen = directory / 'generation'
        path = gen / f'rank-{rank:02d}.jsonl'
        done = json.loads((gen / f'rank-{rank:02d}.complete.json').read_text())
        assert done['complete'] and done['identity'] == identity(directory, rank)
        assert done['output_sha256'] == suite.sha(path)
        shard = suite.read(path)
        expected = requests(prompts[rank::WORLD[name]])
        assert len(shard) == len(expected) == done['responses']
        for row, prompt in zip(shard, expected, strict=True):
            assert all(row[k] == v for k, v in prompt.items())
            assert row['tokens'] == len(row['response_token_ids']) and 0 < row['tokens'] <= CAP
        rows.extend(shard)
        runtimes.append(done['runtime'])
    rows.sort(key=lambda r: (r['index'], r['sample_index']))
    assert [(r['index'], r['sample_index']) for r in rows] == [
        (index, sample) for index in range(QUESTIONS) for sample in range(SAMPLES)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        scored = list(pool.map(suite.score, rows, chunksize=8))
    assert all(not r['gold_missing'] and not r['scoring_timed_out'] and r['acc'] in (0, 1) for r in scored)
    suite.jsonl(directory / 'responses.jsonl', rows)
    suite.jsonl(directory / 'scored.jsonl', scored)
    per_question = [dict(index=index, correct=sum(r['acc'] for r in scored[index * SAMPLES:(index + 1) * SAMPLES]))
                    for index in range(QUESTIONS)]
    suite.save(directory / 'per_question.json', per_question)
    passed = sum(r['correct'] > 0 for r in per_question)
    correct = sum(r['acc'] for r in scored)
    clipped = sum(r['finish_reason'] == 'length' for r in rows)
    summary = dict(model=name, questions=QUESTIONS, responses=len(rows), correct=correct,
        passed_questions=passed, pass_at_4=passed / QUESTIONS, avg_at_4=correct / len(rows),
        decoding='sampling', enable_thinking=False, max_new_tokens=CAP, sampling=SAMPLING,
        clipped=clipped, clipping_rate=clipped / len(rows),
        cap_hits=sum(r['tokens'] == CAP for r in rows),
        mean_tokens=sum(r['tokens'] for r in rows) / len(rows), maximum_tokens=max(r['tokens'] for r in rows),
        scoring_timeouts=0, gold_missing=0, wall_seconds=time.time() - started,
        protocol=protocol, runtimes=runtimes, responses_sha256=suite.sha(directory / 'responses.jsonl'),
        scored_sha256=suite.sha(directory / 'scored.jsonl'))
    suite.save(directory / 'summary.json', summary)
    suite.save(directory / 'complete.json', dict(complete=True, questions=QUESTIONS, responses=len(rows),
        pass_at_4=summary['pass_at_4'], avg_at_4=summary['avg_at_4'],
        scored_sha256=summary['scored_sha256'], finished=time.time()))
    return summary


def check_sources(base):
    for name, digest in json.loads((base / 'implementation.json').read_text()).items():
        assert suite.sha(ROOT / name) == digest, f'Evaluation implementation changed: {name}'


def queue(base):
    lock = (base / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_sources(base)
    started = time.time()
    jobs = []

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        results = {}
        for name in NAMES:
            check_sources(base)
            directory = base / name
            marker = directory / 'complete.json'
            if marker.exists():
                done = json.loads(marker.read_text())
                assert done['complete'] and done['scored_sha256'] == suite.sha(directory / 'scored.jsonl')
                results[name] = json.loads((directory / 'summary.json').read_text())
                continue
            for attempt in range(12):
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                                 '--format=csv,noheader,nounits'], text=True)
                used = {int(line.split(',')[0]): int(line.split(',')[1]) for line in output.splitlines()}
                if all(used[i] < 512 for i in range(6)):
                    break
                if attempt == 11:
                    raise RuntimeError('GPUs 0-5 occupied; MATH evaluation was not started')
                time.sleep(5)
            tick = time.time()
            suite.save(base / 'phase.json', dict(phase='generating', model=name, world=WORLD[name],
                tensor_parallel=TP[name], gpu_ids=list(range(4 if TP[name] == 4 else 6)), started=tick))
            logs = directory / 'logs'
            logs.mkdir(exist_ok=True)
            jobs = []
            for rank in range(WORLD[name]):
                gpus = list(range(4)) if TP[name] == 4 else [rank]
                log = (logs / f'rank-{rank:02d}.log').open('a')
                proc = subprocess.Popen([str(suite.INFER), '-u', str(SCRIPT), 'worker', '--root', str(base),
                    '--model', name, '--rank', str(rank)], cwd=ROOT, env=suite.environment(gpus),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                jobs.append((proc, log))
            suite.wait_jobs(jobs)
            jobs = []
            suite.save(base / 'phase.json', dict(phase='scoring', model=name, time=time.time()))
            results[name] = merge(base, name, tick)
            suite.save(base / 'summary.json', results)
            print(json.dumps({key: results[name][key] for key in
                              ['model', 'pass_at_4', 'avg_at_4', 'clipping_rate']}), flush=True)
        suite.save(base / 'summary.json', results)
        suite.save(base / 'complete.json', dict(complete=True, models=len(NAMES),
            responses=len(NAMES) * QUESTIONS * SAMPLES, results=results,
            wall_seconds=time.time() - started, finished=time.time()))
        suite.save(base / 'phase.json', dict(phase='complete', time=time.time()))
    except BaseException as error:
        suite.save(base / 'phase.json', dict(phase='interrupted' if isinstance(error, SystemExit) else 'failed',
                                            error=str(error), time=time.time()))
        raise
    finally:
        for proc, log in jobs:
            if proc.poll() is None:
                suite.stop_job(proc)
            if not log.closed:
                log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['queue', 'prepare', 'worker', 'status', 'self-test'])
    parser.add_argument('--root', type=Path, default=BASE)
    parser.add_argument('--model', choices=NAMES)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    base = args.root.resolve()
    sys.path.insert(0, str(ROOT / 'third_party/verl'))
    os.environ['MATH_VERIFY_BOXED_ONLY'] = '1'
    os.environ['MATH_VERIFY_TIMEOUT_S'] = '10'
    if args.stage == 'worker':
        assert args.model is not None
        worker(base, args.model, args.rank)
    elif args.stage == 'self-test':
        scorer_checks.self_test()
    elif args.stage == 'prepare':
        prepare(base)
        print('Prepared five native non-thinking MATH-500 evaluations, four official-parameter samples per question')
    elif args.stage == 'status':
        result = dict(phase=json.loads((base / 'phase.json').read_text()) if (base / 'phase.json').exists() else None,
                      models={})
        for name in NAMES:
            directory = base / name
            if (directory / 'summary.json').exists():
                summary = json.loads((directory / 'summary.json').read_text())
                fields = ['questions', 'responses', 'correct', 'passed_questions', 'pass_at_4', 'avg_at_4',
                          'clipped', 'clipping_rate', 'mean_tokens', 'wall_seconds', 'scoring_timeouts']
                result['models'][name] = dict(complete=True, **{key: summary[key] for key in fields})
            else:
                lives = [json.loads(p.read_text()) for p in (directory / 'generation').glob('*.live.json')]
                result['models'][name] = dict(saved=sum(r['saved'] for r in lives), total=QUESTIONS * SAMPLES,
                    clipped=sum(r['clipped'] for r in lives))
        print(json.dumps(result, indent=2))
    elif args.detach:
        base.mkdir(parents=True, exist_ok=True)
        controller = base / 'controller.json'
        if controller.exists():
            try:
                os.kill(json.loads(controller.read_text())['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('MATH-500 official sampling queue is already running')
        prepare(base)
        with (base / 'pipeline.log').open('a') as log:
            proc = subprocess.Popen([str(suite.CPU), '-u', str(SCRIPT), 'queue', '--root', str(base)],
                cwd=ROOT, env=suite.environment([]), stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        suite.save(controller, dict(pid=proc.pid, started=time.time()))
        print(json.dumps(dict(pid=proc.pid, root=str(base))), flush=True)
    else:
        queue(base)


if __name__ == '__main__':
    main()
