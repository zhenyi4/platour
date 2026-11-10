"""Native Qwen3 ARC-Challenge, zero-shot non-thinking, boxed choice avg@4."""
import argparse
from collections import Counter
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import generate_qwen35_train_suite as suite

ROOT = suite.ROOT
SCRIPT = Path(__file__).absolute()
CPU = ROOT / '.runtime/readme-check/scoring/bin/python'
if not CPU.exists():
    CPU = Path(sys.executable)
INFER = Path(os.environ.get('PLATOUR_GENERATION_PYTHON',
    str(ROOT / '.runtime/readme-check/generation/bin/python')))
if not INFER.exists():
    INFER = Path(sys.executable)
REVISION = '210d026faf9955653af8916fad021475a3f00453'
ALLOCATIONS = {'Qwen3-0.6B': [4], 'Qwen3-4B': [5], 'Qwen3-8B': [6, 7]}
SAMPLING = dict(temperature=.7, top_p=.8, top_k=20, min_p=0.,
                presence_penalty=0., frequency_penalty=0., repetition_penalty=1.)
INSTRUCTION = ('Please reason step by step, and put only the letter of your final answer '
               'within \\boxed{}, for example \\boxed{A}.')
ANSWER_ONLY_INSTRUCTION = ('Put only the letter of the correct answer within \\boxed{}, '
                           'for example \\boxed{A}. Do not include an explanation.')
ANSWER_SENTENCE_INSTRUCTION = ('Reply with exactly "The answer is \\boxed{X}", replacing X with '
                               'the letter of the correct option. Do not include any explanation '
                               'or any other text.')


def extract_boxed_choice(text, valid_labels):
    """Use only the last boxed expression; no prose or earlier-answer fallback."""
    matches = list(re.finditer(r'\\boxed\s*\{', text))
    if not matches:
        return None, 'missing_boxed'
    start = matches[-1].end()
    depth = 1
    for end in range(start, len(text)):
        depth += (text[end] == '{') - (text[end] == '}')
        if depth == 0:
            value = text[start:end].strip()
            while True:
                wrapper = re.fullmatch(r'\\(?:text|mathrm|mathbf|textbf)\s*\{\s*(.*?)\s*\}', value,
                                       flags=re.DOTALL)
                if wrapper is None:
                    break
                value = wrapper.group(1).strip()
            value = value.upper()
            return (value, 'valid') if value in valid_labels else (None, 'invalid_boxed')
    return None, 'invalid_boxed'


def score(row):
    answer, status = extract_boxed_choice(row['response'], row['labels'])
    return row | dict(extracted_answer=answer, extraction_status=status,
                      acc=int(answer == row['gold']))


def parser_checks():
    cases = [(r'Answer: B', 'ABCD', None, 'missing_boxed'),
             (r'We find \\boxed{b}', 'ABCD', 'B', 'valid'),
             (r'\\boxed{A} but finally \\boxed{C}', 'ABCD', 'C', 'valid'),
             (r'\\boxed{\\text{D}}', 'ABCD', 'D', 'valid'),
             (r'\\boxed{A or B}', 'ABCD', None, 'invalid_boxed'),
             (r'\\boxed{1}', 'ABCD', None, 'invalid_boxed'),
             (r'\\boxed{E}', 'ABCD', None, 'invalid_boxed'),
             (r'\\boxed{E}', 'ABCDE', 'E', 'valid'),
             (r'\\boxed{A} then \\boxed{B', 'ABCD', None, 'invalid_boxed')]
    # The raw test fixtures above have escaped slashes so they can be read literally.
    for text, labels, answer, status in cases:
        assert extract_boxed_choice(text.replace('\\\\', '\\'), labels) == (answer, status)


def prepare(root, samples, cap, models=None, gpus=None, greedy=False, sequential=False,
            answer_only=False, answer_sentence=False):
    import urllib.request
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    root.mkdir(parents=True, exist_ok=True)
    allocations = ALLOCATIONS
    if models is not None:
        gpus = [4, 5, 6, 7] if gpus is None else gpus
        assert models and len(set(models)) == len(models)
        assert len(set(gpus)) == len(gpus) and all(0 <= g < 8 for g in gpus)
        if sequential:
            assert len(gpus) == 1, 'Sequential mode uses one GPU for every model'
            allocations = {name: list(gpus) for name in models}
        else:
            assert len(models) <= len(gpus)
            allocations = {name: gpus[i * len(gpus) // len(models):(i + 1) * len(gpus) // len(models)]
                           for i, name in enumerate(models)}
    else:
        assert gpus is None and not sequential, '--gpus/--sequential require --models'
    assert not greedy or samples == 1, 'Greedy evaluation needs only one response per question'
    sampling = dict(SAMPLING)
    if greedy:
        sampling.update(temperature=0., top_p=1., top_k=0)
    assert not (answer_only and answer_sentence), 'Choose only one response instruction'
    instruction = (ANSWER_SENTENCE_INSTRUCTION if answer_sentence else
                   ANSWER_ONLY_INSTRUCTION if answer_only else INSTRUCTION)
    dataset_dir = ROOT / 'data/arc_challenge'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset = {}
    splits = {}
    for split, count in [('train', 1119), ('validation', 299), ('test', 1172)]:
        url = (f'https://huggingface.co/datasets/allenai/ai2_arc/resolve/{REVISION}/'
               f'ARC-Challenge/{split}-00000-of-00001.parquet')
        path = dataset_dir / f'{split}.parquet'
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        rows = pq.read_table(path).to_pylist()
        assert len(rows) == count and len({r['id'] for r in rows}) == count
        assert all(r['answerKey'] in r['choices']['label'] for r in rows)
        assert all(len(r['choices']['label']) == len(r['choices']['text']) for r in rows)
        splits[split] = rows
        dataset[split] = dict(count=count, path=str(path), url=url, sha256=suite.sha(path),
            choice_counts={str(k): v for k, v in Counter(len(r['choices']['label']) for r in rows).items()},
            numeric_label_questions=sum(r['choices']['label'][0] == '1' for r in rows))
    suite.save(dataset_dir / 'source.json', dict(repo='allenai/ai2_arc', config='ARC-Challenge',
        revision=REVISION, splits=dataset))
    questions = []
    for index, row in enumerate(splits['test']):
        labels = list('ABCDE'[:len(row['choices']['label'])])
        assert 3 <= len(labels) <= 5
        gold = labels[row['choices']['label'].index(row['answerKey'])]
        options = '\n'.join(f'{label}. {text}' for label, text in zip(labels, row['choices']['text'], strict=True))
        text = row['question'] + '\n\n' + options + '\n\n' + instruction
        questions.append(dict(index=index, id=row['id'], question=row['question'],
            choices=row['choices'], original_answer_key=row['answerKey'], labels=labels,
            gold=gold, input_text=text))
    suite.immutable(root / 'questions.json', questions)
    for name, gpus in allocations.items():
        model = ROOT / 'data/models' / name
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        prompts = []
        for row in questions:
            ids = tokenizer.apply_chat_template([dict(role='user', content=row['input_text'])],
                tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
            if hasattr(ids, 'keys'):
                ids = ids['input_ids']
            rendered = tokenizer.decode(ids)
            assert rendered.endswith('<think>\n\n</think>\n\n')
            assert rendered.count('<|im_start|>user\n') == 1
            prompts.append(dict(**row, prompt_token_ids=ids))
        maximum = max(len(r['prompt_token_ids']) for r in prompts)
        model_len = ((maximum + cap + 127) // 128) * 128
        config = json.loads((model / 'config.json').read_text())
        assert model_len <= config['max_position_embeddings']
        index_path = model / 'model.safetensors.index.json'
        shards = (sorted(set(json.loads(index_path.read_text())['weight_map'].values()))
                  if index_path.exists() else ['model.safetensors'])
        assert all((model / p).is_file() and (model / p).stat().st_size > 0 for p in shards)
        directory = root / name
        suite.immutable(directory / 'prompts.json', prompts)
        suite.immutable(directory / 'protocol.json', dict(model='Qwen/' + name, model_path=str(model),
            task='ARC-Challenge', split='test', dataset=dataset, revision=REVISION,
            questions=len(prompts), samples_per_question=samples, responses=len(prompts) * samples,
            max_new_tokens=cap, max_model_len=model_len, maximum_prompt_tokens=maximum,
            enable_thinking=False, native_chat_template=True, zero_shot=True,
            prompt_instruction=instruction, sampling=sampling,
            seed=f'1234 + {samples} * index + sample_index', request_n=1,
            world_size=len(gpus), gpu_ids=gpus, tensor_parallel_size=1, dtype='bfloat16',
            attention_backend='FLASH_ATTN', flash_attn_version=2, generation_config='vllm',
            quantization=None, max_num_seqs=64, gpu_memory_utilization=.40,
            scorer='Last balanced boxed expression; one valid choice letter; exact match',
            label_normalization='Map original labels in original order to A/B/C/D/E',
            filtering='None; retain all test questions and all responses, including clipped',
            prompts_sha256=suite.sha(directory / 'prompts.json'),
            model_config_sha256=suite.sha(model / 'config.json'),
            tokenizer_config_sha256=suite.sha(model / 'tokenizer_config.json'),
            tokenizer_sha256=suite.sha(model / 'tokenizer.json'),
            weights_index_sha256=suite.sha(index_path) if index_path.exists() else None,
            shard_sizes={p: (model / p).stat().st_size for p in shards}))
    suite.immutable(root / 'plan.json', dict(models=allocations, questions=1172, samples=samples,
        max_new_tokens=cap, enable_thinking=False, sampling=sampling, sequential=sequential,
        total_responses=1172 * samples * len(allocations)))
    paths = [SCRIPT, Path(suite.__file__).absolute()]
    suite.immutable(root / 'implementation.json', {str(p.relative_to(ROOT)): suite.sha(p) for p in paths})
    for path in paths:
        target = root / 'source_snapshot' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)


def check_sources(root):
    assert all(suite.sha(ROOT / p) == h for p, h in json.loads((root / 'implementation.json').read_text()).items())


def requests(prompts, samples):
    return [dict(**r, sample_index=s, sampling_seed=1234 + samples * r['index'] + s)
            for r in prompts for s in range(samples)]


def worker(root, name, rank):
    import torch
    from vllm import LLM, SamplingParams
    torch.set_num_threads(4)
    parser_checks()
    check_sources(root)
    directory = root / name
    p = json.loads((directory / 'protocol.json').read_text())
    assert suite.sha(directory / 'prompts.json') == p['prompts_sha256']
    prompts = json.loads((directory / 'prompts.json').read_text())
    rows = requests(prompts[rank::p['world_size']], p['samples_per_question'])
    path = directory / 'generation' / f'rank-{rank:02d}.jsonl'
    saved = suite.read(path)
    assert len(saved) <= len(rows)
    for actual, expected in zip(saved, rows):
        assert all(actual[k] == v for k, v in expected.items())
    started = time.monotonic()
    llm = LLM(model=p['model_path'], dtype='bfloat16', tensor_parallel_size=1,
        max_model_len=p['max_model_len'], max_num_seqs=64, gpu_memory_utilization=.40,
        enforce_eager=False, enable_prefix_caching=False, seed=1234, disable_log_stats=True,
        attention_config={'backend': 'FLASH_ATTN', 'flash_attn_version': 2}, generation_config='vllm')
    config = llm.llm_engine.vllm_config
    assert str(config.model_config.dtype) == 'torch.bfloat16' and config.model_config.quantization is None
    sampling = p['sampling']
    probe = SamplingParams(n=1, max_tokens=32, seed=1234, **sampling)
    assert all(getattr(probe, k) == v for k, v in sampling.items())
    warm = llm.generate([dict(prompt_token_ids=rows[0]['prompt_token_ids'])], probe, use_tqdm=False)[0]
    assert warm.prompt_token_ids == rows[0]['prompt_token_ids'] and warm.outputs[0].token_ids
    runtime = dict(vllm=importlib.metadata.version('vllm'), torch=torch.__version__,
        transformers=importlib.metadata.version('transformers'), sampling=sampling,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        generation_config='vllm', enforce_eager=config.model_config.enforce_eager,
        attention_config=str(config.attention_config),
        flash_attn_version=config.attention_config.flash_attn_version,
        cudagraph_mode=str(config.compilation_config.cudagraph_mode),
        load_and_warm_seconds=time.monotonic() - started, prompt_token_ids_verified=True)
    suite.save(path.with_suffix('.smoke.json'), dict(passed=True, runtime=runtime,
        prompt_tokens=len(rows[0]['prompt_token_ids']), response=warm.outputs[0].text,
        parser_cases_passed=9))
    tick = time.monotonic()
    for offset in range(len(saved), len(rows), 64):
        batch = rows[offset:offset + 64]
        params = [SamplingParams(n=1, max_tokens=p['max_new_tokens'], seed=r['sampling_seed'], **sampling)
                  for r in batch]
        outputs = llm.generate([dict(prompt_token_ids=r['prompt_token_ids']) for r in batch], params, use_tqdm=False)
        for row, output in zip(batch, outputs, strict=True):
            assert output.prompt_token_ids == row['prompt_token_ids'] and len(output.outputs) == 1
            response = output.outputs[0]
            tokens = list(response.token_ids)
            assert 0 < len(tokens) <= p['max_new_tokens']
            assert response.finish_reason != 'length' or len(tokens) == p['max_new_tokens']
            saved.append(score(dict(**row, response=response.text, response_token_ids=tokens,
                tokens=len(tokens), max_new_tokens=p['max_new_tokens'], finish_reason=response.finish_reason,
                stop_reason=response.stop_reason)))
        suite.jsonl(path, saved)
        suite.save(path.with_suffix('.live.json'), dict(saved=len(saved), total=len(rows),
            seconds=time.monotonic() - tick, correct=sum(r['acc'] for r in saved),
            clipped=sum(r['finish_reason'] == 'length' for r in saved),
            invalid_answers=sum(r['extraction_status'] != 'valid' for r in saved),
            generated_tokens=sum(r['tokens'] for r in saved)))
    suite.save(path.with_suffix('.complete.json'), dict(complete=True, responses=len(saved),
        output_sha256=suite.sha(path), protocol_sha256=suite.sha(directory / 'protocol.json'),
        runtime=runtime, seconds=time.monotonic() - tick))


def merge(root):
    results = {}
    allocations = json.loads((root / 'plan.json').read_text())['models']
    for name, gpus in allocations.items():
        directory = root / name
        p = json.loads((directory / 'protocol.json').read_text())
        prompts = json.loads((directory / 'prompts.json').read_text())
        rows = []
        runtimes = []
        for rank in range(len(gpus)):
            path = directory / 'generation' / f'rank-{rank:02d}.jsonl'
            done = json.loads(path.with_suffix('.complete.json').read_text())
            assert done['complete'] and done['output_sha256'] == suite.sha(path)
            assert done['protocol_sha256'] == suite.sha(directory / 'protocol.json')
            shard = suite.read(path)
            expected = requests(prompts[rank::len(gpus)], p['samples_per_question'])
            assert len(shard) == len(expected) == done['responses']
            for actual, row in zip(shard, expected, strict=True):
                assert all(actual[k] == v for k, v in row.items())
                assert score(actual) == actual
                assert actual['tokens'] == len(actual['response_token_ids']) <= p['max_new_tokens']
            rows.extend(shard)
            runtimes.append(done['runtime'])
        rows.sort(key=lambda r: (r['index'], r['sample_index']))
        samples = p['samples_per_question']
        assert [(r['index'], r['sample_index']) for r in rows] == [(i, s) for i in range(1172) for s in range(samples)]
        suite.jsonl(directory / 'scored.jsonl', rows)
        per_question = [dict(index=i, id=prompts[i]['id'], gold=prompts[i]['gold'],
                             correct=sum(r['acc'] for r in rows[i * samples:(i + 1) * samples]))
                        for i in range(1172)]
        suite.save(directory / 'per_question.json', per_question)
        summary = dict(complete=True, model=name, questions=1172, responses=len(rows),
            samples_per_question=samples, avg_at_k=sum(r['acc'] for r in rows) / len(rows),
            pass_at_k=sum(r['correct'] > 0 for r in per_question) / len(per_question),
            sample_accuracies=[sum(r['acc'] for r in rows if r['sample_index'] == s) / 1172 for s in range(samples)],
            clipping_rate=sum(r['finish_reason'] == 'length' for r in rows) / len(rows),
            clipped=sum(r['finish_reason'] == 'length' for r in rows),
            mean_tokens=sum(r['tokens'] for r in rows) / len(rows),
            extraction_status_counts=dict(Counter(r['extraction_status'] for r in rows)),
            invalid_answer_rate=sum(r['extraction_status'] != 'valid' for r in rows) / len(rows),
            protocol=p, runtimes=runtimes, scored_sha256=suite.sha(directory / 'scored.jsonl'))
        suite.save(directory / 'summary.json', summary)
        results[name] = summary
        suite.save(root / 'summary.json', results)
    return results


def queue(root):
    with (root / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_sources(root)
        plan = json.loads((root / 'plan.json').read_text())
        allocations = plan['models']
        jobs = []
        started = time.time()
        def stopped(sig, frame):
            raise SystemExit(128 + sig)
        signal.signal(signal.SIGTERM, stopped)
        signal.signal(signal.SIGINT, stopped)
        try:
            usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                            '--format=csv,noheader,nounits'], text=True)
            memory = {int(r.split(',')[0]): int(r.split(',')[1]) for r in usage.splitlines()}
            assert all(memory[g] < 512 for gs in allocations.values() for g in gs), 'Selected GPUs must be idle'
            suite.save(root / 'phase.json', dict(phase='generating', started=started))
            for name, gpus in allocations.items():
                for rank, gpu in enumerate(gpus):
                    path = root / 'logs' / f'{name}.rank-{rank:02d}.log'
                    path.parent.mkdir(exist_ok=True)
                    log = path.open('a')
                    proc = subprocess.Popen([str(INFER), '-u', str(SCRIPT), 'worker', '--root', str(root),
                        '--model', name, '--rank', str(rank)], cwd=ROOT, env=suite.environment([gpu]),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    jobs.append((proc, log))
                if plan.get('sequential', False):
                    suite.save(root / 'phase.json', dict(phase='generating', model=name, started=started))
                    suite.save(root / 'workers.json', [dict(pid=p.pid) for p, _ in jobs])
                    suite.wait_jobs(jobs)
                    jobs = []
            suite.save(root / 'workers.json', [dict(pid=p.pid) for p, _ in jobs])
            suite.wait_jobs(jobs)
            jobs = []
            results = merge(root)
            suite.save(root / 'complete.json', dict(complete=True, results=results, seconds=time.time() - started))
            suite.save(root / 'phase.json', dict(phase='complete', finished=time.time()))
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
    parser.add_argument('stage', choices=['prepare', 'queue', 'worker', 'check'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--model', choices=['Qwen3-1.7B', *ALLOCATIONS])
    parser.add_argument('--models', nargs='+', choices=['Qwen3-1.7B', *ALLOCATIONS])
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--greedy', action='store_true', help='Use temperature=0, top_p=1, top_k=0; requires --samples 1')
    parser.add_argument('--sequential', action='store_true', help='Run all --models in order on a single --gpus ID')
    prompt_mode = parser.add_mutually_exclusive_group()
    prompt_mode.add_argument('--answer-only', action='store_true', help='Request only a boxed choice letter without explanation')
    prompt_mode.add_argument('--answer-sentence', action='store_true', help='Request exactly: The answer is \\boxed{X}')
    args = parser.parse_args()
    parser_checks()
    if args.stage == 'check':
        print(json.dumps(dict(parser_checks_passed=9)))
    else:
        assert args.root is not None and args.samples > 0 and args.max_new_tokens > 0
        root = args.root.absolute()
        if args.stage == 'prepare':
            prepare(root, args.samples, args.max_new_tokens, models=args.models, gpus=args.gpus,
                    greedy=args.greedy, sequential=args.sequential, answer_only=args.answer_only,
                    answer_sentence=args.answer_sentence)
            print(json.dumps(dict(prepared=True, root=str(root))))
        elif args.stage == 'worker':
            worker(root, args.model, args.rank)
        elif args.detach:
            assert not (root / 'controller.json').exists(), 'Use a new root or resume in foreground'
            with (root / 'pipeline.log').open('a') as log:
                proc = subprocess.Popen([str(CPU), '-u', str(SCRIPT), 'queue', '--root', str(root)],
                    cwd=ROOT, env=suite.environment([]), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            suite.save(root / 'controller.json', dict(pid=proc.pid, time=time.time()))
            print(json.dumps(dict(pid=proc.pid, root=str(root))))
        else:
            queue(root)
