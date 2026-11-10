"""After 35B-A3B->4B SFT, evaluate the final bridge on MATH-500 sample@4."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import generate_qwen35_train_suite as suite
from eval_qwen35_math500_official_suite import SAMPLING

ROOT = suite.ROOT
SCRIPT = Path(__file__).resolve()
TRAIN = ROOT / 'outputs/llm_communicate/qwen35_35b_to_4b_math_direct_zero_last32_linear_8gpu_fresh'
NATIVE = ROOT / 'outputs/llm_communicate/qwen35_math500_official_sample4_cap4096'
DECODER_BASELINE = NATIVE
BASE = TRAIN / 'eval_epoch3_official_sample4_cap4096'
PYTHON = ROOT / '.runtime/qwen35-bridge/bin/python'
GPUS = list(range(8))
WORLD, CAP, SAMPLES, BATCH = 8, 4096, 4, 16


def prepare(base):
    from transformers import AutoTokenizer
    base.mkdir(parents=True, exist_ok=True)
    native = json.loads((DECODER_BASELINE / 'Qwen3.5-4B/protocol.json').read_text())
    encoder = json.loads((NATIVE / 'Qwen3.5-35B-A3B/protocol.json').read_text())
    prompts = json.loads((DECODER_BASELINE / 'Qwen3.5-4B/prompts.json').read_text())
    assert suite.sha(DECODER_BASELINE / 'Qwen3.5-4B/prompts.json') == native['prompts_sha256']
    assert prompts == json.loads((NATIVE / 'Qwen3.5-35B-A3B/prompts.json').read_text())
    student = AutoTokenizer.from_pretrained(ROOT / 'data/models/Qwen3.5-4B', local_files_only=True)
    teacher = AutoTokenizer.from_pretrained(ROOT / 'data/models/Qwen3.5-35B-A3B', local_files_only=True)
    assert student.get_vocab() == teacher.get_vocab()
    assert [r['index'] for r in prompts] == list(range(500))
    for row in prompts:
        for tokenizer in (student, teacher):
            ids = tokenizer.apply_chat_template([dict(role='user', content=row['input_text'])],
                tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
            assert ids == row['prompt_token_ids'] and len(ids) + CAP <= 6144
    suite.immutable(base / 'prompts.json', prompts)
    protocol = dict(problem_encoder='Qwen/Qwen3.5-35B-A3B', answer_decoder='Qwen/Qwen3.5-4B',
        problem_encoder_revision=encoder['revision'], answer_decoder_revision=native['revision'],
        training=str(TRAIN), checkpoint=str(TRAIN / 'train/epoch-3/bridge.pt'),
        checkpoint_selection='Final epoch 3; no selection on the test set',
        training_implementation_sha256=suite.sha(TRAIN / 'implementation.json'),
        method='h_s[i] + P_i(h_t[i+8])', layer_mapping='last_aligned',
        teacher_layers=list(range(8, 40)), student_layers=list(range(32)),
        hidden_space='decoder block input before RMSNorm', prefix='Prompt except final prompt token',
        response_injection=False, frozen_language_models=True, dtype='bfloat16', quantization=None,
        dataset=native['dataset'], dataset_revision=native['dataset_revision'],
        source_asset_sha256=native['source_asset_sha256'],
        prompts_sha256=suite.sha(base / 'prompts.json'), questions=500, responses=2000,
        samples_per_question=SAMPLES, max_new_tokens=CAP, max_model_len=6144,
        enable_thinking=False, native_chat_template=True, prompt_instruction=suite.SUFFIX,
        decoding='sampling', **SAMPLING, seed='1234 + 4 * index + sample_index',
        presence_penalty_scope='Generated tokens only; prompt tokens are not penalized',
        sampling_method='Inverse CDF over top-k probabilities; independent per-response torch.Generator',
        backend='Transformers native hybrid cache + FLA + SDPA',
        comparison_note='Native baselines use vLLM. Teacher baseline, student baseline and bridge use their official top_k=40 non-thinking reasoning sampler. Prompts, parameters and seed formula match; RNG/backends differ.',
        sampler_source=native['sampling_source'], world_size=WORLD, gpu_ids=GPUS,
        batch_size=BATCH, combined_gpu_limit=8,
        scorer=native['scorer'], boxed_only=True, score_timeout_seconds=10,
        filter='None; include incorrect, clipped and empty-text responses',
        metric='pass@4 = questions with >=1 correct / 500; avg@4 = correct responses / 2000')
    suite.immutable(base / 'protocol.json', protocol)
    paths = [SCRIPT, ROOT / 'recipe/llm_communicate/qwen35_last_aligned_math_bridge.py',
        ROOT / 'recipe/llm_communicate/qwen35_35b_4b_math_bridge.py',
        ROOT / 'recipe/llm_communicate/eval_qwen35_math500_official_suite.py',
        ROOT / 'third_party/verl/verl/utils/reward_score/math_verify.py',
        ROOT / 'third_party/verl/verl/utils/reward_score/math_reward.py']
    paths += [ROOT / name for name in json.loads((TRAIN / 'implementation.json').read_text())]
    files = {str(p.relative_to(ROOT)): suite.sha(p) for p in paths}
    suite.immutable(base / 'implementation.json', files)
    for name, digest in files.items():
        source, dest = ROOT / name, base / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert suite.sha(dest) == digest
        else:
            shutil.copy2(source, dest)


def check_sources(base):
    for name, digest in json.loads((base / 'implementation.json').read_text()).items():
        assert suite.sha(ROOT / name) == digest, f'Evaluation implementation changed: {name}'
    protocol = json.loads((base / 'protocol.json').read_text())
    assert suite.sha(base / 'prompts.json') == protocol['prompts_sha256']
    assert suite.sha(TRAIN / 'implementation.json') == protocol['training_implementation_sha256']


def selected_checkpoint(base):
    protocol = json.loads((base / 'protocol.json').read_text())
    done = json.loads((TRAIN / 'train/complete.json').read_text())
    path = Path(protocol['checkpoint'])
    assert done['complete'] and not done['smoke'] and done['epochs'] == 3 and done['steps'] == 351
    assert done['selected_checkpoint'] == str(path) and done['checkpoint_sha256'] == suite.sha(path)
    return path, done['checkpoint_sha256']


def prompt_tensors(rows, pad, device):
    import torch
    width = max(len(r['prompt_token_ids']) for r in rows)
    ids, masks = [], []
    for row in rows:
        tokens = row['prompt_token_ids']
        left = width - len(tokens)
        ids.append([pad] * left + tokens)
        masks.append([0] * left + [1] * len(tokens))
    return torch.tensor(ids, device=device), torch.tensor(masks, device=device)


def prefill(bridge, ids, mask, extra=None):
    import torch
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    teacher = bridge.teacher.model(input_ids=ids, attention_mask=mask,
        position_ids=positions, use_cache=False, output_hidden_states=True).hidden_states
    features = bridge.teacher_features(teacher)
    inject = mask.bool().clone()
    inject[:, -1] = False
    if extra is not None:
        ids = torch.cat([ids, extra], 1)
        mask = torch.cat([mask, torch.ones_like(extra)], 1)
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
    with bridge.inject(features, inject):
        return bridge.student(input_ids=ids, attention_mask=mask, position_ids=positions,
            use_cache=True, logits_to_keep=1)


def decode_step(student, nxt, mask, cache):
    import torch
    positions = mask.sum(-1, keepdim=True)
    mask = torch.cat([mask, torch.ones_like(nxt)], 1)
    output = student(input_ids=nxt, attention_mask=mask, position_ids=positions,
        past_key_values=cache, use_cache=True, logits_to_keep=1)
    return output, mask


def sample_tokens(logits, seen, uniform):
    """The current official sampler: T=1, top-k=40, top-p=1, presence=2."""
    import torch
    assert SAMPLING == dict(temperature=1., top_p=1., top_k=40, min_p=0.,
                           presence_penalty=2., repetition_penalty=1., frequency_penalty=0.)
    values, indices = torch.topk(logits.float() - 2. * seen, min(40, logits.shape[-1]), dim=-1)
    probabilities = values.softmax(-1)
    selected = (probabilities.cumsum(-1) < uniform[:, None]).sum(-1, keepdim=True)
    return indices.gather(-1, selected.clamp_max(indices.shape[-1] - 1)).squeeze(-1)


def batch_cache_check(bridge, ids, mask):
    import torch
    first = prefill(bridge, ids, mask)
    nxt = first.logits[:, -1].argmax(-1, keepdim=True)
    cached, extended_mask = decode_step(bridge.student, nxt, mask, first.past_key_values)
    full = prefill(bridge, ids, mask, extra=nxt)
    error = (cached.logits.float() - full.logits.float()).abs()
    fresh = prefill(bridge, ids, mask)
    native = bridge.student.generate(input_ids=torch.cat([ids, nxt], 1), attention_mask=extended_mask,
        past_key_values=fresh.past_key_values, max_new_tokens=1, do_sample=False,
        return_dict_in_generate=True, output_logits=True)
    native_error = (native.logits[0].float() - cached.logits[:, -1].float()).abs()
    result = dict(batch_size=len(ids), left_padding=True, native_hybrid_cache=True,
        cache_type=type(cached.past_key_values).__name__, response_injection=False,
        mean_abs_logit_error=float(error.mean()), max_abs_logit_error=float(error.max()),
        mean_abs_logit_tolerance=.05, max_abs_logit_tolerance=.5,
        native_generate_max_abs_logit_error=float(native_error.max()),
        native_generate_greedy_match=bool(torch.equal(native.sequences[:, -1], cached.logits[:, -1].argmax(-1))))
    assert result['mean_abs_logit_error'] < .05 and result['max_abs_logit_error'] <= .5, result
    assert result['native_generate_max_abs_logit_error'] < 1e-5 and result['native_generate_greedy_match'], result
    return result


def worker(base, rank):
    import torch
    from transformers import AutoTokenizer
    from qwen35_35b_4b_math_bridge import load_bridge, cache_check
    assert 0 <= rank < WORLD
    check_sources(base)
    path, digest = selected_checkpoint(base)
    generation = base / 'generation'
    generation.mkdir(exist_ok=True)
    target = generation / f'rank-{rank:02d}.jsonl'
    marker = generation / f'rank-{rank:02d}.complete.json'
    identity = dict(protocol_sha256=suite.sha(base / 'protocol.json'), checkpoint_sha256=digest,
                    rank=rank, world=WORLD)
    prompts = json.loads((base / 'prompts.json').read_text())[rank::WORLD]
    requests = [dict(**row, sample_index=j, sampling_seed=1234 + 4 * row['index'] + j)
                for row in prompts for j in range(SAMPLES)]
    saved = suite.read(target)
    assert len(saved) <= len(requests)
    for actual, expected in zip(saved, requests):
        assert all(actual[k] == value for k, value in expected.items())
        assert actual['checkpoint_sha256'] == digest and 0 < len(actual['response_token_ids']) <= CAP
    if marker.exists():
        done = json.loads(marker.read_text())
        assert done['identity'] == identity and done['output_sha256'] == suite.sha(target)
        assert done['responses'] == len(requests) == len(saved)
        return
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bridge, runtime = load_bridge(ROOT)
    metadata = json.loads(path.with_name('subspace_config.json').read_text())
    assert metadata == bridge.metadata(), 'Evaluation bridge must match the trained layer mapping and method'
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert set(state) == {'P'}
    bridge.maps.load_state_dict(state['P'], strict=True)
    del state
    assert any(torch.count_nonzero(p) > 0 for p in bridge.bridge_parameters())
    bridge.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'data/models/Qwen3.5-4B', local_files_only=True)
    eos = bridge.student.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    eos_tensor = torch.tensor(eos, device='cuda')
    runtime.update(cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], checkpoint_sha256=digest,
        sampler=SAMPLING, native_hybrid_cache=True, response_injection=False, batch_size=BATCH)
    suite.save(generation / f'rank-{rank:02d}.runtime.json', runtime)
    started = time.monotonic()
    with torch.inference_mode():
        sanity_rows = [min(prompts, key=lambda r: len(r['prompt_token_ids'])),
                       max(prompts, key=lambda r: len(r['prompt_token_ids']))]
        ids, mask = prompt_tensors(sanity_rows, tokenizer.pad_token_id, 'cuda')
        sanity = dict(single=cache_check(bridge, sanity_rows[0]['prompt_token_ids']),
                      batch=batch_cache_check(bridge, ids, mask))
        suite.save(generation / f'rank-{rank:02d}.sanity.json', sanity)
        del ids, mask
        for offset in range(len(saved), len(requests), BATCH):
            batch = requests[offset:offset + BATCH]
            ids, mask = prompt_tensors(batch, tokenizer.pad_token_id, 'cuda')
            output = prefill(bridge, ids, mask)
            logits, cache = output.logits[:, -1], output.past_key_values
            del output
            uniforms = torch.stack([torch.rand(CAP, device='cuda', generator=torch.Generator(device='cuda')
                .manual_seed(r['sampling_seed'])) for r in batch])
            seen = torch.zeros_like(logits, dtype=torch.bool)
            done = torch.zeros(len(batch), device='cuda', dtype=torch.bool)
            tokens = []
            for step in range(CAP):
                nxt = sample_tokens(logits, seen, uniforms[:, step])
                nxt = torch.where(done, tokenizer.pad_token_id, nxt)
                tokens.append(nxt)
                seen.scatter_(1, nxt[:, None], True)
                done |= (nxt[:, None] == eos_tensor).any(-1)
                if (step + 1) % 256 == 0:
                    print(json.dumps(dict(rank=rank, offset=offset, step=step + 1,
                        done=int(done.sum()), batch=len(batch), seconds=time.monotonic() - started)), flush=True)
                if bool(done.all()) or step + 1 == CAP:
                    break
                output, mask = decode_step(bridge.student, nxt[:, None], mask, cache)
                logits, cache = output.logits[:, -1], output.past_key_values
                del output
            for row, response in zip(batch, torch.stack(tokens, 1).cpu().tolist(), strict=True):
                stop = next((i for i, token in enumerate(response) if token in eos), None)
                if stop is not None:
                    response = response[:stop + 1]
                saved.append(dict(**row, response=tokenizer.decode(response, skip_special_tokens=True),
                    response_token_ids=response, tokens=len(response), max_new_tokens=CAP,
                    finish_reason='stop' if stop is not None else 'length', checkpoint_sha256=digest))
            suite.jsonl(target, saved)
            live = dict(rank=rank, saved=len(saved), total=len(requests),
                seconds=time.monotonic() - started, clipped=sum(r['finish_reason'] == 'length' for r in saved),
                generated_tokens=sum(r['tokens'] for r in saved))
            suite.save(generation / f'rank-{rank:02d}.live.json', live)
            print(json.dumps(live), flush=True)
            del ids, mask, cache, logits, seen, tokens, uniforms, done
    assert len(saved) == len(requests)
    suite.save(marker, dict(complete=True, identity=identity, responses=len(saved), output_sha256=suite.sha(target)))


def merge(base, started):
    _, digest = selected_checkpoint(base)
    prompts = json.loads((base / 'prompts.json').read_text())
    rows = []
    for rank in range(WORLD):
        generation = base / 'generation'
        target = generation / f'rank-{rank:02d}.jsonl'
        done = json.loads((generation / f'rank-{rank:02d}.complete.json').read_text())
        assert done['complete'] and done['output_sha256'] == suite.sha(target)
        assert done['identity'] == dict(protocol_sha256=suite.sha(base / 'protocol.json'),
                                       checkpoint_sha256=digest, rank=rank, world=WORLD)
        sanity = json.loads((generation / f'rank-{rank:02d}.sanity.json').read_text())
        assert all(value['native_generate_greedy_match'] for value in sanity.values())
        shard = suite.read(target)
        assert len(shard) == done['responses'] == len(prompts[rank::WORLD]) * SAMPLES
        rows.extend(shard)
    rows.sort(key=lambda r: (r['index'], r['sample_index']))
    assert [(r['index'], r['sample_index']) for r in rows] == [(i, j) for i in range(500) for j in range(4)]
    for row in rows:
        assert all(row[key] == value for key, value in prompts[row['index']].items())
        assert row['checkpoint_sha256'] == digest and row['sampling_seed'] == 1234 + 4 * row['index'] + row['sample_index']
        assert row['tokens'] == len(row['response_token_ids']) and 0 < row['tokens'] <= CAP
        assert row['finish_reason'] in ('stop', 'length')
        if row['finish_reason'] == 'length':
            assert row['tokens'] == CAP
    suite.jsonl(base / 'responses.jsonl', rows)
    with ProcessPoolExecutor(max_workers=6) as pool:
        scored = list(pool.map(suite.score, rows, chunksize=8))
    assert all(not r['gold_missing'] and not r['scoring_timed_out'] and r['acc'] in (0, 1) for r in scored)
    suite.jsonl(base / 'scored.jsonl', scored)
    per_question = [dict(index=i, correct=sum(r['acc'] for r in scored[4 * i:4 * i + 4])) for i in range(500)]
    suite.save(base / 'per_question.json', per_question)
    correct = sum(r['acc'] for r in scored)
    passed = sum(r['correct'] > 0 for r in per_question)
    summary = dict(model='Qwen3.5-35B-A3B encode -> Qwen3.5-4B decode; zero-linear last-32 bridge, MATH SFT epoch 3',
        questions=500, responses=2000, correct=correct, passed_questions=passed,
        pass_at_4=passed / 500, avg_at_4=correct / 2000,
        clipped=sum(r['finish_reason'] == 'length' for r in rows),
        clipping_rate=sum(r['finish_reason'] == 'length' for r in rows) / 2000,
        mean_tokens=sum(r['tokens'] for r in rows) / 2000, maximum_tokens=max(r['tokens'] for r in rows),
        checkpoint_sha256=digest, scoring_timeouts=0, gold_missing=0,
        protocol=json.loads((base / 'protocol.json').read_text()), wall_seconds=time.time() - started,
        responses_sha256=suite.sha(base / 'responses.jsonl'), scored_sha256=suite.sha(base / 'scored.jsonl'))
    suite.save(base / 'summary.json', summary)
    comparison = {'Qwen3.5-4B': json.loads((DECODER_BASELINE / 'Qwen3.5-4B/summary.json').read_text()),
                  'Qwen3.5-35B-A3B': json.loads((NATIVE / 'Qwen3.5-35B-A3B/summary.json').read_text())}
    comparison['35B_A3B_to_4B_bridge_sft_epoch3'] = summary
    suite.save(base / 'comparison.json', comparison)
    suite.save(base / 'complete.json', dict(complete=True, responses=2000, checkpoint_sha256=digest,
        pass_at_4=summary['pass_at_4'], avg_at_4=summary['avg_at_4'],
        scored_sha256=summary['scored_sha256'], finished=time.time()))
    print(json.dumps({k: summary[k] for k in ['pass_at_4', 'avg_at_4', 'clipping_rate']}), flush=True)


def queue(base):
    lock = (base / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_sources(base)
    if (base / 'complete.json').exists():
        done = json.loads((base / 'complete.json').read_text())
        assert done['scored_sha256'] == suite.sha(base / 'scored.jsonl')
        return
    jobs = []

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        suite.save(base / 'phase.json', dict(phase='waiting_for_sft', dependency=str(TRAIN), gpu_ids=GPUS, time=time.time()))
        while not (TRAIN / 'complete.json').exists():
            state = json.loads((TRAIN / 'phase.json').read_text())
            if state['phase'] in ('failed', 'interrupted'):
                raise RuntimeError(f'Bridge SFT did not finish: {state}')
            os.kill(json.loads((TRAIN / 'controller.json').read_text())['pid'], 0)
            time.sleep(5)
        check_sources(base)
        selected_checkpoint(base)
        for attempt in range(12):
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                             '--format=csv,noheader,nounits'], text=True)
            usage = {int(line.split(',')[0]): int(line.split(',')[1]) for line in output.splitlines()}
            if all(usage[gpu] < 512 for gpu in GPUS):
                break
            if attempt == 11:
                raise RuntimeError('GPUs 0-7 occupied; eight-GPU bridge evaluation was not started')
            time.sleep(5)
        started = time.time()
        (base / 'logs').mkdir(exist_ok=True)
        for rank, gpu in enumerate(GPUS):
            log = (base / 'logs' / f'rank-{rank:02d}.log').open('a')
            proc = subprocess.Popen([str(PYTHON), '-u', str(SCRIPT), 'worker', '--root', str(base), '--rank', str(rank)],
                cwd=ROOT, env=suite.environment([gpu]), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            jobs.append((proc, log))
        suite.save(base / 'phase.json', dict(phase='evaluating', gpu_ids=GPUS, pids=[p.pid for p, _ in jobs], started=started))
        suite.wait_jobs(jobs)
        jobs = []
        suite.save(base / 'phase.json', dict(phase='scoring', time=time.time()))
        merge(base, started)
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


def self_test():
    """Exercise penalties, RNG invariance and actual unequal-length hybrid caches."""
    import inspect
    import torch
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    from qwen35_last_aligned_math_bridge import LastAlignedResidualBridge
    # CPU reference kernels; the real workers bind the CUDA FLA implementation.
    modeling.torch_chunk_gated_delta_rule = inspect.unwrap(modeling.torch_chunk_gated_delta_rule)
    modeling.torch_recurrent_gated_delta_rule = inspect.unwrap(modeling.torch_recurrent_gated_delta_rule)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    logits = torch.tensor([[3., 2., -100.]])
    unseen = torch.zeros_like(logits, dtype=torch.bool)
    uniform = torch.tensor([.5])
    assert sample_tokens(logits, unseen, uniform).item() == 0
    repeated = unseen.clone()
    repeated[0, 0] = True
    assert sample_tokens(logits, repeated, uniform).item() == 1
    # Counts of generated repeats do not scale a presence penalty.
    repeated.scatter_(1, torch.tensor([[0]]), True)
    assert sample_tokens(logits, repeated, uniform).item() == 1
    seed_ids = [1234, 1235, 2000]
    together = torch.stack([torch.rand(16, generator=torch.Generator().manual_seed(seed)) for seed in seed_ids])
    for i, seed in enumerate(seed_ids):
        assert torch.equal(together[i], torch.rand(16, generator=torch.Generator().manual_seed(seed)))

    def model(width, depth):
        config = Qwen3_5TextConfig(vocab_size=128, hidden_size=width, intermediate_size=width * 2,
            num_hidden_layers=depth, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            layer_types=(['linear_attention'] * 3 + ['full_attention']) * (depth // 4), full_attention_interval=4,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
            linear_conv_kernel_dim=4, max_position_embeddings=256, eos_token_id=2, pad_token_id=0,
            rope_parameters=dict(rope_type='default', rope_theta=1000., partial_rotary_factor=1.,
                                 mrope_interleaved=True, mrope_section=[2, 2, 4]))
        config._attn_implementation = 'sdpa'
        return Qwen3_5ForCausalLM(config).eval()

    bridge = LastAlignedResidualBridge(model(32, 4), model(64, 8))
    assert bridge.layer_map == [4, 5, 6, 7]
    assert bridge.metadata()['layer_mapping'] == 'last_aligned'
    hidden = tuple(torch.full((1, 3, 64), float(i)) for i in range(9))
    assert all(torch.equal(value, hidden[i + 4]) for i, value in enumerate(bridge.teacher_features(hidden)))
    with torch.no_grad():
        for p in bridge.bridge_parameters():
            p.normal_(0, .01)
    # Compare the actual token-weighted microbatch objective with a complete
    # two-example batch on a deterministic FP32 hybrid model.
    def training_batch(indices):
        prompts = [[1, 4, 5], [1, 6, 7, 8, 9]]
        answers = [[10, 11, 12, 2], [13, 2]]
        length = max(len(prompts[i]) for i in indices)
        ids = torch.tensor([prompts[i] + answers[i] for i in indices])
        labels = ids.clone()
        prefix, mask = [], []
        for row, i in enumerate(indices):
            labels[row, :len(prompts[i])] = -100
            prefix.append(prompts[i] + [0] * (length - len(prompts[i])))
            mask.append([True] * len(prompts[i]) + [False] * (length - len(prompts[i])))
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                    prefix_ids=torch.tensor(prefix), prefix_mask=torch.tensor(mask))
    params = bridge.bridge_parameters()
    local = []
    for i, weight in enumerate([4 / 6, 2 / 6]):
        for p in params:
            p.grad = None
        _, value, _ = bridge.chat_loss(training_batch([i]))
        (value * weight).backward()
        local.append([p.grad.clone() for p in params])
    for p in params:
        p.grad = None
    _, value, _ = bridge.chat_loss(training_batch([0, 1]))
    value.backward()
    expected = [a + b for a, b in zip(*local)]
    numerator = sum((p.grad.double() - target.double()).square().sum() for p, target in zip(params, expected))
    denominator = sum(target.double().square().sum() for target in expected)
    normalization_relative = float((numerator / denominator).sqrt())
    assert normalization_relative < 2e-5, normalization_relative
    assert all(p.grad is None and not p.requires_grad for m in [bridge.student, bridge.teacher] for p in m.parameters())
    rows = [dict(prompt_token_ids=[1, 4, 5]), dict(prompt_token_ids=[1, 6, 7, 8, 9])]
    with torch.inference_mode():
        ids, mask = prompt_tensors(rows, 0, 'cpu')
        report = batch_cache_check(bridge, ids, mask)
        batch = prefill(bridge, ids, mask)
        maximum = 0.
        for i, row in enumerate(rows):
            single_ids, single_mask = prompt_tensors([row], 0, 'cpu')
            single = prefill(bridge, single_ids, single_mask)
            error = float((single.logits[0] - batch.logits[i]).abs().max())
            maximum = max(maximum, error)
            assert error < 1e-5, error
        original_mask = mask
        reply = torch.empty((2, 0), dtype=torch.long)
        for step in range(5):
            nxt = batch.logits[:, -1].argmax(-1, keepdim=True)
            reply = torch.cat([reply, nxt], 1)
            batch, mask = decode_step(bridge.student, nxt, mask, batch.past_key_values)
            full = prefill(bridge, ids, original_mask, extra=reply)
            assert torch.allclose(batch.logits, full.logits, atol=1e-5, rtol=1e-5)
        top_logits = torch.arange(128, dtype=torch.float)[None].expand(3, -1)
        top_seen = torch.zeros_like(top_logits, dtype=torch.bool)
        picks = sample_tokens(top_logits, top_seen, together[:, 0])
        for i in range(3):
            assert picks[i] == sample_tokens(top_logits[i:i + 1], top_seen[i:i + 1], together[i:i + 1, 0])[0]
    print(json.dumps(dict(passed=True, official_presence_penalty=True, independent_rng=True,
        token_weighted_microbatch_gradient_relative_l2=normalization_relative,
        unequal_length_prefill_matches_single_max_error=maximum,
        five_response_steps_match_teacher_forcing=True, cache=report)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['queue', 'prepare', 'worker', 'status', 'self-test'])
    parser.add_argument('--root', type=Path, default=BASE)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    base = args.root.resolve()
    if args.stage == 'self-test':
        self_test()
    elif args.stage == 'worker':
        worker(base, args.rank)
    elif args.stage == 'status':
        state = {name: json.loads((base / name).read_text()) for name in
                 ['phase.json', 'summary.json', 'complete.json'] if (base / name).exists()}
        state['workers'] = [json.loads(p.read_text()) for p in (base / 'generation').glob('*.live.json')]
        print(json.dumps(state, indent=2))
    elif args.stage == 'prepare':
        prepare(base)
    elif args.detach:
        prepare(base)
        controller = base / 'controller.json'
        if controller.exists():
            try:
                os.kill(json.loads(controller.read_text())['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('35B-A3B->4B MATH bridge evaluation queue is already running')
        with (base / 'pipeline.log').open('a') as log:
            proc = subprocess.Popen([str(suite.CPU), '-u', str(SCRIPT), 'queue', '--root', str(base)],
                cwd=ROOT, env=suite.environment([]), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        suite.save(controller, dict(pid=proc.pid, started=time.time()))
        print(json.dumps(dict(pid=proc.pid, root=str(base), dependency=str(TRAIN), gpu_ids=GPUS)), flush=True)
    else:
        prepare(base)
        queue(base)


if __name__ == '__main__':
    main()
