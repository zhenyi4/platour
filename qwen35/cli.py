"""Portable Qwen3.5 GRPO training (one process or torchrun data parallel) and evaluation.

The published histories originate from native distributed/vLLM runs. Portable
training uses HF backward with native vLLM rollouts and does not claim bitwise identity.
Training completes one epoch, selects the validation-best checkpoint and, by
default, tests it once at the end.
"""

import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from .catalog import select
from .checkpoints import (atomic_json, latest, load_step, read, reconcile_history,
                          restore_rng, rng_state, save_step, sha256, valid)
from .scoring import code_score, math_score

DATA = Path(__file__).resolve().parents[1] / 'data/qwen35'


def signed_rows(path, *, expected=None):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if (set(row) - {'index', 'prompt_token_ids', 'gold', 'judge'} or
                        type(row.get('index')) is not int or
                        not isinstance(row.get('prompt_token_ids'), list) or
                        not isinstance(row.get('gold'), str)):
                    raise ValueError('A Qwen3.5 input row differs from the public schema')
                rows.append(row)
    if expected is not None and len(rows) != expected:
        raise ValueError('Original data split size differs')
    if [row['index'] for row in rows] != list(range(len(rows))):
        raise ValueError('Original prompt question order/identities differ')
    return rows


def inputs(case, data_root=DATA):
    from platour_data import ensure_qwen35_data
    ensure_qwen35_data(case.identifier, data_root, 'train')
    data_root = Path(data_root)
    meta = read(data_root / 'manifest.json')
    kind = case.task
    result = {}
    for part, count in (('train', case.train_questions), ('validation', case.val_questions)):
        path = data_root / kind / (part + '.jsonl')
        if sha256(path) != meta[kind][part]['sha256']:
            raise ValueError('Frozen dataset SHA changed: ' + kind + '/' + part)
        result[part] = signed_rows(path, expected=count)
    return result


def evaluate(actor, tokenizer, case, rows, *, count, cap=16384,
             code_cases=None, official_checker=None, docker_image=None,
             strict_train=False, subset=None):
    questions = set()
    passed = set()
    correct = clipped_count = 0
    subset_correct = subset_answers = 0
    for row in rows:
        questions.add(row['index'])
        for sample in range(count):
            seed = 1234 + count * row['index'] + sample
            tokens = actor.generate(row['prompt_token_ids'], seed=seed, cap=cap,
                                    temperature=1., top_p=.95, top_k=20,
                                    presence_penalty=1.5)
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            clipped = len(tokens) >= cap
            clipped_count += clipped
            if case.task == 'dapo':
                accuracy = math_score(row['gold'], text, clipped=clipped,
                                      strict_format=strict_train)
            else:
                if code_cases is None or official_checker is None or not docker_image:
                    raise ValueError('Docker image and official judge/test-case paths required')
                accuracy = code_score(row['judge'], text, case_dir=code_cases,
                                      checker_dir=official_checker, docker_image=docker_image)
            correct += accuracy
            if subset is not None and row['index'] in subset:
                subset_correct += accuracy
                subset_answers += 1
            if accuracy:
                passed.add(row['index'])
    total = len(rows) * count
    if len(questions) != len(rows) or not total:
        raise RuntimeError('Benchmark missing/duplicate question index')
    result = {'complete': True, 'questions': len(questions), 'responses': total,
              'correct': correct, 'accuracy': correct / total,
              'pass_at_k': len(passed) / len(questions), 'samples_per_question': count,
              'clipped': clipped_count, 'clipping_rate': clipped_count / total,
              'gold_missing': 0, 'scoring_timeouts': 0, 'backend': 'transformers'}
    if subset is not None:
        if subset_answers != len(subset) * count:
            raise RuntimeError('AIME Clean80 sample count differs from the fixed question mask')
        result.update(clean80_correct=subset_correct, clean80_responses=subset_answers,
                      clean80_accuracy=subset_correct / subset_answers)
    return result


def evaluate_vllm(decoder, case, rows, *, count, cap, code_cases=None,
                  official_checker=None, docker_image=None, strict_train=False,
                  subset=None):
    """Score the original full denominator from the native vLLM-0.17.1 decoder."""
    if case.task == 'deepcoder' and (code_cases is None or official_checker is None
                                      or not docker_image):
        raise ValueError('Official isolated code judge and cases are required')
    outputs = decoder.generate(rows, count=count, cap=cap)
    if len(outputs) != len(rows) * count or [
        (item['index'], item['sample_index'], item['sampling_seed']) for item in outputs
    ] != [(row['index'], sample, 1234 + count * row['index'] + sample)
          for row in rows for sample in range(count)]:
        raise RuntimeError('Qwen3.5 vLLM output coverage or seeds differ')
    passed = set()
    correct = subset_correct = 0
    clipped = sum(item['tokens'] >= cap for item in outputs)
    for item, row in zip(outputs, (row for row in rows for _ in range(count)), strict=True):
        if case.task == 'dapo':
            value = math_score(row['gold'], item['response'],
                               clipped=item['tokens'] >= cap, strict_format=strict_train)
        else:
            value = code_score(row['judge'], item['response'], case_dir=code_cases,
                               checker_dir=official_checker, docker_image=docker_image)
        correct += value
        if subset is not None and row['index'] in subset:
            subset_correct += value
        if value:
            passed.add(row['index'])
    responses = len(rows) * count
    result = {'complete': True, 'questions': len(rows), 'responses': responses,
              'correct': correct, 'accuracy': correct / responses,
              'question_indices': [row['index'] for row in rows], 'passed_questions': len(passed),
              'pass_at_k': len(passed) / len(rows), 'samples_per_question': count,
              'clipped': clipped, 'clipping_rate': clipped / responses,
              'gold_missing': 0, 'scoring_timeouts': 0, 'backend': 'vllm'}
    if subset is not None:
        if len(subset) > len(rows) or not subset <= {row['index'] for row in rows}:
            raise ValueError('AIME subset differs from the original evaluation set')
        result.update(clean80_correct=subset_correct,
                      clean80_responses=len(subset) * count,
                      clean80_accuracy=subset_correct / (len(subset) * count) if subset else None)
    return result


def _grade(case, row, text, tokens, tokenizer, *, code_cases, official_checker, docker_image,
           cap=16384):
    if case.task == 'dapo':
        return math_score(row['gold'], text, clipped=len(tokens) >= cap, strict_format=True)
    if code_cases is None or official_checker is None or not docker_image:
        raise ValueError('DeepCoder train requires the pinned test cases and sandbox checker')
    verdict = code_score(row['judge'], text, case_dir=code_cases,
                         checker_dir=official_checker, docker_image=docker_image)
    return 0 if len(tokens) >= cap else verdict


def train_step(actor, optimizer, tokenizer, case, batch, step, *,
               code_cases=None, official_checker=None, docker_image=None, cap=16384, receiver=None, output=None, vllm_python=None):
    import torch
    from . import distributed
    from .native_grpo_core import policy_loss
    if not batch:
        raise RuntimeError('Empty training minibatch')
    global_examples = len(batch)
    batch = distributed.local_slice(batch)
    group = getattr(case, 'group_size', 8)
    groups = []
    from .vllm_decode import training_decoder
    from training_protocol import rollout_log_probs
    if output is None or receiver is None:
        raise ValueError('GRPO requires receiver path and a private vLLM rollout directory')
    generated = []
    if batch:
        with torch.no_grad(), training_decoder(actor, case, receiver, output, cap=cap,
                prompt_length=max(len(row['prompt_token_ids']) for row in batch),
                worker_python=vllm_python) as decoder:
            generated = decoder.generate(batch, count=group, cap=cap,
                seed_base=42 + 1000 * step, training=True)
    if [(r['index'], r['sample_index']) for r in generated] != [
            (r['index'], sample) for r in batch for sample in range(group)]:
        raise RuntimeError('GRPO vLLM rollout coverage differs')
    generated = iter(generated)
    for row in batch:
        results = []
        for sample in range(group):
            response = next(generated)
            tokens = response['response_token_ids']
            original_logprobs = rollout_log_probs(response, cap)
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            reward = _grade(case, row, text, tokens, tokenizer, code_cases=code_cases,
                            official_checker=official_checker, docker_image=docker_image,
                            cap=cap)
            results.append((tokens, reward, original_logprobs))
        rewards = torch.tensor([x[1] for x in results], device=actor.student.device,
                               dtype=torch.float32)
        advantages = (rewards - rewards.mean()) / (rewards.std(correction=1) + 1e-6)
        groups.append((row, results, advantages))
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.
    updates = 0
    for row, results, advantages in groups:
        for (tokens, reward, original), advantage in zip(results, advantages):
            if not float(advantage):
                continue
            log_probs = actor.response_log_probs(row['prompt_token_ids'], tokens)
            part, _ = policy_loss(log_probs, advantage, original)
            part = part / (global_examples * group)
            part.backward()
            loss_sum += float(part.detach())
            updates += 1
    parameters = tuple(actor.trainable())
    distributed.all_reduce_gradients(parameters)
    reward_sum = sum(x[1] for _, answers, _ in groups for x in answers)
    mixed = sum(0 < sum(x[1] for x in answers) < group for _, answers, _ in groups)
    loss_sum, updates, reward_sum, mixed, examples = distributed.all_reduce_sum(
        (loss_sum, updates, reward_sum, mixed, len(batch)), parameters[0].device)
    if int(examples) != global_examples:
        raise RuntimeError('Distributed GRPO minibatch coverage differs')
    gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
    if updates:
        optimizer.step()
    return {'train/policy_loss': loss_sum, 'train/grad_norm': gradient,
            'train/accuracy': reward_sum / (global_examples * group),
            'train/responses': global_examples * group, 'train/examples': global_examples,
            'train/learning_rate': optimizer.param_groups[0]['lr'],
            'train/mixed_groups': int(mixed)}


def _model(case, args, *, for_training=False):
    import torch
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    from .model import Actor, load_text
    from . import distributed
    base = load_text(args.receiver, device=args.device)
    teacher = None
    if case.method == 'bridge':
        if args.sender is None:
            raise ValueError('This bridge requires an independently downloaded frozen public sender')
        # One process may spread the sender over all visible GPUs; a torchrun rank
        # keeps its full replica on its own local GPU.
        placement = 'auto' if distributed.world() == 1 else None
        from qwen3.eval_parallel import SHARD_ENV, sender_device_map
        if SHARD_ENV in os.environ or 'PLATOUR_GPU_GROUP' in os.environ:
            configuration = json.loads((Path(args.sender) / 'config.json').read_text())
            text_config = configuration.get('text_config', configuration)
            family = 'qwen3' if configuration.get('model_type') == 'qwen3_5_text' else 'qwen35'
            placement = sender_device_map(family, text_config['num_hidden_layers'], torch.cuda.device_count())
        teacher = load_text(args.sender, device=args.device, device_map=placement)
        if (teacher.config.hidden_size != case.teacher_dim or
                teacher.config.num_hidden_layers <= max(case.layer_map)):
            raise ValueError('Sender topology differs from released Qwen3.5 checkpoint')
    initial_template = None
    if case.method == 'random':
        if for_training and args.template_init is None:
            raise ValueError('Random training must initialize V from TRAIN-only template statistics')
        if for_training:
            with safe_open_public(args.template_init) as handle:
                meta = handle.metadata() or {}
                if (meta.get('task') != case.task or meta.get('sample_count') != '500' or
                        set(handle.keys()) != {'virtual_hidden', 'template_token_ids'}):
                    raise ValueError('Random V must be fresh 500-train-only statistics')
            origin = Path(str(args.template_init) + '.manifest.json')
            evidence = read(origin)
            if (not evidence.get('complete') or evidence.get('task') != case.task or
                    evidence.get('train_sha256') != sha256(Path(args.data_root) / case.task / 'train.jsonl') or
                    evidence.get('checkpoint_sha256') != sha256(args.template_init)):
                raise ValueError('Random initial V/train-only provenance differs')
            initial_template = load_file(args.template_init, device='cpu')
        else:
            with safe_open_public(args.weights) as handle:
                initial_template = {key: handle.get_tensor(key)
                                    for key in ('virtual_hidden', 'template_token_ids')}
    actor = Actor(case, base, teacher=teacher, template=initial_template)
    tokenizer = AutoTokenizer.from_pretrained(str(args.receiver), local_files_only=True)
    return actor, tokenizer


class safe_open_public:
    def __init__(self, file):
        self.file = file

    def __enter__(self):
        from safetensors import safe_open
        self.source = safe_open(self.file, framework='pt', device='cpu')
        return self.source

    def __exit__(self, *args):
        del self.source


def verify_code_training_environment(case, args):
    """Fail before model loading if the original isolated code judge is absent."""
    if case.task != 'deepcoder':
        return
    if args.code_cases is None or args.official_checker is None or not args.docker_image:
        raise ValueError('DeepCoder training needs authorized cases, checker and Docker image')
    cases_input, checker_input = Path(args.code_cases), Path(args.official_checker)
    if cases_input.is_symlink() or checker_input.is_symlink():
        raise ValueError('Original code judge root may not redirect through a symlink')
    cases, checker = cases_input.resolve(), checker_input.resolve()
    checker_file = checker / 'testing_util.py'
    if not cases.is_dir() or checker_file.is_symlink() or not checker_file.is_file():
        raise FileNotFoundError('Original isolated code judge cases/checker are unavailable')
    service = os.environ.get('PLATOUR_CODE_JUDGE_SERVICE')
    if service:
        from .scoring import judge_service_info
        info = judge_service_info(service)
        if (info.get('case_root') != str(cases) or info.get('checker_root') != str(checker) or
                info.get('image') != args.docker_image or
                str(case.identifier) not in info.get('supported_experiments', ()) or
                os.environ.get('PLATOUR_CODE_JUDGE_EXPERIMENT') != str(case.identifier)):
            raise RuntimeError('Isolated code judge service does not match authorized inputs')
        image_id = info.get('image_id', '')
    else:
        try:
            status = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}',
                                     args.docker_image], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError('Isolated Docker judge daemon or image is unavailable') from error
        image_id = status.stdout.strip() if not status.returncode else ''
    if (not image_id.startswith('sha256:') or len(image_id) != 71 or
            any(ch not in '0123456789abcdef' for ch in image_id[7:])):
        raise RuntimeError('Isolated Docker judge daemon or image is unavailable')
    proof = cases.parent / 'complete.json'
    checker_hash = sha256(checker_file)
    if service and (info.get('checker_sha256') != checker_hash or
                    info.get('case_proof_sha256') != sha256(proof)):
        raise RuntimeError('Isolated code judge service checker or case seal differs')
    if proof.is_file():
        sealed = read(proof)
        if (not sealed.get('complete') or not sealed.get('per_question_sha256_verified') or
                sealed.get('training_validation_cases') != 18873 + 201 or
                sealed.get('lcb_v6_cases') != 175 or
                sealed.get('checker_sha256') != checker_hash):
            raise RuntimeError('Sealed original judge cases/checker identity differs')
    from .scoring import SAFE_KEY
    identity = hashlib.sha256((image_id + ':' + checker_hash).encode())
    for split, rows in inputs(case, args.data_root).items():
        for row in rows:
            spec = row.get('judge')
            if (not isinstance(spec, dict) or spec.get('kind') != 'deepcoder' or
                    not isinstance(spec.get('key'), str) or not SAFE_KEY.fullmatch(spec['key']) or
                    not isinstance(spec.get('sample_sha256'), str) or
                    len(spec['sample_sha256']) != 64):
                raise ValueError('Original signed DeepCoder judge key differs')
            case_file = cases / (spec['key'] + '.json')
            if (case_file.is_symlink() or not case_file.is_file() or
                    not case_file.resolve().is_relative_to(cases)):
                raise FileNotFoundError('Original signed DeepCoder train cases are missing')
            if sha256(case_file) != spec['sample_sha256']:
                raise RuntimeError('Original signed DeepCoder case SHA differs: ' + split)
            identity.update((split + ':' + spec['key'] + ':' + spec['sample_sha256']).encode())
    return identity.hexdigest()


def verify_code_evaluation_environment(case, args):
    """Verify every original LCB test case and the isolated judge before GPU inference."""
    if case.task != 'deepcoder':
        return
    if args.code_cases is None or args.official_checker is None or not args.docker_image:
        raise ValueError('DeepCoder evaluation requires authorized cases, checker and image')
    cases_root, checker_root = Path(args.code_cases), Path(args.official_checker)
    if cases_root.is_symlink() or checker_root.is_symlink():
        raise ValueError('Original judge paths may not redirect through a symlink')
    cases, checker = cases_root.resolve(), checker_root.resolve()
    checker_file = checker / 'testing_util.py'
    if not cases.is_dir() or checker_file.is_symlink() or not checker_file.is_file():
        raise FileNotFoundError('Original isolated checker or LCB cases are unavailable')
    service = os.environ.get('PLATOUR_CODE_JUDGE_SERVICE')
    if service:
        from .scoring import judge_service_info
        info = judge_service_info(service)
        if (info.get('case_root') != str(cases) or info.get('checker_root') != str(checker) or
                info.get('image') != args.docker_image or
                str(case.identifier) not in info.get('supported_experiments', ()) or
                os.environ.get('PLATOUR_CODE_JUDGE_EXPERIMENT') != str(case.identifier)):
            raise RuntimeError('Isolated code judge service does not match authorized inputs')
        image = info.get('image_id', '')
    else:
        try:
            inspected = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}',
                                        args.docker_image], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError('Original isolated Docker code judge is unavailable') from error
        image = inspected.stdout.strip() if not inspected.returncode else ''
    if (not image.startswith('sha256:') or len(image) != 71 or
            any(ch not in '0123456789abcdef' for ch in image[7:])):
        raise RuntimeError('Original isolated Docker code judge is unavailable')
    if service and (info.get('checker_sha256') != sha256(checker_file) or
                    info.get('case_proof_sha256') != sha256(cases.parent / 'complete.json')):
        raise RuntimeError('Isolated code judge service checker or case seal differs')
    base = Path(args.data_root) / 'deepcoder'
    manifest = read(Path(args.data_root) / 'manifest.json')['benchmark']['lcb']
    source = base / 'lcb.jsonl'
    if sha256(source) != manifest['sha256']:
        raise RuntimeError('Original LCB v6 test prompts changed')
    rows = signed_rows(source, expected=manifest['rows'])
    if len(rows) != 175:
        raise RuntimeError('Original LCB v6 test denominator differs')
    from .scoring import SAFE_KEY
    seen = set()
    sealed_file = cases.parent / 'complete.json'
    sealed = read(sealed_file) if sealed_file.is_file() else None
    if sealed and (not sealed.get('complete') or sealed.get('checker_sha256') != sha256(checker_file) or
                   sealed.get('lcb_v6_cases') != 175):
        raise RuntimeError('Sealed original LCB judge/checker identity differs')
    for row in rows:
        spec = row.get('judge')
        if (not isinstance(spec, dict) or spec.get('kind') != 'lcb' or
                not isinstance(spec.get('key'), str) or not SAFE_KEY.fullmatch(spec['key']) or
                spec['key'] in seen):
            raise ValueError('Original LCB v6 question/judge keys differ')
        seen.add(spec['key'])
        entry = cases / (spec['key'] + '.json')
        if entry.is_symlink() or not entry.is_file() or not entry.resolve().is_relative_to(cases):
            raise FileNotFoundError('Original LCB v6 evaluation cases are missing')
        if sealed and sha256(entry) != sealed.get('lcb_case_sha256', {}).get(entry.name):
            raise RuntimeError('Sealed original LCB v6 evaluation case SHA differs')


def run_train(case, args):
    from .vllm_decode import training_decoder_scope
    eligible = getattr(case, 'method', None) in ('bridge', 'random')
    mode = getattr(args, 'rollout_worker_mode', None) or ('persistent' if eligible else 'restart')
    if mode not in ('persistent', 'restart') or (mode == 'persistent' and not eligible):
        raise ValueError('Persistent rollout workers require a frozen receiver')
    # Pool creation is lazy: preflight/data checks still precede model loading.
    # RANK is provided by the launcher before distributed.init runs below.
    output = Path(args.output) / 'rollout-worker'
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        output /= f"rank-{int(os.environ['RANK']):02d}"
    scope = training_decoder_scope(output, max_model_len=0, mode=mode) if eligible else nullcontext()
    with scope:
        return _run_train(case, args, rollout_worker_mode=mode)


def _run_train(case, args, *, rollout_worker_mode):
    """Portable GRPO recipe.

    Budgets default to the published recipe values (64 questions x 8 responses per update,
    16,384-token cap); a smaller case object, e.g. the CPU simulation, may set
    ``batch_size``, ``group_size`` and ``response_cap``.
    """
    if getattr(args, 'backend', 'vllm') != 'vllm':
        raise ValueError('Generated validation and TEST require vLLM')
    smoke = bool(getattr(args, 'smoke', False))
    if smoke and case.identifier not in (26, 29):
        raise ValueError('Portable code smoke is scoped to experiments 26 and 29')
    cap = getattr(args, 'smoke_cap', 512) if smoke else getattr(case, 'response_cap', 16384)
    if type(cap) is not int or not 1 <= cap <= 16384:
        raise ValueError('Smoke output cap must be in 1..16384')
    batch_size = getattr(case, 'batch_size', 64)
    judge_identity = verify_code_training_environment(case, args)
    import torch
    from . import distributed
    from . import selection as chooser
    rank_id, world, local = distributed.init()
    if world > 1:
        if case.sender is not None and 'FP8' in case.sender and 'PLATOUR_GPU_GROUP' not in os.environ:
            raise ValueError('A 397B FP8 sender cannot be replicated per rank; use --train-backend '
                             'native-multinode for IDs 20/25')
        args.device = distributed.device(local)
    output = Path(args.output)
    data = inputs(case, args.data_root)
    from .vllm_decode import reserve_training_context
    reserve_training_context(max(3072, cap + max((len(row['prompt_token_ids'])
        for row in data['train'] + data['validation']), default=0)))
    from training_protocol import budget
    _, formal_steps = budget(len(data['train']), batch_size, 1, 1)
    if not smoke and formal_steps != case.steps:
        raise ValueError('Published validation schedule differs from the one-epoch TRAIN budget')
    protocol = dict(version=3, selection_policy=chooser.POLICY, epochs=1, total_steps=2 if smoke else formal_steps,
        batch_size=batch_size, group_size=getattr(case, 'group_size', 8), rollout_backend='vllm', evaluation_backend='vllm',
        smoke_only=smoke, response_cap=cap, validation_samples=2 if smoke else case.val_samples,
        train_sampling=dict(temperature=1., top_p=1., top_k=-1, presence_penalty=0.))
    protocol_path = output / 'training_protocol.json'
    if case.method in ('bridge', 'random'):
        legacy_restart = (rollout_worker_mode == 'restart' and protocol_path.is_file()
                          and 'rollout_worker_mode' not in read(protocol_path))
        if not legacy_restart:
            protocol['rollout_worker_mode'] = rollout_worker_mode
    from qwen3.parallel import sign_topology
    sign_topology(protocol)
    def prepare_protocol():
        if protocol_path.exists():
            if read(protocol_path) != protocol:
                raise RuntimeError('Training backend/budget protocol changed; use a new run')
        elif (output / 'checkpoints').exists() or (output / 'complete.json').exists():
            raise RuntimeError('Resume lacks the current vLLM training protocol')
        else:
            atomic_json(protocol_path, protocol)
    distributed.on_rank_zero(prepare_protocol)
    torch.manual_seed(42)
    random.seed(42)
    identity = sha256(Path(args.data_root) / 'manifest.json') + ':' + case.tag + ':' + sha256(protocol_path)
    if smoke:
        identity += ':portable-code-smoke-v2-cap' + str(cap)
    if case.task == 'deepcoder':
        if not isinstance(judge_identity, str) or len(judge_identity) != 64:
            raise RuntimeError('Missing original isolated judge identity for DeepCoder resume')
        identity += ':' + judge_identity
    final_test = not smoke and not getattr(args, 'no_final_test', False)
    terminal = output / 'complete.json'
    if terminal.is_file():
        if final_test:
            _final_test(case, args, output)
        return
    actor, tokenizer = _model(case, args, for_training=True)
    optimizer = torch.optim.AdamW(actor.trainable(), lr=case.learning_rate,
                                  betas=(.9, .999), weight_decay=0.)
    if rank_id == 0:
        output.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    protected = ({0, 1, 2} if smoke else set(case.checkpoint_steps))
    if case.identifier == 27:
        protected.add(294)

    def commit(step, record):
        states = distributed.gather(rng_state())
        distributed.on_rank_zero(lambda: save_step(
            output, step, actor, optimizer, record, protected=protected,
            identity=identity, rank_rng=states))

    prior = latest(output)
    if prior is None:
        if (output / 'metrics.jsonl').is_file() or any((output / 'checkpoints').glob('*')):
            raise FileExistsError('Use a new output root or a signed resumable checkpoint')
        commit(0, None)
        start = 0
    else:
        start, record = load_step(prior, actor, optimizer, expected_identity=identity,
                                  rank=rank_id, world=world)
        distributed.on_rank_zero(lambda: reconcile_history(output, start, record))
    evaluation = output / 'evaluation'
    if rank_id == 0:
        evaluation.mkdir(exist_ok=True)
    distributed.barrier()

    def run_validation(step):
        validate_once(step)

    def validate_once(step):
        destination = evaluation / f'step-{step:04d}.json'
        if destination.is_file():
            if not read(destination).get('complete'):
                raise RuntimeError('Previously attempted validation is incomplete')
            return
        checkpoint = output / 'checkpoints' / f'step-{step:04d}'
        if not checkpoint.is_dir() or not read(checkpoint / 'complete.json').get('complete'):
            raise RuntimeError('Must sign optimizer and RNG checkpoint BEFORE validation')
        saved = rng_state()
        try:
            if getattr(args, 'backend', 'vllm') == 'vllm':
                from tempfile import TemporaryDirectory
                from safetensors.torch import save_file
                from .vllm_decode import decoder_session
                if smoke:
                    evidence = evaluation / f'validation-{step:04d}-attempt-{time.time_ns()}'
                    evidence.mkdir()
                    validation_dir = nullcontext(str(evidence))
                else:
                    validation_dir = TemporaryDirectory(prefix=f'validation-{step:04d}-',
                                                        dir=evaluation)
                with validation_dir as folder:
                    temporary = Path(folder)
                    weight = temporary / (case.tag + '.safetensors')
                    save_file({name: tensor.detach().cpu().contiguous()
                               for name, tensor in actor.inference_state().items()}, weight)
                    with decoder_session(case, args.receiver, weight, temporary,
                            actor=actor if case.method in ('bridge', 'random') else None,
                            device=next(iter(actor.trainable())).device,
                            worker_python=args.vllm_python,
                            memory_fraction=(args.vllm_memory_fraction if args.vllm_memory_fraction is not None
                                             else .30 if case.task == 'dapo' else .22),
                            max_model_len=20480,
                            max_num_seqs=8 if case.task == 'dapo' else 4) as decoder:
                        from qwen3.parallel import collective_evaluation
                        count = 2 if smoke else case.val_samples
                        scores = collective_evaluation(
                            data['validation'][:2] if smoke else data['validation'], count, temporary,
                            lambda rows, path: evaluate_vllm(decoder, case, rows,
                                count=count, cap=cap, code_cases=args.code_cases,
                                official_checker=args.official_checker, docker_image=args.docker_image))
            else:
                scores = evaluate(actor, tokenizer, case,
                    data['validation'][:2] if smoke else data['validation'],
                    count=2 if smoke else case.val_samples, cap=cap,
                    code_cases=args.code_cases, official_checker=args.official_checker,
                    docker_image=args.docker_image)
        finally:
            restore_rng(saved)
        distributed.on_rank_zero(lambda: atomic_json(destination, {'step': step, **scores}))
    validation_steps = (1, 2) if smoke else case.validation_steps
    if start == 0 or start in validation_steps:
        run_validation(start)
    total = 2 if smoke else formal_steps

    def finish(step):
        def write():
            if not (evaluation / f'step-{step:04d}.json').is_file():
                raise RuntimeError('Cannot sign final training without its complete validation')
            chosen = None
            if not smoke:
                state = chooser.replay(case, output, step, steps=validation_steps)
                chosen = chooser.selection(case, output, state)
                atomic_json(output / 'selection.json', chosen)
            atomic_json(terminal, {'complete': True, 'steps': step,
                                   'experiment': case.identifier, 'smoke_only': smoke,
                                   'original_total_steps': case.steps, 'response_cap': cap,
                                   'world_size': world,
                                   'selected_step': None if chosen is None else chosen['selected_step'],
                                   'checkpoint': f'checkpoints/step-{step:04d}'})
        distributed.on_rank_zero(write)

    if start == total:
        finish(start)
    else:
        order = torch.randperm(case.train_questions,
                               generator=torch.Generator().manual_seed(42)).tolist()
        for step in range(start + 1, total + 1):
            start_index = (step - 1) * batch_size
            indices = order[start_index:(start_index + 2 if smoke else step * batch_size)]
            metrics = train_step(actor, optimizer, tokenizer, case,
                                 [data['train'][i] for i in indices], step,
                                 code_cases=args.code_cases, official_checker=args.official_checker,
                                 docker_image=args.docker_image, cap=cap, receiver=args.receiver,
                                 output=output / 'rollouts' / f'step-{step:04d}-rank-{rank_id}',
                                 vllm_python=args.vllm_python)
            record = {'step': step, **metrics}
            commit(step, record)

            def log():
                with (output / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, sort_keys=True) + '\n')
            distributed.on_rank_zero(log)
            if step in validation_steps:
                run_validation(step)
        finish(total)
    if final_test:
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _final_test(case, args, output, actor=actor, tokenizer=tokenizer)


def _final_test(case, args, run, *, actor=None, tokenizer=None):
    """Test once, at the signed validation-best checkpoint of a finished run."""
    import copy
    destination = Path(run) / 'test'
    if (destination / 'summary.json').is_file():
        return read(destination / 'summary.json')
    from qwen3.parallel import root_call
    def archive():
        if destination.exists():
            recovery = Path(run) / 'recovery'
            recovery.mkdir(exist_ok=True)
            destination.rename(recovery / f'incomplete-test-{time.time_ns()}')
    root_call(archive)
    test_args = copy.copy(args)
    test_args.run, test_args.output, test_args.weights = Path(run), destination, None
    test_args.smoke = False
    return run_eval(case, test_args, actor=actor, tokenizer=tokenizer)


def _match_state(expected, found):
    """Map a training run's tensors onto the release inference keys; never guess shapes."""
    for prefix in ('', 'P.'):
        candidate = {prefix + key: value for key, value in found.items()}
        if set(candidate) == set(expected) and all(
                tuple(candidate[name].shape) == tuple(expected[name].shape) for name in expected):
            return {name: candidate[name].to(expected[name].dtype) for name in expected}
    raise ValueError('Selected training checkpoint tensors do not match this method/architecture')


def _selected_state(case, run):
    """Return (inference tensors or None for the untouched receiver, provenance)."""
    import torch
    from safetensors.torch import load_file
    run = Path(run).absolute()
    if (run / 'selection.json').is_file():
        chosen = read(run / 'selection.json')
        step = chosen['selected_step']
        checkpoint = run / 'checkpoints' / f'step-{step:04d}'
        if (chosen.get('experiment') != case.identifier or
                not valid(checkpoint, expected_step=step) or
                case.tag not in str(read(checkpoint / 'checkpoint.json').get('identity'))):
            raise RuntimeError('Selected portable checkpoint is unsigned or from another experiment')
        return load_file(checkpoint / 'model.safetensors', device='cpu'), {
            'source_run': str(run), 'selected_step': step, 'selection_policy': chosen['policy'],
            'validation_accuracy': chosen['validation_accuracy'],
            'selected_checkpoint_sha256': sha256(checkpoint / 'model.safetensors')}
    for folder in (run / 'train', run / 'run/train', run / 'run', run):
        early = folder / 'early_stop.json'
        terminal = folder / 'complete.json'
        record = read(early) if early.is_file() else read(terminal) if terminal.is_file() else {}
        if 'best_step' not in record:
            continue
        step = record['best_step']
        if case.method == 'full':
            source = folder / 'weights' / f'step-{step:04d}' / 'model.safetensors'
        elif case.method == 'lora':
            source = folder / 'checkpoints' / f'step-{step:04d}' / 'adapter.pt'
        else:
            source = folder / ('initial' if step == 0 else f'checkpoints/step-{step:04d}') / 'bridge.pt'
        provenance = {'source_run': str(run), 'selected_step': step,
                      'selection_policy': 'native_early_stop_best_step'}
        if case.method in ('full', 'lora') and step == 0 and not source.is_file():
            return None, dict(provenance, selected_checkpoint_sha256=None)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError('Native validation-best checkpoint is missing: ' + str(source))
        tensors = (load_file(source, device='cpu') if source.suffix == '.safetensors' else
                   torch.load(source, map_location='cpu', weights_only=True))
        if isinstance(tensors, dict) and set(tensors) == {'P'}:
            tensors = tensors['P']
        return tensors, dict(provenance, selected_checkpoint_sha256=sha256(source))
    raise FileNotFoundError('No validation-best selection in this training run')


def run_eval(case, args, *, actor=None, tokenizer=None, _local=False):
    from qwen3.parallel import dist, root_call
    if dist().world() > 1 and not _local:
        return _parallel_test(case, args, actor=actor, tokenizer=tokenizer)
    from safetensors.torch import save_file
    if getattr(args, 'backend', 'vllm') != 'vllm':
        raise ValueError('Generated validation and TEST require vLLM')
    from qwen3.eval_parallel import evaluation_suites, three_suite_macro
    suites = evaluation_suites(case.task, getattr(args, 'eval_suites', None))
    smoke = bool(getattr(args, 'smoke', False))
    if smoke and case.identifier not in (26, 29):
        raise ValueError('Portable code test smoke is scoped to experiments 26 and 29')
    cap = getattr(args, 'smoke_cap', 512) if smoke else (
        getattr(case, 'test_cap', None) or (32768 if case.task == 'dapo' else 16384))
    if smoke and (type(cap) is not int or not 1 <= cap <= 16384):
        raise ValueError('Smoke output cap must be in 1..16384')
    override = getattr(args, 'response_cap', None)
    if override is not None:
        if smoke or not 1 <= override <= cap:
            raise ValueError('--response-cap requires a non-smoke evaluation and cannot exceed the published cap')
        cap = override
    from platour_data import ensure_qwen35_data
    ensure_qwen35_data(case.identifier, getattr(args, 'data_root', DATA), 'evaluate')
    verify_code_evaluation_environment(case, args)
    source_run = getattr(args, 'run', None)
    if source_run is not None and args.weights is not None:
        raise ValueError('Choose either --weights or a training --run, not both')
    selected = provenance = None
    if source_run is not None:
        selected, provenance = _selected_state(case, source_run)
    else:
        from platour_exps import resolve_weight_input
        args.weights = resolve_weight_input(case.identifier, args.weights)
    from platour_exps import resolve_model_input
    for name in ('receiver', 'sender'):
        if getattr(args, name, None) is not None:
            setattr(args, name, resolve_model_input(getattr(args, name)))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a new result directory; never overwrite an earlier test')
    output.mkdir(parents=True)
    backend = getattr(args, 'backend', 'vllm')
    if backend != 'vllm':
        raise ValueError('TEST answer generation requires vLLM')
    if source_run is not None:
        args.weights = output / 'selected' / (case.tag + '.safetensors')
        args.weights.parent.mkdir()
        if selected is None:
            from .model import load_text
            selected = {key: value.detach().cpu() for key, value in
                        load_text(args.receiver, device='cpu').state_dict().items()}
        if actor is not None:
            selected = _match_state(actor.inference_state(), selected)
        save_file({key: value.contiguous() for key, value in selected.items()}, args.weights)
    if backend == 'vllm' and case.method in ('full', 'lora'):
        from transformers import AutoTokenizer
        actor = None
        tokenizer = AutoTokenizer.from_pretrained(str(args.receiver), local_files_only=True)
    else:
        if actor is None:
            actor, tokenizer = _model(case, args)
        if source_run is None:
            actor.load_public(args.weights)
        else:
            actor.load_inference_state(_match_state(actor.inference_state(), selected))
    base = Path(args.data_root) / case.task
    mask = set(read(base / 'clean80.json')) if case.task == 'dapo' else None
    if mask is not None and (len(mask) != 80 or not all(0 <= index < 135 for index in mask)):
        raise RuntimeError('Frozen AIME Clean80 question mask differs')
    results = {}
    if backend == 'vllm':
        from .vllm_decode import decoder_session
        import torch
        from qwen3.parallel import local_device
        decoder_context = decoder_session(case, args.receiver, args.weights, output, actor=actor,
            device=torch.device(local_device(getattr(args, 'device', 'cuda:0'))),
            worker_python=getattr(args, 'vllm_python', None),
            memory_fraction=(args.vllm_memory_fraction if args.vllm_memory_fraction is not None
                             else .30 if case.task == 'dapo' else .22),
            max_model_len=cap + 4096,
            max_num_seqs=(8 if case.task == 'dapo' else
                          4 if case.method in ('bridge', 'random') else 16))
    else:
        decoder_context = nullcontext(None)
    with decoder_context as decoder:
        if decoder is not None:
            atomic_json(output / 'runtime.json', decoder.runtime)
        for name in suites:
            suffix = name + ('_template_adapted' if case.method == 'random' and
                             name in ('humaneval', 'mbpp') else '')
            rows = signed_rows(base / (suffix + '.jsonl'))
            expected = read(Path(args.data_root) / 'manifest.json')['benchmark'][suffix]
            if sha256(base / (suffix + '.jsonl')) != expected['sha256'] or len(rows) != expected['rows']:
                raise RuntimeError('Published original evaluation input identity differs: ' + suffix)
            if smoke:
                rows = rows[:2]
            from qwen3.eval_parallel import shard_rows
            rows = shard_rows(rows)
            if _local:
                rows = dist().local_slice(rows)
            samples = (2 if smoke else getattr(case, 'test_samples', None) or
                       (8 if case.task == 'deepcoder' else 4 if name == 'imo' else 16))
            options = dict(count=samples, cap=cap,
                           code_cases=args.code_cases, official_checker=args.official_checker,
                           docker_image=args.docker_image,
                           subset=mask.intersection(row['index'] for row in rows) if name == 'aime' else None)
            results[name] = (evaluate_vllm(decoder, case, rows, **options)
                             if backend == 'vllm' else evaluate(actor, tokenizer, case, rows, **options))
            atomic_json(output / f'{name}.json', results[name])
    macro = three_suite_macro(results)
    atomic_json(output / 'summary.json', {'complete': True, 'experiment': case.identifier,
                                           'smoke_only': smoke, 'response_cap': cap,
                                           'three_suite_macro': macro,
                                           'checkpoint_source': ('validation_best_of_training_run'
                                                                 if provenance else 'published_release'),
                                           'selection': provenance,
                                           'results': results,
                                           'math_clean80_from_same_answers': mask is not None,
                                           'template_adapted_code_suites':
                                               ('humaneval', 'mbpp') if case.task == 'deepcoder'
                                               and case.method == 'random' else []})


def _parallel_test(case, args, *, actor=None, tokenizer=None):
    """All training ranks test the selected state, with separate decoder outputs."""
    import copy
    from qwen3.parallel import dist, root_call, rank_output
    from qwen3.eval_parallel import evaluation_suites, merged_metrics, three_suite_macro
    root = Path(args.output)
    def create():
        if root.exists():
            raise FileExistsError('Use a fresh distributed TEST directory')
        root.mkdir(parents=True)
    root_call(create)
    local_args = copy.copy(args)
    local_args.output = rank_output(root)
    error = part = None
    try:
        run_eval(case, local_args, actor=actor, tokenizer=tokenizer, _local=True)
        part = read(local_args.output / 'summary.json')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    gathered = dist().gather((error, part))
    if any(e for e, _ in gathered):
        raise RuntimeError('Distributed TEST failed: ' + str([e for e, _ in gathered if e]))
    parts = [p for _, p in gathered]
    if any(p['selection'] != parts[0]['selection'] for p in parts):
        raise RuntimeError('Distributed TEST checkpoint selection differs')
    result = dict(parts[0], data_parallel_replicas=dist().world(), results={})
    for name in evaluation_suites(case.task, getattr(args, 'eval_suites', None)):
        suffix = name + ('_template_adapted' if case.method == 'random' and name in ('humaneval', 'mbpp') else '')
        rows = signed_rows(Path(args.data_root) / case.task / (suffix + '.jsonl'))
        samples = parts[0]['results'][name]['samples_per_question']
        result['results'][name] = merged_metrics([p['results'][name] for p in parts],
                                                 [r['index'] for r in rows], samples)
        root_call(lambda: atomic_json(root / f'{name}.json', result['results'][name]))
    result['three_suite_macro'] = three_suite_macro(result['results'])
    root_call(lambda: atomic_json(root / 'summary.json', result))
    return result


def main():
    from qwen3.parallel import maybe_launch
    if maybe_launch(sys.argv[1:]):
        return
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('train', 'evaluate', 'init-random'):
        cmd = sub.add_parser(action)
        cmd.add_argument('--experiment', type=int, choices=range(19, 31), required=True)
        cmd.add_argument('--receiver', required=True,
                         type=str if action == 'evaluate' else Path,
                         help='Local backbone directory or, for evaluation, HF repository ID')
        cmd.add_argument('--sender', type=str if action == 'evaluate' else Path,
                         help='Local backbone directory or, for evaluation, HF repository ID')
        cmd.add_argument('--output', type=Path, required=True)
        cmd.add_argument('--data-root', type=Path, default=DATA)
        cmd.add_argument('--device', default='cuda:0')
        cmd.add_argument('--code-cases', type=Path)
        cmd.add_argument('--official-checker', type=Path)
        cmd.add_argument('--docker-image', default='lcb-checker:local')
        if action != 'init-random':
            cmd.add_argument('--backend', choices=('vllm',), default='vllm',
                             help='Native vLLM 0.17.1 is the default for validation and tests')
            cmd.add_argument('--vllm-python', type=Path,
                             help='Optional vLLM interpreter override; defaults to current project Python')
            cmd.add_argument('--vllm-memory-fraction', type=float,
                             help='Override task-specific native vLLM memory fraction')
            cmd.add_argument('--smoke', action='store_true',
                             help='ID26/29 portable train and three-suite test: only tiny signed subsets')
            cmd.add_argument('--smoke-cap', type=int, default=512,
                             help='ID26/29 portable smoke-only maximum generated tokens (1..16384)')
        if action == 'train':
            from qwen3.parallel import add_arguments
            add_arguments(cmd)
            cmd.add_argument('--train-backend',
                             choices=('portable', 'historical-native', 'native-multinode'),
                             default='portable',
                             help='19/21–24: one-host native trainer; 20/25: two-host TP4 via native-multinode')
            cmd.add_argument('--rollout-worker-mode', choices=('persistent', 'restart'),
                             help='Portable frozen-receiver GRPO: reuse workers by default; restart for legacy behavior')
            cmd.add_argument('--resume-step', type=int,
                             help='Same-run signed native validation or rolling optimizer step')
            cmd.add_argument('--resume-latest', action='store_true',
                             help='23/24 or 20/25: choose newest complete model/optimizer/all-rank RNG step')
            cmd.add_argument('--native-stage', choices=('plan', 'prepare', 'probe', 'smoke', 'train'),
                             default='plan', help='Two-host original 20/25 stage; plan does not touch GPU')
            cmd.add_argument('--node-rank', type=int, choices=(0, 1))
            cmd.add_argument('--master-addr', help='Reachable node-0 address for both authorized hosts')
            cmd.add_argument('--master-port', type=int, default=29500)
            cmd.add_argument('--native-attempt', default='first')
            cmd.add_argument('--judge-service-root', type=Path,
                             help='ID25 only: running pinned CPU Docker judge on shared storage')
            cmd.add_argument('--scoring-python', type=Path,
                             help='Optional math-verify interpreter override; defaults to current project Python')
            cmd.add_argument('--statistics-root', type=Path,
                             help='ID22: original 500-TRAIN-only scalar moments/template root; no inference checkpoint')
            cmd.add_argument('--template-init', type=Path,
                             help='Fresh, train-only seeded V; never initialize from a selected test checkpoint')
            cmd.add_argument('--no-final-test', action='store_true',
                             help='Skip the default one-time test of the validation-best checkpoint')
        elif action == 'evaluate':
            from qwen3.eval_parallel import add_arguments
            add_arguments(cmd)
            cmd.add_argument('--response-cap', type=int,
                             help='Optional smaller generation cap for this evaluation; recorded in summary.json')
            cmd.add_argument('--eval-suites', nargs='+', choices=('aime', 'hmmt', 'imo', 'lcb', 'humaneval', 'mbpp'),
                             help='Evaluate only these suites; defaults to all three suites for the experiment')
            cmd.add_argument('--weights',
                             help='Local checkpoint or HF repository ID; defaults to the verified release')
            cmd.add_argument('--run', type=Path,
                             help='Test the validation-best checkpoint of this training output instead')
            cmd.set_defaults(template_init=None)
        else:
            cmd.add_argument('--native-statistics-root', type=Path,
                             help='ID22 only: new private destination for train-only moments needed by native GRPO')
    args = parser.parse_args()
    from qwen3.eval_parallel import configure_worker, launch
    configure_worker()
    if args.action == 'evaluate':
        if args.n_gpu is not None:
            launch(args)
            return
        if args.gpus_per_replica is not None:
            parser.error('--gpus-per-replica requires --n-gpu')
    case = select(args.experiment)
    historical = args.action == 'train' and args.train_backend == 'historical-native'
    multinode = args.action == 'train' and args.train_backend == 'native-multinode'
    if historical or multinode:
        args.vllm_python = args.vllm_python or Path(sys.executable)
        args.scoring_python = args.scoring_python or Path(sys.executable)
    if (historical or multinode) and args.rollout_worker_mode is not None:
        parser.error('--rollout-worker-mode applies only to portable training')
    if int(os.environ.get('WORLD_SIZE', '1')) > 1 and (
            args.action != 'train' or historical or multinode):
        raise RuntimeError('Only portable training may run under torchrun; native backends '
                           'launch their own ranks and evaluation is one process')
    if multinode:
        if (case.identifier not in (20, 25) or args.backend != 'vllm' or
                args.sender is None or args.statistics_root is not None or
                args.template_init is not None or args.smoke):
            raise ValueError('Two-host original ID20/25 needs a real sender, native vLLM and --native-stage')
        from . import native_multinode
        if args.native_stage != 'train' and (args.resume_step is not None or args.resume_latest):
            parser.error('Only formal native two-host TRAIN may request checkpoint resume')
        if args.native_stage == 'prepare':
            native_multinode.prepare(case.identifier, args.sender, args.receiver, args.output,
                                     data_root=args.data_root, service=args.judge_service_root)
            return
        if not args.master_addr:
            parser.error('Native two-host plan/probe/train needs reachable --master-addr')
        if args.native_stage == 'plan':
            plans = [native_multinode.torchrun_command(args.output, node,
                        args.master_addr, args.master_port, args.native_attempt)
                     for node in (0, 1)]
            print(json.dumps({'experiment': case.identifier, 'requires_two_hosts': True,
                              'gpu_started': False, 'internal_torchrun_commands': plans}, indent=2))
            return
        if args.node_rank is None:
            parser.error('Native two-host run needs --node-rank')
        from platour_exps import require_vllm_python
        require_vllm_python(args.vllm_python)
        if args.resume_latest:
            if args.resume_step is not None or args.native_stage != 'train':
                parser.error('--resume-latest applies only to formal two-host TRAIN')
            protocol = native_multinode.verify_prepared(args.output, case.identifier)
            args.resume_step = native_multinode._checkpoint_step(args.output, case.identifier, protocol)
        native_multinode.run(args.output, case.identifier, args.node_rank,
            args.master_addr, args.master_port, args.native_stage, args.native_attempt,
            bridge_python=Path(sys.executable), generation_python=args.vllm_python,
            scoring_python=args.scoring_python, resume_step=args.resume_step)
        return
    if historical:
        if case.identifier not in (19, 21, 22, 23, 24) or args.backend != 'vllm' or args.template_init is not None:
            raise ValueError('Frozen native Qwen3.5 currently supports 19 and 21–24; other IDs are not bundled')
        if case.identifier in (19, 21) and (args.sender is None or args.statistics_root is not None):
            parser.error('Native ID19/21 needs its original --sender and no external Gaussian statistics')
        if case.identifier == 22 and (args.sender is not None or
                                      (args.statistics_root is None and args.resume_step is None)):
            parser.error('Native ID22 fresh run needs --statistics-root and no external sender')
        if case.identifier in (23, 24) and (args.sender is not None or
                                             args.statistics_root is not None):
            parser.error('Native ID23/24 uses only its independent 4B receiver')
        if args.resume_latest:
            if args.resume_step is not None or args.smoke or case.identifier not in (23, 24):
                parser.error('--resume-latest requires formal ID23/24 and no --resume-step')
            from .native_formal import newest_checkpoint
            args.resume_step = newest_checkpoint(args.output / 'run/train',
                'full' if case.identifier == 23 else 'lora', case.steps)
        if args.resume_step is not None and (args.smoke or case.identifier not in (19, 21, 22, 23, 24)):
            parser.error('Native resume needs formal ID19/21/22/23/24 signed checkpoint')
    if (args.action != 'init-random' and args.smoke and not historical and
            (case.identifier not in (26, 29) or args.backend != 'vllm')):
        raise ValueError('Portable --smoke needs ID26/29 with native vLLM validation/test')
    if args.action == 'train' and args.resume_step is not None and not historical:
        raise ValueError('--resume-step applies only to an opt-in original native trainer')
    if args.action != 'init-random' and args.backend == 'vllm':
        from platour_exps import require_vllm_python
        require_vllm_python(args.vllm_python)
    if historical:
        signed = inputs(case, args.data_root)
        if case.identifier in (19, 21):
            from .native_training import execute, prepare
            if args.resume_step is None:
                prepare(case, args.sender, args.receiver, signed, args.output,
                        smoke=args.smoke)
            else:
                previous = read(args.output / 'protocol.json')
                if (previous.get('native_experiment_id', 21) != case.identifier or
                        Path(previous['teacher_checkpoint']).resolve() != args.sender.resolve() or
                        Path(previous['student_checkpoint']).resolve() != args.receiver.resolve()):
                    raise ValueError('Native resume sender, receiver or experiment differs')
        elif case.identifier == 22:
            from .native_random import execute, prepare
            if args.resume_step is None:
                prepare(case, args.receiver, signed, args.statistics_root, args.output,
                        smoke=args.smoke)
            else:
                previous = read(args.output / 'protocol.json')
                if (previous.get('execution_backend') !=
                        'frozen native train-only Virtual Sender GRPO' or
                        Path(previous['student_checkpoint']).resolve() != args.receiver.resolve() or
                        (args.statistics_root is not None and
                         sha256(args.statistics_root / 'dapo/statistics.pt') != previous['statistics_sha256'])):
                    raise ValueError('Native Virtual Sender resume model or TRAIN-only moments differ')
        elif case.identifier in (23, 24):
            if case.identifier == 23:
                from .native_full import execute, prepare
            else:
                from .native_lora import execute, prepare
            if args.resume_step is None:
                prepare(case, args.receiver, signed, args.output, smoke=args.smoke)
            else:
                previous = read(args.output / 'run/protocol.json')
                if (previous.get('smoke_only') is not False or
                        previous.get('student_checkpoint') != str(args.receiver.resolve()) or
                        previous.get('dataset') != case.task or
                        previous.get('method') != ('full_parameter_grpo' if case.identifier == 23
                                                   else 'lora_grpo')):
                    raise ValueError('Native Full4B/LoRA resume model or method differs')
        if args.resume_step is not None:
            location = (args.output / 'run' if case.identifier in (23, 24) else args.output)
            for split, filename in (('train', 'train_prompts.json'),
                                    ('validation', 'eval_prompts.json')):
                if read(location / filename) != signed[split]:
                    raise ValueError('Native resume signed train/validation rows differ')
        native_options = {'resume_step': args.resume_step}
        execute(args.output, generation_python=args.vllm_python,
                scoring_python=args.scoring_python, **native_options)
    elif args.action == 'train':
        run_train(case, args)
    elif args.action == 'evaluate':
        run_eval(case, args)
    else:
        from .random_init import init
        init(case, args.receiver, args.output, data_root=args.data_root,
             native_statistics_root=args.native_statistics_root)


if __name__ == '__main__':
    main()
