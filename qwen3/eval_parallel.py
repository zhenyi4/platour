"""Independent evaluation replicas, each owning a disjoint CUDA device group."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SHARD_ENV = 'PLATOUR_EVAL_SHARD'


def default_gpus_per_replica(experiment):
    """Released-model topology; this is not automatic GPU-memory sizing."""
    if experiment in (*range(1, 7), 13, 14, 15):
        return 2  # Qwen3-235B-A22B-FP8 sender.
    if experiment in (20, 25):
        return 4  # Qwen3.5-397B-A17B-FP8 sender.
    return 1


def add_arguments(parser):
    parser.add_argument('--n-gpu', '--n_gpu', type=int, default=None,
                        help='Total GPUs for standalone replica evaluation, using the first N visible GPUs. '
                             'Default: one process with automatic model placement, no data parallelism. '
                             'Use 1 for a small-model single-GPU run or 8 for an eight-GPU run; '
                             'do not combine with torchrun')
    parser.add_argument('--gpus-per-replica', type=int, default=None,
                        help='Requires --n-gpu. Default by experiment: Qwen3-235B-FP8=2, '
                             'Qwen3.5-397B-FP8=4, all others=1 (including 35B and receiver-only). '
                             'Total GPUs must be divisible by this value; each replica must fit')


def shard_rows(rows):
    spec = json.loads(os.environ.get(SHARD_ENV, 'null'))
    if spec is None:
        return rows
    rank, world = spec['rank'], spec['world']
    if not (type(rank) is int and type(world) is int and 0 <= rank < world):
        raise ValueError('Invalid evaluation shard')
    part = rows[rank::world]
    if not part:
        raise ValueError('More evaluation replicas than questions')
    return part


def configure_worker():
    if SHARD_ENV in os.environ:
        def terminate(signum, frame):
            raise SystemExit(128 + signum)  # Unwind decoder contexts and close owned workers.
        signal.signal(signal.SIGTERM, terminate)


def gpu_groups(n_gpu, per_replica, visible):
    if n_gpu < 1 or per_replica < 1 or n_gpu % per_replica or n_gpu > len(visible):
        raise ValueError('Require 1 <= n-gpu <= visible GPUs and n-gpu divisible by gpus-per-replica')
    return [visible[i:i + per_replica] for i in range(0, n_gpu, per_replica)]


def merged_metrics(parts, indices, samples):
    """Aggregate counts, never unweighted averages of uneven shards."""
    for rank, part in enumerate(parts):
        expected = indices[rank::len(parts)]
        if part['question_indices'] != expected or part['responses'] != len(expected) * samples:
            raise ValueError('Evaluation shard coverage/count differs')
        if not 0 <= part['correct'] <= part['responses']:
            raise ValueError('Invalid correct count')
    result = dict(parts[0])
    for key in ('correct', 'responses', 'at_token_cap', 'clipped', 'passed_questions',
                'gold_missing', 'scoring_timeouts', 'clean80_correct', 'clean80_responses'):
        if key in result:
            result[key] = sum(p[key] for p in parts)
    result.update(question_indices=indices, questions=len(indices),
                  accuracy=result['correct'] / result['responses'])
    if 'passed_questions' in result:
        result['pass_at_k'] = result['passed_questions'] / len(indices)
        result['clipping_rate'] = result['clipped'] / result['responses']
    if 'clean80_correct' in result:
        result['clean80_accuracy'] = (result['clean80_correct'] / result['clean80_responses']
                                      if result['clean80_responses'] else None)
    if 'runtime' in result:
        result['runtime'] = [p['runtime'] for p in parts]
    if 'parity' in result:
        result['parity'] = [p['parity'] for p in parts]
    return result


def child_arguments(argv, replacements):
    result = []
    skip = False
    removed = {'--n-gpu', '--n_gpu', '--gpus-per-replica', *replacements}
    for value in argv:
        if skip:
            skip = False
            continue
        key = value.split('=', 1)[0]
        if key in removed:
            skip = '=' not in value
        else:
            result.append(value)
    for key, value in replacements.items():
        if value is not None:
            result.extend([key, str(value)])
    return result


def evaluation_suites(task, selected=None):
    available = ('aime', 'hmmt', 'imo') if task == 'dapo' else ('lcb', 'humaneval', 'mbpp')
    if selected is None:
        return available
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(available):
        raise ValueError(f'Choose distinct evaluation suites from {available}')
    return tuple(name for name in available if name in selected)


def three_suite_macro(results):
    if set(results) not in ({'aime', 'hmmt', 'imo'}, {'lcb', 'humaneval', 'mbpp'}):
        return None
    values = [value['clean80_accuracy'] if name == 'aime' else value['accuracy']
              for name, value in results.items()]
    return None if any(v is None for v in values) else sum(values) / 3


def prepare(args):
    from platour_exps import resolve_model_input, resolve_weight_input
    if getattr(args, 'run', None) is not None and args.weights is not None:
        raise ValueError('Choose weights or run, not both')
    if args.experiment <= 18:
        from .cli import audit_inputs, read_rows, select, ROOT
        case = select(args.experiment)
        args.test_data = args.test_data or ROOT / 'data/qwen3' / case['tag'] / 'test.jsonl'
        audit_inputs(case, {'test': args.test_data})
        expected = {'test': ([r['index'] for r in read_rows(args.test_data)], case['eval_samples'])}
    else:
        from qwen35.cli import select, signed_rows, verify_code_evaluation_environment
        from platour_data import ensure_qwen35_data
        case = select(args.experiment)
        if args.smoke:
            raise ValueError('Use the existing single-process smoke without --n-gpu')
        ensure_qwen35_data(case.identifier, args.data_root, 'evaluate')
        verify_code_evaluation_environment(case, args)
        expected = {}
        for name in evaluation_suites(case.task, getattr(args, 'eval_suites', None)):
            suffix = name + ('_template_adapted' if case.method == 'random' and name in ('humaneval', 'mbpp') else '')
            rows = signed_rows(args.data_root / case.task / (suffix + '.jsonl'))
            samples = getattr(case, 'test_samples', None) or (8 if case.task == 'deepcoder' else 4 if name == 'imo' else 16)
            expected[name] = ([r['index'] for r in rows], samples)
    for name in ('sender', 'receiver'):
        if getattr(args, name, None) is not None:
            setattr(args, name, resolve_model_input(getattr(args, name)))
    if getattr(args, 'run', None) is None:
        args.weights = resolve_weight_input(args.experiment, args.weights)
    return expected


def launch(args):
    if SHARD_ENV in os.environ or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('--n-gpu launches its own replicas; do not combine with torchrun')
    if getattr(args, 'device', 'cuda:0') != 'cuda:0':
        raise ValueError('With --n-gpu select devices using CUDA_VISIBLE_DEVICES, leaving --device=cuda:0')
    import torch
    count = torch.cuda.device_count()
    raw = os.environ.get('CUDA_VISIBLE_DEVICES')
    visible = raw.split(',')[:count] if raw is not None else [str(i) for i in range(count)]
    per = (args.gpus_per_replica if args.gpus_per_replica is not None
           else default_gpus_per_replica(args.experiment))
    groups = gpu_groups(args.n_gpu, per, visible)
    output = args.output.absolute()
    if output.exists():
        raise FileExistsError('Use a fresh evaluation directory')
    expected = prepare(args)  # Download and verify common artifacts before starting replicas.
    if any(len(ids) < len(groups) for ids, _ in expected.values()):
        raise ValueError('More replicas than evaluation questions')
    output.mkdir(parents=True)
    plan = dict(n_gpu=args.n_gpu, gpus_per_replica=per, gpu_groups=groups,
                replicas=len(groups), experiment=args.experiment, active_sequence_limits_changed=False)
    from .checkpoints import atomic_json
    atomic_json(plan, output / 'parallel.json')
    print(json.dumps(plan), flush=True)
    started = time.monotonic()
    jobs = []
    try:
        for rank, group in enumerate(groups):
            folder = output / f'replica-{rank:02d}'
            replacements = {'--output': folder, '--sender': getattr(args, 'sender', None),
                            '--receiver': args.receiver, '--weights': args.weights}
            argv = child_arguments(sys.argv[1:], replacements)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(group))
            env[SHARD_ENV] = json.dumps(dict(rank=rank, world=len(groups)))
            for key in list(env):
                if key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT') or key.startswith('TORCHELASTIC_'):
                    env.pop(key)
            env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH', '')
            with (output / f'replica-{rank:02d}.log').open('w') as log:
                process = subprocess.Popen([sys.executable, '-u', '-m', 'platour_exps', *argv],
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            jobs.append(process)
        while any(p.poll() is None for p in jobs):
            if any(p.poll() not in (None, 0) for p in jobs):
                raise RuntimeError(f'Evaluation replica failed; inspect {output}/replica-*.log')
            time.sleep(.5)
        if any(p.returncode != 0 for p in jobs):
            raise RuntimeError('Evaluation replica failed')
        parts = [json.loads((output / f'replica-{rank:02d}' / 'summary.json').read_text()) for rank in range(len(groups))]
        for key in ('experiment', 'checkpoint_source', 'selection'):
            if any(p.get(key) != parts[0].get(key) for p in parts):
                raise ValueError('Replica checkpoint provenance differs')
        if args.experiment <= 18:
            result = merged_metrics(parts, *expected['test'])
        else:
            result = dict(parts[0])
            result['results'] = {name: merged_metrics([p['results'][name] for p in parts], ids, samples)
                                 for name, (ids, samples) in expected.items()}
            result['three_suite_macro'] = three_suite_macro(result['results'])
            for name, value in result['results'].items():
                atomic_json(value, output / f'{name}.json')
        result.update(complete=True, parallel=plan, evaluation_wall_seconds=time.monotonic() - started)
        atomic_json(result, output / 'summary.json')
        print(json.dumps(result), flush=True)
    except BaseException as error:
        atomic_json(dict(complete=False, error=str(error)), output / 'failed.json')
        raise
    finally:
        for process in jobs:
            if process.poll() is None:
                process.terminate()
        for process in jobs:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def sender_device_map(family, depth, devices):
    """Place a replica's sender on its GPU group, leaving room on decoder GPU 0."""
    if not 1 <= devices <= depth:
        raise ValueError('Invalid sender GPU partition')
    if family == 'qwen35' and depth == 60 and devices == 4:
        counts = [12, 16, 16, 16]  # Same 397B partition as main.
    else:
        counts = [depth // devices + (i >= devices - depth % devices) for i in range(devices)]
    prefix = 'model.language_model' if family == 'qwen35' else 'model'
    placement = {prefix + '.embed_tokens': 0, prefix + '.rotary_emb': 0,
                 prefix + '.norm': devices - 1, 'lm_head': devices - 1}
    if family == 'qwen35':
        placement['model.visual'] = devices - 1
    offset = 0
    for device, count in enumerate(counts):
        placement.update({f'{prefix}.layers.{i}': device for i in range(offset, offset + count)})
        offset += count
    return placement
