"""Shared single-host launch and collective operations for training and PACT.

Each process owns one GPU group. Data parallelism is across groups; sender
layers may span a group while receiver computation stays on its first GPU.
"""
import functools
import json
import os
from pathlib import Path
import subprocess
import sys

GROUP_ENV = 'PLATOUR_GPU_GROUP'


def dist():
    from qwen35 import distributed
    return distributed


def root_call(function):
    return dist().on_rank_zero(function)


def root_only(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        return root_call(lambda: function(*args, **kwargs))
    return wrapped


def api_json(value, path):
    from qwen3.checkpoints import atomic_json
    return root_call(lambda: atomic_json(value, path))


def api_write_json(path, value):
    from qwen35.checkpoints import atomic_json
    return root_call(lambda: atomic_json(path, value))


def protocol_root(root, protocol):
    """Only rank zero creates/validates shared metadata, then releases peers."""
    def prepare():
        path = Path(root) / 'protocol.json'
        if Path(root).exists():
            if not path.is_file() or json.loads(path.read_text()) != protocol:
                raise RuntimeError('Cannot resume with different model/data/budget/GPU topology protocol')
        else:
            Path(root).mkdir(parents=True)
            from qwen3.checkpoints import atomic_json
            atomic_json(protocol, path)
    return root_call(prepare)


def sign_topology(protocol):
    if dist().world() > 1 or GROUP_ENV in os.environ:
        protocol['parallel'] = dict(world_size=dist().world(),
            gpus_per_replica=int(os.environ.get(GROUP_ENV, '1')))
    return protocol


def append_metric(root, record):
    def write():
        with (Path(root) / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
    return root_call(write)


def save_actor(*args, **kwargs):
    from qwen35.checkpoints import save_step, rng_state
    states = dist().gather(rng_state())
    return root_call(lambda: save_step(*args, **kwargs, rank_rng=states))


def load_actor(*args, **kwargs):
    from qwen35.checkpoints import load_step
    result = load_step(*args, **kwargs, rank=dist().rank(), world=dist().world())
    dist().barrier()
    return result


def sync_parameters(parameters):
    """Identical initial trainable weights, including randomly initialized LoRA."""
    if dist().world() > 1:
        import torch.distributed as td
        for parameter in parameters:
            td.broadcast(parameter.data, src=0)


def local_device(default='cuda:0'):
    if dist().world() > 1 or GROUP_ENV in os.environ:
        return dist().device(int(os.environ.get('LOCAL_RANK', '0')))
    return default


def rank_output(path):
    return Path(path) / f'rank-{dist().rank():02d}' if dist().world() > 1 else Path(path)


def sum_values(values, device):
    return dist().all_reduce_sum(values, device)


def collective_evaluation(rows, samples, output, evaluate):
    """Score disjoint global IDs and return the same checked totals to every rank.

    An empty rank participates in aggregation without starting a decoder.
    Exceptions are exchanged before aggregation so peers do not hang at it.
    """
    if dist().world() == 1:
        return evaluate(rows, output)
    local = rows[dist().rank()::dist().world()]
    result = error = None
    try:
        if local:
            result = dict(evaluate(local, rank_output(output)))
            result['question_indices'] = [r['index'] for r in local]
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    parts = dist().gather((error, result))
    if any(error for error, _ in parts):
        raise RuntimeError('Distributed evaluation failed: ' + str([e for e, _ in parts if e]))
    template = next(value for _, value in parts if value is not None)
    keys = ('correct', 'responses', 'at_token_cap', 'clipped', 'passed_questions',
            'gold_missing', 'scoring_timeouts', 'clean80_correct', 'clean80_responses')
    completed = []
    for _, value in parts:
        if value is None:
            value = dict(template, question_indices=[], **{k: 0 for k in keys if k in template})
        completed.append(value)
    from qwen3.eval_parallel import merged_metrics
    merged = merged_metrics(completed, [r['index'] for r in rows], samples)
    merged['complete'] = True
    merged['data_parallel_replicas'] = dist().world()
    return merged


def parallel_evaluate(function):
    """Collective API evaluate without changing the caller's datasets or seeds."""
    @functools.wraps(function)
    def wrapped(self, data, **options):
        if dist().world() == 1:
            return function(self, data, **options)
        from platour_api import load_dataset
        rows = load_dataset(data)
        samples = options.get('samples', 1)
        output = options.get('output')
        if output is None:
            raise ValueError('Distributed evaluation needs an output directory')
        return collective_evaluation(rows, samples, output,
            lambda local, path: function(self, local, **dict(options, output=path)))
    return wrapped


def add_arguments(parser):
    parser.add_argument('--n-gpu', '--n_gpu', type=int,
        help='GPUs on this host: 1 for single-GPU 8B->1.7B, 8 for replica parallelism; '
             'omitted keeps the existing placement/torchrun behavior')
    parser.add_argument('--gpus-per-replica', type=int,
        help='GPUs per replica: default 235B=2, 397B=4, otherwise 1; must divide --n-gpu')


def maybe_launch(argv):
    """Route only train/PACT --n-gpu; evaluation has its own standalone launcher."""
    if not argv or argv[0] not in ('train', 'pact-check') or any(x in argv for x in ('-h', '--help', '--prepare-only')):
        return False
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    add_arguments(p)
    p.add_argument('--experiment', type=int)
    p.add_argument('--sender', default='')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--controls', nargs='+')
    p.add_argument('--train-backend', default='portable')
    args, _ = p.parse_known_args(argv[1:])
    if args.n_gpu is None:
        if args.gpus_per_replica is not None:
            raise ValueError('--gpus-per-replica requires --n-gpu')
        return False
    if GROUP_ENV in os.environ or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('--n-gpu owns its launcher; do not combine with torchrun')
    if args.device != 'cuda:0':
        raise ValueError('With --n-gpu select physical GPUs using CUDA_VISIBLE_DEVICES; leave --device=cuda:0')
    from qwen3.eval_parallel import default_gpus_per_replica, gpu_groups, child_arguments
    import torch
    count = torch.cuda.device_count()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(count)))).split(',')[:count]
    per = default_gpus_per_replica(args.experiment) if args.experiment is not None else (
        4 if '397B' in args.sender else 2 if '235B' in args.sender else 1)
    if argv[0] == 'pact-check' and args.controls and set(args.controls) <= {'SC', 'TR'}:
        per = 1
    if args.gpus_per_replica is not None:
        per = args.gpus_per_replica
    groups = gpu_groups(args.n_gpu, per, visible)
    native = argv[0] == 'train' and args.train_backend != 'portable'
    if native and (args.n_gpu != 8 or args.gpus_per_replica is not None):
        raise ValueError('Native training keeps its fixed eight-GPU-per-host topology; use portable for --n-gpu 1')
    child = child_arguments(argv, {})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env['CUDA_VISIBLE_DEVICES'] = ','.join(visible[:args.n_gpu])
    if native and args.train_backend != 'historical-sft':
        command = [sys.executable, '-u', '-m', 'platour_exps', *child]
    else:
        replicas = 4 if native else len(groups)
        env['PLATOUR_LAUNCH_GROUPS'] = json.dumps(groups)
        env['PLATOUR_LAUNCH_NATIVE_SFT'] = '1' if native else '0'
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   f'--nproc-per-node={replicas}', '-m', 'qwen3.parallel', *child]
    print(json.dumps(dict(n_gpu=args.n_gpu, gpu_groups=groups, action=argv[0],
                          native_topology=native)), flush=True)
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = process.wait()
        if code:
            raise RuntimeError(f'Parallel {argv[0]} exited with status {code}')
    finally:
        if process.poll() is None:
            import signal
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return True


def configure_group(groups, local_rank):
    """Mask physical GPUs before CUDA initialization in a torchrun child."""
    group = groups[local_rank]
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(group), LOCAL_RANK='0',
                      **{GROUP_ENV: str(len(group))})


def worker():
    import signal
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    if os.environ.pop('PLATOUR_LAUNCH_NATIVE_SFT') != '1':
        groups = json.loads(os.environ.pop('PLATOUR_LAUNCH_GROUPS'))
        configure_group(groups, int(os.environ['LOCAL_RANK']))
    from platour_exps import main
    main()


if __name__ == '__main__':
    worker()
