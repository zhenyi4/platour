"""Native LoRA generation, adapter identity and complete rank-aware checkpoints."""
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import torch.distributed as dist

from qwen35_bridge_opd_core import ROOT, save
from qwen35_native_actor_vllm_worker import tensor_digest
from train_qwen35_bridge_opd_vllm import RPC as LocalRPC


def adapter_digest(actor):
    return tensor_digest(actor.bridge_state_tensors())


def base_digest(actor):
    return tensor_digest({name: parameter for name, parameter in actor.student.named_parameters()
                          if not parameter.requires_grad})


def validation_update(state, step, summary, dataset, maximum):
    import math
    accuracy, clipping = summary['accuracy'], summary['clipping_rate']
    if not math.isfinite(accuracy) or not math.isfinite(clipping) or not 0 <= accuracy <= 1 or not 0 <= clipping <= 1:
        raise RuntimeError('Invalid held-out LoRA validation metrics')
    if state is None:
        return dict(best_step=step, best_accuracy=accuracy, best_clipping=clipping, fails=0), None
    guard = dataset == 'dapo' and (clipping > .05 or state['best_clipping'] > .01 and clipping >= 2 * state['best_clipping'])
    improved = accuracy > state['best_accuracy'] and not guard
    result = dict(best_step=step if improved else state['best_step'],
                  best_accuracy=accuracy if improved else state['best_accuracy'],
                  best_clipping=clipping if improved else state['best_clipping'],
                  fails=0 if improved else state['fails'] + 1, clip_guard=guard)
    return result, 'max_steps' if step >= maximum else None


class LoRADecoder(LocalRPC):
    def __init__(self, root, rank, context=18432, protocol=None):
        self.directory = Path(root) / 'workers' / f'rank-{rank:02d}' / 'vllm'
        self.directory.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), HF_HUB_OFFLINE='1',
                   HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                   VLLM_NO_USAGE_STATS='1', VLLM_PORT=str(52000 + rank * 512),
                   PYTHONDONTWRITEBYTECODE='1')
        for key in list(env):
            if key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT') or key.startswith('TORCHELASTIC_'):
                env.pop(key, None)
        python = Path(os.environ['PLATOUR_GENERATION_PYTHON'])
        env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH', '')
        for key, name in [('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton'), ('VLLM_CACHE_ROOT', 'vllm')]:
            env[key] = str(Path('/tmp/platour_lora_math_v3') / f'{Path(root).parent.name}-{Path(root).name}' / f'rank-{rank:02d}' / name)
        command = [str(python), '-u', str(Path(__file__).with_name('qwen35_native_actor_vllm_worker.py')),
                   '--directory', str(self.directory), '--model-path', str(ROOT / 'data/models/Qwen3.5-4B'),
                   '--memory-fraction', '.25', '--max-model-len', str(context), '--max-num-seqs', '16',
                   '--decode-batch-size', '32', '--protocol', str(protocol or Path(root).parent / 'protocol.json')]
        self.log = (self.directory / 'worker.log').open('a')
        self.proc = subprocess.Popen(command, env=env, cwd=ROOT, stdin=subprocess.PIPE,
                                     stdout=self.log, stderr=subprocess.STDOUT, text=True)
        self.wait(self.directory / 'ready.json', timeout=3600)

    def call(self, request):
        target = Path(request['result'])
        if target.exists():
            raise RuntimeError('Refusing stale native LoRA RPC output')
        self.proc.stdin.write(json.dumps(request) + '\n')
        self.proc.stdin.flush()
        return self.wait(target, timeout=7200)


def generate(actor, decoder, rows, sampling, cap, directory, logprobs=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = adapter_digest(actor)
    payload, _ = actor.generate_native_actor(decoder, rows, identity, sampling, cap, directory, logprobs)
    if not payload.get('complete') or len(payload['rows']) != len(rows):
        raise RuntimeError('Incomplete native LoRA generation')
    for actual, expected in zip(payload['rows'], rows, strict=True):
        if any(actual.get(key) != expected.get(key) for key in ('index', 'sample_index', 'prompt_token_ids', 'sampling_seed')):
            raise RuntimeError('Native LoRA prompt/seed identity mismatch')
        if actual['bridge_state_sha256'] != identity or not 0 < actual['tokens'] <= cap:
            raise RuntimeError('Native LoRA adapter identity or token count mismatch')
        if actual['finish_reason'] == 'length' and actual['tokens'] != cap:
            raise RuntimeError('Clipped native response must reach the output limit')
        if logprobs and len(actual['rollout_log_probs']) != actual['tokens']:
            raise RuntimeError('Missing sampled token log probabilities')
    for report in payload.get('actor_weight_audits', []):
        if not report.get('native_lora') or not report.get('all_actor_weights_equal'):
            raise RuntimeError('Native LoRA factors were not loaded exactly')
    return payload['rows']


def checkpoint(actor, optimizer, run, step, state, rank, world, frozen_identity, file_sha):
    directory = Path(run) / 'checkpoints' / f'step-{step:04d}'
    directory.mkdir(parents=True, exist_ok=True)
    identity = adapter_digest(actor)
    identities = [None] * world
    dist.all_gather_object(identities, identity)
    if len(set(identities)) != 1:
        raise RuntimeError('LoRA replicas differ at checkpoint')
    if rank == 0:
        for name, value in [('adapter.pt', actor.bridge_checkpoint_state()), ('optimizer.pt', optimizer.state_dict())]:
            temporary = directory / (name + '.tmp')
            torch.save(value, temporary)
            temporary.replace(directory / name)
    dist.barrier()
    torch.save(dict(step=step, cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state()),
               directory / f'rng.rank-{rank:02d}.pt')
    dist.barrier()
    if rank == 0:
        if not all((directory / f'rng.rank-{index:02d}.pt').exists() for index in range(world)):
            raise RuntimeError('Incomplete LoRA rank RNG checkpoint')
        save(directory / 'checkpoint.json', dict(complete=True, step=step, world_size=world,
            adapter_sha256=file_sha(directory / 'adapter.pt'), optimizer_sha256=file_sha(directory / 'optimizer.pt'),
            adapter_state_sha256=identity, frozen_base_sha256=frozen_identity, early_state=state,
            model='Qwen3.5-4B', lora_rank=128, lora_alpha=128, lora_dropout=0., learning_rate=1e-5))
    dist.barrier()


def load_checkpoint(actor, optimizer, directory, rank, world, frozen_identity, file_sha):
    directory = Path(directory)
    metadata = json.loads((directory / 'checkpoint.json').read_text())
    if (not metadata.get('complete') or metadata['world_size'] != world or
            metadata['frozen_base_sha256'] != frozen_identity or metadata['learning_rate'] != 1e-5 or
            file_sha(directory / 'adapter.pt') != metadata['adapter_sha256'] or
            file_sha(directory / 'optimizer.pt') != metadata['optimizer_sha256'] or
            not all((directory / f'rng.rank-{index:02d}.pt').exists() for index in range(world))):
        raise RuntimeError('LoRA checkpoint/base/optimizer/RNG identity mismatch')
    actor.load_checkpoint(directory / 'adapter.pt')
    optimizer.load_state_dict(torch.load(directory / 'optimizer.pt', weights_only=True, map_location='cpu'))
    if adapter_digest(actor) != metadata['adapter_state_sha256']:
        raise RuntimeError('Restored LoRA factors differ from checkpoint')
    rng = torch.load(directory / f'rng.rank-{rank:02d}.pt', weights_only=True, map_location='cpu')
    return metadata, rng
