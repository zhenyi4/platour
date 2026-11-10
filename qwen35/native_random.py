"""Train-only-statistics native Virtual Sender GRPO from the frozen original run."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from .checkpoints import atomic_json
from .native_training import digest, original_math_checker
from .random_init import template


ASSETS = Path(__file__).with_name('native_reference') / '22'
HASHES = Path(__file__).with_name('native_random_source_sha256.json')
PROTOCOLS = Path(__file__).with_name('native_protocols.json')

STATISTICS_VERIFIER = '''"""Verify train-only native Qwen3.5-4B scalar moments without site paths."""
from pathlib import Path
from qwen35.native_random import verify_statistics as verify

def verify_statistics(dataset):
    return verify(dataset, Path(__file__).resolve().parent)
'''


def verify_bundle():
    expected = json.loads(HASHES.read_text())
    actual = {p.relative_to(ASSETS).as_posix() for p in ASSETS.rglob('*.py')}
    if (len(expected) != 49 or actual != set(expected) or
            any('..' in Path(name).parts or digest(ASSETS / name) != value
                for name, value in expected.items())):
        raise ValueError('Frozen native Random4B training source differs')
    return expected


def verify_statistics(dataset, root):
    root = Path(root)
    if dataset != 'dapo':
        raise ValueError('Only signed native Random4B mathematics is bundled')
    meta = json.loads((root / 'manifest.json').read_text())
    stats = root / dataset / 'statistics.pt'
    template_file = root / dataset / 'template.json'
    if (digest(stats) != meta['statistics_sha256'] or
            digest(template_file) != meta['template_sha256']):
        raise ValueError('Random4B training statistics identity differs')
    state = torch.load(stats, weights_only=True, map_location='cpu')
    record = json.loads(template_file.read_text())
    mean, std = state['mean'], state['std']
    if (state.get('statistics_source_model') != 'Qwen3.5-4B' or
            state['sample_count'] != 500 or state['seed'] != 42 or
            state['layer_map'] != list(range(32)) or
            state['variance_correction'] != 0 or
            tuple(mean.shape) != (63, 32, 1) or mean.shape != std.shape or
            not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()) or
            not bool((std >= 0).all()) or
            state['template_token_ids'] != record['token_template']['template_token_ids'] or
            state['sample_indices'] != record['selected_sample_indices'] or
            state['broadcast_shape'] != [63, 32, 2560]):
        raise ValueError('Virtual Sender moments/template/source differ from train-only contract')
    return state


def prepare(case, receiver, signed, statistics_root, output, *, smoke=False):
    if case.identifier != 22 or case.task != 'dapo' or case.method != 'random':
        raise ValueError('Only ID22 native train-only Virtual Sender is bundled')
    receiver = Path(receiver).resolve()
    if receiver.name != 'Qwen3.5-4B' or not (receiver / 'config.json').is_file():
        raise ValueError('Native Random4B needs the original local Qwen3.5-4B receiver')
    if len(signed['train']) != 16891 or len(signed['validation']) != 500:
        raise ValueError('Random4B signed train/validation question count differs')
    sources = verify_bundle()
    source_root = Path(statistics_root).resolve()
    state_path = source_root / 'dapo/statistics.pt'
    record_path = source_root / 'dapo/template.json'
    record = json.loads(record_path.read_text())
    stats = torch.load(state_path, weights_only=True, map_location='cpu')
    prefix, suffix, fixed = template(signed['train'])
    selected = sorted(sorted(signed['train'], key=lambda row: hashlib.sha256(
        f'42:{row["index"]}'.encode()).hexdigest())[:500], key=lambda row: row['index'])
    if ((prefix, suffix, len(fixed)) != (42, 22, 63) or
            record['token_template']['template_token_ids'] != fixed or
            record['token_template']['prefix'] != prefix or
            record['token_template']['suffix'] != suffix or
            state_path.stat().st_size < 1 or
            stats['sample_indices'] != record['selected_sample_indices'] or
            stats['sample_indices'] != [row['index'] for row in selected] or
            stats['template_token_ids'] != fixed or
            stats['source_model_config_sha256'] != digest(receiver / 'config.json')):
        raise ValueError('External Gaussian moments are not from the signed 500 train-only questions/model')
    if any(row['prompt_token_ids'][:prefix] + row['prompt_token_ids'][-suffix:-1] != fixed
           for row in signed['validation']):
        raise ValueError('Validation prompt template differs from TRAIN-only statistics')
    model_shards = stats.get('source_model_shards_sha256')
    if (not isinstance(model_shards, dict) or len(model_shards) != 2 or
            stats.get('source_model_index_sha256') != digest(receiver / 'model.safetensors.index.json') or
            any(Path(name).name != name or not (receiver / name).is_file() or
                digest(receiver / name) != expected
                for name, expected in model_shards.items())):
        raise ValueError('Train-only V moments were not measured on these exact 4B model weights')
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError('Native Random4B requires a fresh output root')
    visible = list(map(int, os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').split(',')))
    protocol = json.loads(PROTOCOLS.read_text())['22']
    if (len(visible) != 8 or len(set(visible)) != 8 or
            protocol['world_size'] != 8 or protocol['total_steps'] != case.steps or
            protocol['group_size'] != 8 or protocol['batch_size'] != 64 or
            tuple(protocol['eval_steps']) != case.validation_steps):
        raise ValueError('Native Virtual Sender GPU topology or original schedule differs')
    output.mkdir(parents=True)
    frozen = output / 'reference_source'
    shutil.copytree(ASSETS, frozen)
    (frozen / 'data').mkdir()
    (frozen / 'data/models').symlink_to(receiver.parent, target_is_directory=True)
    atomic_json(output / 'source_manifest.json', {'files': sources,
        'site_launcher_and_private_test_excluded': True})
    atomic_json(output / 'train_prompts.json', signed['train'])
    atomic_json(output / 'eval_prompts.json', signed['validation'])
    target = output / 'statistics/dapo'
    target.mkdir(parents=True)
    shutil.copyfile(state_path, target / 'statistics.pt')
    shutil.copyfile(record_path, target / 'template.json')
    (target.parent / 'template_statistics.py').write_text(STATISTICS_VERIFIER)
    atomic_json(target.parent / 'manifest.json', {
        'statistics_sha256': digest(target / 'statistics.pt'),
        'template_sha256': digest(target / 'template.json'),
        'statistics_verifier_sha256': digest(target.parent / 'template_statistics.py')})
    verify_statistics('dapo', target.parent)
    protocol = dict(protocol)
    protocol.update(student_checkpoint=str(receiver),
        source_execution_root=str(frozen),
        statistics_root=str(target.parent),
        statistics_sha256=digest(target / 'statistics.pt'),
        source_manifest_sha256=digest(output / 'source_manifest.json'),
        train_prompts_sha256=digest(output / 'train_prompts.json'),
        eval_prompts_sha256=digest(output / 'eval_prompts.json'),
        gpu_ids=visible, decoder_gpu_ids=visible,
        wandb_mode='disabled', wandb_enabled=False,
        wandb_project='disabled', wandb_run_name='native_random4b_dapo_reference',
        execution_backend='frozen native train-only Virtual Sender GRPO',
        smoke_only=smoke)
    from .native_selection import contract
    protocol.update(contract())
    atomic_json(output / 'protocol.json', protocol)
    atomic_json(output / 'source_audit.json', {
        'experiment': 22, 'frozen_training_source_files': len(sources),
        'external_train_only_statistics_file_sha256': protocol['statistics_sha256'],
        'selected_train_only_questions': 500,
        'excluded_validation_and_test_for_statistics': True,
        'portable_hf_rollouts_used': False})
    return output


def verify_prepared(root):
    root = Path(root).absolute()
    q = json.loads((root / 'protocol.json').read_text())
    expected = json.loads((root / 'source_manifest.json').read_text())['files']
    if (expected != verify_bundle() or
            q['source_manifest_sha256'] != digest(root / 'source_manifest.json') or
            q['source_execution_root'] != str(root / 'reference_source') or
            q['statistics_root'] != str(root / 'statistics') or
            q['wandb_mode'] != 'disabled'):
        raise ValueError('Prepared Random4B frozen source/protocol differs')
    for name, value in expected.items():
        if digest(root / 'reference_source' / name) != value:
            raise ValueError('Native Random4B trainer source changed: ' + name)
    for name in ('train', 'eval'):
        if digest(root / f'{name}_prompts.json') != q[f'{name}_prompts_sha256']:
            raise ValueError('Native Random4B signed split changed')
    if (digest(root / 'statistics/template_statistics.py') !=
            json.loads((root / 'statistics/manifest.json').read_text())['statistics_verifier_sha256'] or
            digest(root / 'statistics/dapo/statistics.pt') != q['statistics_sha256']):
        raise ValueError('Native train-only statistics verifier or tensor changed')
    verify_statistics('dapo', root / 'statistics')
    from .native_selection import verify
    verify(q)
    return q


def worker(root, *, resume_step=None):
    root = Path(root).absolute()
    q = verify_prepared(root)
    frozen = root / 'reference_source'
    sys.path[:0] = [str(frozen / 'recipe/llm_communicate'), str(frozen / 'scripts')]
    import random_config

    def verify_frozen_source():
        return verify_prepared(root)['source_manifest_sha256']

    random_config.verify_sources = verify_frozen_source
    import train_qwen35_bridge_opd_vllm as trainer
    sys.argv = [str(frozen / 'recipe/llm_communicate/train_qwen35_bridge_opd_vllm.py'),
                '--root', str(root), '--objective', 'grpo']
    if q['smoke_only']:
        sys.argv += ['--smoke', '--stress']
    if resume_step is not None:
        sys.argv += ['--resume-step', str(resume_step)]
    from .native_selection import install
    install(trainer)
    trainer.main()


def execute(root, *, generation_python, scoring_python, resume_step=None):
    root = Path(root).absolute()
    q = verify_prepared(root)
    if resume_step is not None:
        from .native_training import verify_resume
        prior = json.loads((root / 'native_process.json').read_text())
        if type(prior.get('pid')) is not int or prior['pid'] <= 1:
            raise ValueError('Prior native Virtual Sender process identity differs')
        try:
            os.killpg(prior['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Prior native Virtual Sender process is still running')
        verify_resume(root, q, resume_step)
    if not all(Path(p).is_file() for p in (generation_python, scoring_python)):
        raise FileNotFoundError('Native Random4B generation and scorer Pythons required')
    frozen = root / 'reference_source'
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, q['gpu_ids'])),
        WANDB_MODE='disabled', WANDB_DISABLED='true', PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', PLATOUR_BRIDGE_PYTHON=sys.executable,
        PLATOUR_GENERATION_PYTHON=str(generation_python),
        PLATOUR_SCORING_PYTHON=str(scoring_python))
    environment['PYTHONPATH'] = os.pathsep.join((str(frozen / 'recipe/llm_communicate'),
        str(frozen / 'scripts'), original_math_checker(), str(Path(__file__).resolve().parents[1]),
        os.environ.get('PYTHONPATH', '')))
    for key in ('WANDB_API_KEY', 'WANDB_RUN_ID', 'WANDB_ENTITY'):
        environment.pop(key, None)
    port = int(environment.get('PLATOUR_NATIVE_MASTER_PORT', '39122'))
    if not 1024 <= port <= 65535:
        raise ValueError('Native Random4B rendezvous port must be in 1024..65535')
    environment['MASTER_ADDR'] = '127.0.0.1'
    environment['MASTER_PORT'] = str(port)
    command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--nnodes=1',
               '--nproc-per-node=8', '--master-addr=127.0.0.1',
               f'--master-port={port}', '--max-restarts=0', '-m',
               'qwen35.native_random', 'worker', '--root', str(root)]
    if resume_step is not None:
        command += ['--resume-step', str(resume_step)]
        archive = root / 'recovery'
        archive.mkdir(exist_ok=True)
        shutil.copyfile(root / 'phase.json',
                        archive / f'phase-before-resume-{resume_step:04d}-{time.time_ns()}.json')
    atomic_json(root / 'phase.json', {'phase': 'smoke' if q['smoke_only'] else 'train',
                                     'world_size': 8, 'time': time.time(),
                                     'resumed_from_step': resume_step})
    try:
        from platour_exps import run_native_training
        folder = root / ('smoke' if q['smoke_only'] else 'train')
        run_native_training(command, cwd=frozen, environment=environment, output=root,
                            failure_directory=folder, smoke=q['smoke_only'],
                            resume_step=resume_step)
        complete = json.loads((folder / 'complete.json').read_text())
        if (not complete['complete'] or complete['world_size'] != 8 or
                complete['nonzero_gradient_updates'] < 1 or
                not complete['replica_exact_weights']):
            raise RuntimeError('Native Virtual Sender training structural checks failed')
        if q['smoke_only']:
            score = json.loads((folder / 'evaluation/step-0002/summary.json').read_text())
            if (not complete['singleton_tail_verified'] or not score['complete'] or
                    score['questions'] != 500 or score['responses'] != 2000 or
                    score.get('scored_responses') != 2000 or
                    score.get('gold_missing') or score.get('scoring_timeouts')):
                raise RuntimeError('Native Random4B 500×4 validation incomplete')
            for rank in range(8):
                pressure = json.loads((folder / f'worst_case_profile.rank-{rank:02d}.json').read_text())
                if not pressure['passed'] or not pressure['vllm_resident']:
                    raise RuntimeError('Native Random4B 16K GPU pressure failed')
        elif complete['steps'] != q['total_steps']:
            raise RuntimeError('Native Random4B full budget not completed')
        atomic_json(root / 'phase.json', {'phase': 'complete', 'time': time.time()})
    except BaseException as error:
        atomic_json(root / 'phase.json', {'phase': 'failed', 'error': repr(error),
                                         'time': time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('worker', 'execute'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--vllm-python', type=Path)
    parser.add_argument('--scoring-python', type=Path)
    parser.add_argument('--resume-step', type=int)
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.root, resume_step=args.resume_step)
    elif args.vllm_python is None or args.scoring_python is None:
        parser.error('execute needs native vLLM and scorer Python')
    else:
        execute(args.root, generation_python=args.vllm_python,
                scoring_python=args.scoring_python, resume_step=args.resume_step)


if __name__ == '__main__':
    main()
