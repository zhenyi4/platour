"""Offline, SHA-checked native full-parameter Qwen3.5-4B GRPO for ID23."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .checkpoints import atomic_json
from .native_training import digest, original_math_checker


ASSETS = Path(__file__).with_name('native_reference') / '23'
HASHES = Path(__file__).with_name('native_full_source_sha256.json')
PROTOCOLS = Path(__file__).with_name('native_protocols.json')


def verify_bundle():
    info = json.loads(HASHES.read_text())
    sources = info['files']
    actual = {p.relative_to(ASSETS).as_posix() for p in ASSETS.rglob('*.py')}
    trainer = ASSETS / 'recipe/llm_communicate/full4b_train.py'
    if (len(sources) != 35 or actual != set(sources) or
            digest(trainer) != info['offline_only_trainer_sha256'] or
            'mode=\'online\'' in trainer.read_text() or
            'queued_wandb.json' in trainer.read_text() or
            any('..' in Path(name).parts or digest(ASSETS / name) != value
                for name, value in sources.items())):
        raise ValueError('Offline full-parameter GRPO source or network gate differs')
    return sources


def prepare(case, receiver, signed, output, *, smoke=False):
    if case.identifier != 23 or case.method != 'full' or case.task != 'dapo':
        raise ValueError('Only Qwen3.5 Full-GRPO mathematics ID23 is supported')
    receiver = Path(receiver).resolve()
    if (receiver.name != 'Qwen3.5-4B' or not (receiver / 'config.json').is_file() or
            len(signed['train']) != 16891 or len(signed['validation']) != 500):
        raise ValueError('Native Full4B original model or signed data count differs')
    sources = verify_bundle()
    proto = dict(json.loads(PROTOCOLS.read_text())['23'])
    if (proto['world_size'] != 8 or proto['total_steps'] != case.steps or
            proto['learning_rate'] != case.learning_rate or
            proto['eval_samples'] != 4 or
            tuple(proto['eval_steps']) != case.validation_steps):
        raise ValueError('Original Full4B DAPO training schedule differs')
    visible = list(map(int, os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').split(',')))
    if len(visible) != 8 or len(set(visible)) != 8:
        raise ValueError('Native Full4B requires eight independent visible GPUs')
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError('Native Full4B needs a fresh output container')
    output.mkdir(parents=True)
    source = output / 'reference_source'
    shutil.copytree(ASSETS, source)
    if not smoke:
        from .native_formal import SOURCES, patch_sources
        from .checkpoints import sha256
        originals = {name: (source / name).read_text() for name in SOURCES['full']}
        modified = patch_sources('full', originals)
        for name, text in modified.items():
            (source / name).write_text(text)
        atomic_json(output / 'formal_source_patch.json', {'experiment': 23,
            'original_sha256': {name: sources[name] for name in modified},
            'active_sha256': {name: sha256(source / name) for name in modified},
            'only_checkpoint_io_changed': True})
        sources = {**sources, **{name: sha256(source / name) for name in modified}}
    (source / 'data').mkdir()
    (source / 'data/models').symlink_to(receiver.parent, target_is_directory=True)
    atomic_json(output / 'source_manifest.json', {
        'reference_source/' + name: value for name, value in sources.items()})
    run = output / 'run'
    run.mkdir()
    atomic_json(run / 'train_prompts.json', signed['train'])
    atomic_json(run / 'eval_prompts.json', signed['validation'])
    proto.update(student_checkpoint=str(receiver),
                 source_manifest_sha256=digest(output / 'source_manifest.json'),
                 train_prompts_sha256=digest(run / 'train_prompts.json'),
                 eval_prompts_sha256=digest(run / 'eval_prompts.json'),
                 gpu_ids=visible, wandb_mode='disabled', wandb_enabled=False,
                 native_training_scope='full_parameter_bf16_sharded_fp32_adam',
                 per_step_checkpoints=not smoke, rolling_predecessor=1,
                 smoke_only=smoke)
    from .native_selection import contract
    proto.update(contract())
    atomic_json(run / 'protocol.json', proto)
    provenance = json.loads(HASHES.read_text())
    atomic_json(output / 'source_audit.json', {
        'experiment': 23, 'frozen_source_files': len(sources),
        'unmodified_original_trainer_sha256': provenance['unmodified_original_trainer_sha256'],
        'offline_only_trainer_sha256': provenance['offline_only_trainer_sha256'],
        'offline_change': provenance['network_change'], 'full_training_started': False,
        'model_weights_bundled': False})
    return output


def verify_prepared(output):
    output = Path(output).absolute()
    sources = verify_bundle()
    run = output / 'run'
    protocol = json.loads((run / 'protocol.json').read_text())
    marker = output / 'formal_source_patch.json'
    if protocol.get('smoke_only') is False:
        from .native_formal import SOURCES, patch_sources
        from .checkpoints import sha256
        modified = patch_sources('full', {name: (ASSETS / name).read_text()
                                          for name in SOURCES['full']})
        expected = {name: sha256(ASSETS / name) for name in modified}
        active = {name: hashlib.sha256(text.encode()).hexdigest()
                  for name, text in modified.items()}
        receipt = json.loads(marker.read_text()) if marker.is_file() else {}
        if (receipt.get('experiment') != 23 or receipt.get('original_sha256') != expected or
                receipt.get('active_sha256') != active or
                receipt.get('only_checkpoint_io_changed') is not True):
            raise ValueError('Native Full4B formal checkpoint source seal differs')
        sources = {**sources, **active}
    elif protocol.get('smoke_only') is not True or marker.exists():
        raise ValueError('Native Full4B smoke source identity differs')
    manifest = json.loads((output / 'source_manifest.json').read_text())
    if manifest != {'reference_source/' + name: value for name, value in sources.items()}:
        raise ValueError('Frozen Full4B scientific source list changed')
    for name, value in manifest.items():
        if digest(output / name) != value:
            raise ValueError('Prepared Full4B source changed: ' + name)
    if (protocol['source_manifest_sha256'] != digest(output / 'source_manifest.json') or
            protocol['world_size'] != 8 or protocol['total_steps'] != 264 or
            protocol.get('per_step_checkpoints') is not (not protocol['smoke_only']) or
            protocol.get('rolling_predecessor') != 1 or
            protocol['wandb_mode'] != 'disabled'):
        raise ValueError('Native Full4B signed offline/checkpoint protocol differs')
    for split in ('train', 'eval'):
        if digest(run / f'{split}_prompts.json') != protocol[f'{split}_prompts_sha256']:
            raise ValueError('Frozen Full4B signed input differs: ' + split)
    from .native_selection import verify
    verify(protocol)
    return protocol


def worker(output, *, resume_step=None):
    output = Path(output).absolute()
    protocol = verify_prepared(output)
    source = output / 'reference_source'
    sys.path[:0] = [str(source / 'recipe/llm_communicate'), str(source / 'scripts')]
    import full4b_train as trainer
    sys.argv = [str(source / 'recipe/llm_communicate/full4b_train.py'),
                '--root', str(output / 'run')]
    if protocol['smoke_only']:
        sys.argv.append('--smoke')
    elif resume_step is not None:
        sys.argv.extend(('--resume-step', str(resume_step)))
    from .native_selection import install
    install(trainer)
    trainer.main()


def execute(output, *, generation_python, scoring_python, resume_step=None):
    output = Path(output).absolute()
    protocol = verify_prepared(output)
    if not all(Path(p).is_file() for p in (generation_python, scoring_python)):
        raise FileNotFoundError('Native Full4B vLLM and MathVerify Python must exist')
    if resume_step is not None:
        from .native_formal import archive_resume_tail, newest_checkpoint
        if protocol['smoke_only'] or type(resume_step) is not int or resume_step < 1:
            raise ValueError('Only a signed unfinished formal Full4B run may resume')
        if newest_checkpoint(output / 'run/train', 'full', protocol['total_steps']) != resume_step:
            raise ValueError('Native Full4B must resume its newest signed optimizer step')
        receipt = json.loads((output / 'native_process.json').read_text())
        if receipt.get('smoke') is not False or type(receipt.get('pid')) is not int:
            raise RuntimeError('Original native Full4B process identity differs')
        try:
            os.killpg(receipt['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Original native Full4B training group is still alive')
        archive_resume_tail(output / 'run/train', resume_step)
    source = output / 'reference_source'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               WANDB_MODE='disabled', WANDB_DISABLED='true', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               PLATOUR_BRIDGE_PYTHON=sys.executable,
               PLATOUR_GENERATION_PYTHON=str(generation_python),
               PLATOUR_SCORING_PYTHON=str(scoring_python))
    env['PYTHONPATH'] = os.pathsep.join((str(source / 'recipe/llm_communicate'),
        str(source / 'scripts'), original_math_checker(), str(Path(__file__).resolve().parents[1]),
        os.environ.get('PYTHONPATH', '')))
    port = int(env.get('PLATOUR_NATIVE_MASTER_PORT', '39124'))
    if not 1024 <= port <= 65535:
        raise ValueError('Native Full4B rendezvous port must be in 1024..65535')
    env['MASTER_ADDR'] = '127.0.0.1'
    env['MASTER_PORT'] = str(port)
    cmd = [sys.executable, '-B', '-m', 'torch.distributed.run', '--nnodes=1',
           '--nproc-per-node=8', '--master-addr=127.0.0.1',
           f'--master-port={port}', '--max-restarts=0',
           '-m', 'qwen35.native_full', 'worker', '--root', str(output)]
    if resume_step is not None:
        cmd.extend(('--resume-step', str(resume_step)))
    mode = 'smoke' if protocol['smoke_only'] else 'train'
    atomic_json(output / 'phase.json', {'phase': mode, 'resume_step': resume_step,
                                        'world_size': 8, 'time': time.time()})
    try:
        from platour_exps import run_native_training
        run = output / 'run' / mode
        run_native_training(cmd, cwd=source, environment=env, output=output,
                            failure_directory=run, smoke=protocol['smoke_only'],
                            resume_step=resume_step)
        done = json.loads((run / 'complete.json').read_text())
        if not protocol['smoke_only']:
            from .native_formal import valid_checkpoint
            final_step = done.get('steps_run')
            if type(final_step) is not int or not 0 < final_step <= protocol['total_steps']:
                raise RuntimeError('Native Full4B final step is not an authorized optimizer boundary')
            path = run / 'evaluation' / f'step-{final_step:04d}' / 'summary.json'
            validation = json.loads(path.read_text()) if path.is_file() else {}
            if (not done.get('complete') or done.get('smoke') is not False or
                    done.get('world_size') != 8 or type(final_step) is not int or
                    not 0 < final_step <= protocol['total_steps'] or
                    not valid_checkpoint(run / 'checkpoints' / f'step-{final_step:04d}',
                                         'full', final_step) or
                    validation.get('responses') != 500 * protocol['eval_samples'] or
                    not validation.get('complete')):
                raise RuntimeError('Formal Full4B final weights, optimizer or full validation incomplete')
            atomic_json(output / 'phase.json', {'phase': 'complete', 'smoke': False,
                                                 'step': final_step, 'time': time.time()})
            return done
        if (not done.get('complete') or not done.get('smoke') or
                done.get('steps_run') != 2 or done.get('world_size') != 8 or
                done.get('optimizer_updates', 0) < 1 or
                not done.get('all_parameters_trainable')):
            raise RuntimeError('Native Full4B two-update structural qualification failed')
        for step in (0, 2):
            score = json.loads((run / f'evaluation/step-{step:04d}/summary.json').read_text())
            if (not score.get('complete') or score.get('responses') != 16 or
                    score.get('scored_responses') != 16 or
                    score.get('gold_missing') or score.get('scoring_timeouts')):
                raise RuntimeError('Native Full4B 8-question ×2 smoke evaluation incomplete')
        for rank in range(8):
            pressure = json.loads((run / f'stress.rank-{rank:02d}.json').read_text())
            if not pressure.get('passed') or pressure.get('response_tokens') != 16384:
                raise RuntimeError('Full4B all-parameter 16K GPU stress incomplete')
            checkpoint = run / 'checkpoints/step-0002'
            for name in (f'optimizer.rank-{rank:02d}.pt', f'rng.rank-{rank:02d}.pt'):
                if not (checkpoint / name).is_file():
                    raise RuntimeError('Full4B sharded FP32 optimizer/rank RNG incomplete: ' + name)
        atomic_json(output / 'phase.json', {'phase': 'complete', 'time': time.time()})
    except BaseException as error:
        atomic_json(output / 'phase.json', {'phase': 'failed', 'error': repr(error),
                                           'time': time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('worker', 'execute'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--vllm-python', type=Path)
    parser.add_argument('--scoring-python', type=Path)
    parser.add_argument('--resume-step', type=int,
                        help='Most recent signed full-model/8-rank optimizer step from this formal run')
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.root, resume_step=args.resume_step)
    elif args.vllm_python is None or args.scoring_python is None:
        parser.error('execute needs native vLLM and math-scoring Python')
    else:
        execute(args.root, generation_python=args.vllm_python,
                scoring_python=args.scoring_python, resume_step=args.resume_step)


if __name__ == '__main__':
    main()
