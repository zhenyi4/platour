"""Offline, SHA-checked native rank-128 LoRA-GRPO for Qwen3.5 ID24."""

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


ASSETS = Path(__file__).with_name('native_reference') / '24'
HASHES = Path(__file__).with_name('native_lora_source_sha256.json')
PROTOCOLS = Path(__file__).with_name('native_protocols.json')


def verify_bundle():
    manifest = json.loads(HASHES.read_text())
    files = manifest['files']
    actual = {p.relative_to(ASSETS).as_posix() for p in ASSETS.rglob('*.py')}
    scripts = ('lora4b_train.py', 'full4b_train.py')
    for script in scripts:
        content = (ASSETS / 'recipe/llm_communicate' / script).read_text()
        if 'queued_wandb.json' in content or "mode='online'" in content:
            raise ValueError('Native LoRA dependencies may not launch a remote W&B run')
    if (len(files) != 42 or set(files) != actual or
            any('..' in Path(name).parts or digest(ASSETS / name) != sha
                for name, sha in files.items()) or
            files['recipe/llm_communicate/lora4b_train.py'] != manifest['offline_only_trainer_sha256'] or
            files['recipe/llm_communicate/full4b_train.py'] != manifest['offline_only_full4b_helper_sha256']):
        raise ValueError('Original rank-128 LoRA trainer or offline audit differs')
    return files


def prepare(case, receiver, signed, output, *, smoke=False):
    if case.identifier != 24 or case.method != 'lora' or case.task != 'dapo':
        raise ValueError('Only Qwen3.5 mathematics LoRA-GRPO experiment 24 is bundled')
    receiver = Path(receiver).resolve()
    if (receiver.name != 'Qwen3.5-4B' or not (receiver / 'config.json').is_file() or
            len(signed['train']) != 16891 or len(signed['validation']) != 500):
        raise ValueError('Native LoRA original 4B model or signed split size differs')
    sources = verify_bundle()
    protocol = dict(json.loads(PROTOCOLS.read_text())['24'])
    if (protocol['world_size'] != 8 or protocol['total_steps'] != case.steps or
            protocol['learning_rate'] != case.learning_rate or
            protocol['lora_rank'] != 128 or protocol['lora_alpha'] != 128 or
            protocol['eval_samples'] != 4 or
            tuple(protocol['eval_steps']) != case.validation_steps):
        raise ValueError('Original LoRA rank/alpha or GRPO schedule differs')
    physical = list(map(int, os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').split(',')))
    if len(physical) != 8 or len(set(physical)) != 8:
        raise ValueError('Native LoRA requires eight distinct GPUs')
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError('Native LoRA requires a fresh isolated container')
    output.mkdir(parents=True)
    frozen = output / 'reference_source'
    shutil.copytree(ASSETS, frozen)
    if not smoke:
        from .native_formal import SOURCES, patch_sources
        from .checkpoints import sha256
        originals = {name: (frozen / name).read_text() for name in SOURCES['lora']}
        modified = patch_sources('lora', originals)
        for name, text in modified.items():
            (frozen / name).write_text(text)
        atomic_json(output / 'formal_source_patch.json', {'experiment': 24,
            'original_sha256': {name: sources[name] for name in modified},
            'active_sha256': {name: sha256(frozen / name) for name in modified},
            'only_checkpoint_io_changed': True})
        sources = {**sources, **{name: sha256(frozen / name) for name in modified}}
    (frozen / 'data').mkdir()
    (frozen / 'data/models').symlink_to(receiver.parent, target_is_directory=True)
    atomic_json(output / 'source_manifest.json', {
        'reference_source/' + name: value for name, value in sources.items()})
    run = output / 'run'
    run.mkdir()
    atomic_json(run / 'train_prompts.json', signed['train'])
    atomic_json(run / 'eval_prompts.json', signed['validation'])
    protocol.update(student_checkpoint=str(receiver),
        scientific_source_manifest_sha256=digest(output / 'source_manifest.json'),
        train_prompts_sha256=digest(run / 'train_prompts.json'),
        eval_prompts_sha256=digest(run / 'eval_prompts.json'),
        gpu_ids=physical, wandb_mode='disabled', wandb_enabled=False,
        per_step_checkpoints=not smoke, rolling_predecessor=1,
        smoke_only=smoke, native_training_scope='frozen_base_rank128_alpha128_lora_grpo')
    from .native_selection import contract
    protocol.update(contract())
    atomic_json(run / 'protocol.json', protocol)
    manifest = json.loads(HASHES.read_text())
    atomic_json(output / 'source_audit.json', {
        'experiment': 24, 'frozen_source_files': len(sources),
        'original_scientific_trainer_sha256': manifest['unmodified_original_trainer_sha256'],
        'offline_only_scientific_trainer_sha256': manifest['offline_only_trainer_sha256'],
        'original_full4b_helper_sha256': manifest['unmodified_original_full4b_helper_sha256'],
        'offline_only_full4b_helper_sha256': manifest['offline_only_full4b_helper_sha256'],
        'only_logging_changed': True, 'full_budget_training_started': False,
        'model_or_adapter_weights_bundled': False})
    return output


def verify_prepared(output):
    output = Path(output).absolute()
    files = verify_bundle()
    run = output / 'run'
    protocol = json.loads((run / 'protocol.json').read_text())
    marker = output / 'formal_source_patch.json'
    if protocol.get('smoke_only') is False:
        from .native_formal import SOURCES, patch_sources
        modified = patch_sources('lora', {name: (ASSETS / name).read_text()
                                          for name in SOURCES['lora']})
        old = {name: digest(ASSETS / name) for name in modified}
        active = {name: hashlib.sha256(text.encode()).hexdigest()
                  for name, text in modified.items()}
        receipt = json.loads(marker.read_text()) if marker.is_file() else {}
        if (receipt.get('experiment') != 24 or receipt.get('original_sha256') != old or
                receipt.get('active_sha256') != active or
                receipt.get('only_checkpoint_io_changed') is not True):
            raise ValueError('Native LoRA full-budget checkpoint source seal differs')
        files = {**files, **active}
    elif protocol.get('smoke_only') is not True or marker.exists():
        raise ValueError('Native LoRA smoke source identity differs')
    prepared = json.loads((output / 'source_manifest.json').read_text())
    if prepared != {'reference_source/' + name: sha for name, sha in files.items()}:
        raise ValueError('Frozen native LoRA source closure differs')
    for name, sha in prepared.items():
        if digest(output / name) != sha:
            raise ValueError('Native LoRA source changed: ' + name)
    if (protocol['scientific_source_manifest_sha256'] != digest(output / 'source_manifest.json') or
            protocol['world_size'] != 8 or protocol['total_steps'] != 264 or
            protocol['lora_rank'] != 128 or protocol['lora_alpha'] != 128 or
            protocol.get('per_step_checkpoints') is not (not protocol['smoke_only']) or
            protocol.get('rolling_predecessor') != 1 or
            protocol['wandb_mode'] != 'disabled'):
        raise ValueError('Native LoRA signed offline full/smoke protocol differs')
    for split in ('train', 'eval'):
        if digest(run / f'{split}_prompts.json') != protocol[f'{split}_prompts_sha256']:
            raise ValueError('Signed LoRA training/validation input changed')
    from .native_selection import verify
    verify(protocol)
    return protocol


def worker(output, *, resume_step=None):
    output = Path(output).absolute()
    protocol = verify_prepared(output)
    frozen = output / 'reference_source'
    sys.path[:0] = [str(frozen / 'recipe/llm_communicate'), str(frozen / 'scripts')]
    import lora4b_train as trainer
    sys.argv = [str(frozen / 'recipe/llm_communicate/lora4b_train.py'),
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
        raise FileNotFoundError('Native rank-128 LoRA vLLM or scorer Python is missing')
    if resume_step is not None:
        from .native_formal import archive_resume_tail, newest_checkpoint
        if protocol['smoke_only'] or type(resume_step) is not int or resume_step < 1:
            raise ValueError('Only a signed unfinished formal LoRA run may resume')
        if newest_checkpoint(output / 'run/train', 'lora', protocol['total_steps']) != resume_step:
            raise ValueError('Native LoRA must resume its newest signed optimizer step')
        receipt = json.loads((output / 'native_process.json').read_text())
        if receipt.get('smoke') is not False or type(receipt.get('pid')) is not int:
            raise RuntimeError('Original native LoRA process identity differs')
        try:
            os.killpg(receipt['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Original native LoRA training group is still alive')
        archive_resume_tail(output / 'run/train', resume_step)
    source = output / 'reference_source'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        WANDB_MODE='disabled', WANDB_DISABLED='true', PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        PLATOUR_BRIDGE_PYTHON=sys.executable, PLATOUR_GENERATION_PYTHON=str(generation_python),
        PLATOUR_SCORING_PYTHON=str(scoring_python))
    env['PYTHONPATH'] = os.pathsep.join((str(source / 'recipe/llm_communicate'),
        str(source / 'scripts'), original_math_checker(), str(Path(__file__).resolve().parents[1]),
        os.environ.get('PYTHONPATH', '')))
    port = int(env.get('PLATOUR_NATIVE_MASTER_PORT', '39125'))
    if not 1024 <= port <= 65535:
        raise ValueError('Native LoRA rendezvous port must be in 1024..65535')
    env['MASTER_ADDR'] = '127.0.0.1'
    env['MASTER_PORT'] = str(port)
    command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--nnodes=1',
               '--nproc-per-node=8', '--master-addr=127.0.0.1',
               f'--master-port={port}', '--max-restarts=0', '-m',
               'qwen35.native_lora', 'worker', '--root', str(output)]
    if resume_step is not None:
        command.extend(('--resume-step', str(resume_step)))
    mode = 'smoke' if protocol['smoke_only'] else 'train'
    atomic_json(output / 'phase.json', {'phase': mode, 'resume_step': resume_step,
                                        'world_size': 8, 'time': time.time()})
    try:
        from platour_exps import run_native_training
        run = output / 'run' / mode
        run_native_training(command, cwd=source, environment=env, output=output,
                            failure_directory=run, smoke=protocol['smoke_only'],
                            resume_step=resume_step)
        done = json.loads((run / 'complete.json').read_text())
        if not protocol['smoke_only']:
            from .native_formal import valid_checkpoint
            final_step = done.get('steps_run')
            if type(final_step) is not int or not 0 < final_step <= protocol['total_steps']:
                raise RuntimeError('Native LoRA final step differs from approved budget')
            validation_file = run / 'evaluation' / f'step-{final_step:04d}' / 'summary.json'
            validation = json.loads(validation_file.read_text()) if validation_file.is_file() else {}
            if (not done.get('complete') or done.get('smoke') is not False or
                    done.get('world_size') != 8 or
                    not done.get('frozen_base_verified') or not done.get('native_lora') or
                    not valid_checkpoint(run / 'checkpoints' / f'step-{final_step:04d}',
                                         'lora', final_step) or
                    validation.get('responses') != 500 * protocol['eval_samples'] or
                    not validation.get('complete')):
                raise RuntimeError('Formal rank-128 LoRA adapter/optimizer or full VAL incomplete')
            atomic_json(output / 'phase.json', {'phase': 'complete', 'smoke': False,
                                                 'step': final_step, 'time': time.time()})
            return done
        if (not done.get('complete') or not done.get('smoke') or
                done.get('steps_run') != 2 or done.get('world_size') != 8 or
                done.get('effective_optimizer_updates', 0) < 1 or
                not done.get('native_lora') or not done.get('frozen_base_verified')):
            raise RuntimeError('Rank-128 native LoRA two-update structural checks failed')
        for step in (0, 2):
            score = json.loads((run / f'evaluation/step-{step:04d}/summary.json').read_text())
            if (not score.get('complete') or score.get('responses') != 16 or
                    score.get('scored_responses') != 16 or
                    score.get('gold_missing') or score.get('scoring_timeouts')):
                raise RuntimeError('Native LoRA 8-question ×2 smoke validation differs')
        for rank in range(8):
            stress = json.loads((run / f'stress.rank-{rank:02d}.json').read_text())
            if (not stress.get('passed') or stress.get('response_tokens') != 16384 or
                    not stress.get('optimizer_state_resident') or not stress.get('frozen_base')):
                raise RuntimeError('Native LoRA all-rank 16K BF16/FP32 pressure differs')
            checkpoint = run / 'checkpoints/step-0002'
            for name in (f'rng.rank-{rank:02d}.pt',):
                if not (checkpoint / name).is_file():
                    raise RuntimeError('Native LoRA rank RNG missing: ' + name)
        checkpoint = run / 'checkpoints/step-0002'
        if not (checkpoint / 'adapter.pt').is_file() or not (checkpoint / 'optimizer.pt').is_file():
            raise RuntimeError('Native LoRA FP32 adapter/optimizer checkpoint incomplete')
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
                        help='Nearest signed LoRA/FP32 AdamW/eight-rank RNG optimizer step')
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.root, resume_step=args.resume_step)
    elif args.vllm_python is None or args.scoring_python is None:
        parser.error('execute requires native LoRA vLLM/scorer Pythons')
    else:
        execute(args.root, generation_python=args.vllm_python,
                scoring_python=args.scoring_python, resume_step=args.resume_step)


if __name__ == '__main__':
    main()
