"""Frozen original native mathematics bridges for 35B PLatour and 4B Self-Communication."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from .checkpoints import atomic_json


ASSETS = Path(__file__).with_name('native_reference') / '21'
HASHES = Path(__file__).with_name('native_source_sha256.json')
PROTOCOLS = Path(__file__).with_name('native_protocols.json')
BRIDGE_HASHES = Path(__file__).with_name('native_bridge_source_sha256.json')
BRIDGE_PROTOCOLS = Path(__file__).with_name('native_bridge_protocols.json')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(piece)
    return result.hexdigest()


MATH_CHECKER = ASSETS.parent / '19' / 'third_party' / 'verl'
MATH_CHECKER_SHA256 = '9015798e362ac6811499661ce14d198cd8982910ada43e7e5a2901a9e2fc9d9a'


def original_math_checker():
    """Import root of the original ``verl.utils.reward_score.math_verify`` (byte-identical, Apache-2.0).

    The frozen DAPO scorers import it from ``<source>/third_party/verl``; the 21–24 bundles
    omit that directory, so their scorer gets the copy shipped with the ID19 bundle.
    """
    if digest(MATH_CHECKER / 'verl/utils/reward_score/math_verify.py') != MATH_CHECKER_SHA256:
        raise ValueError('Bundled original verl math_verify checker differs')
    return str(MATH_CHECKER)


def verify_bundle(identifier=21):
    if identifier not in (19, 21):
        raise ValueError('Only original native mathematics bridges 19 and 21 are bundled')
    assets = ASSETS.parent / str(identifier)
    sources = (json.loads(BRIDGE_HASHES.read_text())['19']['files'] if identifier == 19
               else json.loads(HASHES.read_text()))
    actual = {p.relative_to(assets).as_posix() for p in
              (assets.rglob('*') if identifier == 19 else assets.rglob('*.py')) if p.is_file()}
    if (len(sources) != (36 if identifier == 19 else 47) or actual != set(sources) or
            any('..' in Path(key).parts or Path(key).is_absolute() or
                digest(assets / key) != value for key, value in sources.items())):
        raise ValueError(f'Qwen3.5-{identifier} frozen native source or license differs')
    return sources


def zero_native_19_sources(core_text, trainer_text):
    """Create a private ID19 zero-P variant without modifying the frozen original sources."""
    import ast

    old = ("    bound = 1e-4 / math.sqrt(bridge.teacher.config.hidden_size)\n"
           "    rng = torch.Generator(device=bridge.student.device).manual_seed(42)\n"
           "    with torch.no_grad():\n"
           "        for p in bridge.bridge_parameters():\n"
           "            p.uniform_(-bound, bound, generator=rng)\n"
           "    bridge.initialization = 'random'\n"
           "    return dict(kind='small random uniform', seed=42, bound=bound,\n"
           "                scale=1e-4, constant_gate=1., sft_checkpoint_loaded=False)")
    new = ("    with torch.no_grad():\n"
           "        for p in bridge.bridge_parameters():\n"
           "            p.zero_()\n"
           "    bridge.initialization = 'zero'\n"
           "    return dict(kind='exact-zero linear bridge', seed=42,\n"
           "                projection_initialization='exact zero', constant_gate=1.,\n"
           "                sft_checkpoint_loaded=False)")
    old_metadata = "initialization=f'small random uniform +/-1e-4/sqrt({bridge.teacher.config.hidden_size})'"
    if core_text.count(old) != 1 or trainer_text.count(old_metadata) != 1:
        raise ValueError('Frozen native ID19 source differs; refuse a blind initialization edit')
    core_text = core_text.replace(old, new, 1)
    trainer_text = trainer_text.replace(old_metadata, "initialization='exact-zero linear bridge'", 1)
    ast.parse(core_text)
    ast.parse(trainer_text)
    return core_text, trainer_text


def prepare(case, sender, receiver, signed, output, *, smoke=False):
    """Freeze signed train/validation inputs before eight or four native GPU ranks."""
    if case.identifier not in (19, 21) or case.method != 'bridge' or case.task != 'dapo':
        raise ValueError('Only signed native mathematics bridges 19 and 21 are bundled')
    sender = Path(sender).resolve()
    receiver = Path(receiver).resolve()
    if (sender.name != case.sender or receiver.name != 'Qwen3.5-4B' or
            sender.parent != receiver.parent or
            not all((model / 'config.json').is_file() for model in (sender, receiver)) or
            (case.identifier == 21 and sender != receiver)):
        raise ValueError('Native bridge needs its original sender and a local Qwen3.5-4B receiver')
    if (len(signed['train']) != 16891 or len(signed['validation']) != 500 or
            [row['index'] for row in signed['train']] != list(range(16891)) or
            [row['index'] for row in signed['validation']] != list(range(500))):
        raise ValueError('Native mathematics original signed train/validation identity differs')
    sources = verify_bundle(case.identifier)
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError('Native Qwen3.5 training requires a fresh output root')
    visible = list(map(int, os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').split(',')))
    if len(visible) != 8 or len(set(visible)) != 8:
        raise ValueError('Native mathematics bridges require eight distinct GPUs')
    original = json.loads((BRIDGE_PROTOCOLS if case.identifier == 19 else PROTOCOLS).read_text())[str(case.identifier)]
    parallel = 1 if case.identifier == 19 else 2
    if (original['total_steps'] != case.steps or original['world_size'] != 8 // parallel or
            original['batch_size'] != 64 or original['group_size'] != 8 or
            original['learning_rate'] != case.learning_rate or
            tuple(original['eval_steps']) != case.validation_steps or
            tuple(original['checkpoint_steps']) != case.validation_steps or
            tuple(original['layer_map']) != case.layer_map):
        raise ValueError('Original bridge schedule or layer mapping differs from public recipe')
    output.mkdir(parents=True)
    frozen = output / 'reference_source'
    shutil.copytree(ASSETS.parent / str(case.identifier), frozen)
    if case.identifier == 19:
        core_name = 'recipe/llm_communicate/qwen35_bridge_opd_core.py'
        trainer_name = 'recipe/llm_communicate/train_qwen35_bridge_opd_vllm.py'
        core, trainer = frozen / core_name, frozen / trainer_name
        new_core, new_trainer = zero_native_19_sources(core.read_text(), trainer.read_text())
        core.write_text(new_core)
        trainer.write_text(new_trainer)
        atomic_json(output / 'zero_bridge_initialization_patch.json', {
            'experiment': 19, 'original_sha256': {
                core_name: sources[core_name], trainer_name: sources[trainer_name]},
            'active_sha256': {core_name: digest(core), trainer_name: digest(trainer)},
            'not_identical_to_published_id19_training_initialization': True})
    (frozen / 'data').mkdir()
    (frozen / 'data/models').symlink_to(receiver.parent, target_is_directory=True)
    atomic_json(output / 'source_manifest.json', {'files': sources,
        'excluded_inactive_site_launcher': True})
    atomic_json(output / 'train_prompts.json', signed['train'])
    atomic_json(output / 'eval_prompts.json', signed['validation'])
    protocol = dict(original)
    protocol.update(teacher_checkpoint=str(sender), student_checkpoint=str(receiver),
        source_model_revision=str(receiver), source_execution_root=str(frozen),
        gpu_ids=visible, decoder_gpu_ids=visible[::parallel],
        teacher_gpu_groups=[visible[i:i + parallel] for i in range(0, 8, parallel)],
        train_prompts_sha256=digest(output / 'train_prompts.json'),
        eval_prompts_sha256=digest(output / 'eval_prompts.json'),
        source_manifest_sha256=digest(output / 'source_manifest.json'),
        wandb_mode='disabled', wandb_enabled=False, wandb_project='disabled',
        wandb_run_name=f'native_bridge{case.identifier}_dapo_reference',
        native_experiment_id=case.identifier,
        execution_backend='frozen native bridge vLLM GRPO', smoke_only=smoke)
    if case.identifier == 19:
        protocol.update(initialization='exact-zero linear bridge',
                        historical_initialization=original['initialization'],
                        zero_initialization_variant=True,
                        execution_backend='native ID19 zero-initialized private variant')
    from .native_selection import contract
    protocol.update(contract())
    atomic_json(output / 'protocol.json', protocol)
    atomic_json(output / 'source_audit.json', {
        'experiment': case.identifier, 'source_files': len(sources),
        'site_launcher_excluded': True, 'frozen_source_manifest_sha256':
            protocol['source_manifest_sha256'], 'signed_train_questions': 16891,
        'signed_validation_questions': 500, 'portable_hf_rollouts_used': False})
    return output


def verify_prepared(root):
    root = Path(root).absolute()
    protocol = json.loads((root / 'protocol.json').read_text())
    expected = json.loads((root / 'source_manifest.json').read_text())['files']
    identifier = protocol.get('native_experiment_id', 21)
    if expected != verify_bundle(identifier) or digest(root / 'source_manifest.json') != protocol['source_manifest_sha256']:
        raise ValueError('Frozen Qwen3.5 native source bundle changed')
    variant = root / 'zero_bridge_initialization_patch.json'
    active = {}
    if protocol.get('zero_initialization_variant'):
        if identifier != 19 or protocol.get('initialization') != 'exact-zero linear bridge' or not variant.is_file():
            raise ValueError('Native ID19 zero bridge source identity differs')
        core_name = 'recipe/llm_communicate/qwen35_bridge_opd_core.py'
        trainer_name = 'recipe/llm_communicate/train_qwen35_bridge_opd_vllm.py'
        original_core = (ASSETS.parent / '19' / core_name).read_text()
        original_trainer = (ASSETS.parent / '19' / trainer_name).read_text()
        new_core, new_trainer = zero_native_19_sources(original_core, original_trainer)
        active = {core_name: hashlib.sha256(new_core.encode()).hexdigest(),
                  trainer_name: hashlib.sha256(new_trainer.encode()).hexdigest()}
        seal = json.loads(variant.read_text())
        if (seal.get('experiment') != 19 or
                seal.get('original_sha256') != {name: expected[name] for name in active} or
                seal.get('active_sha256') != active or
                not seal.get('not_identical_to_published_id19_training_initialization')):
            raise ValueError('Native ID19 zero source seal differs from its frozen originals')
    elif variant.exists():
        raise ValueError('Unexpected native zero initialization patch on a historical run')
    for name, value in expected.items():
        if digest(root / 'reference_source' / name) != active.get(name, value):
            raise ValueError('Prepared native trainer file differs: ' + name)
    for name in ('train', 'eval'):
        if digest(root / f'{name}_prompts.json') != protocol[f'{name}_prompts_sha256']:
            raise ValueError('Prepared Qwen3.5 train/validation split changed')
    original = json.loads((BRIDGE_PROTOCOLS if identifier == 19 else PROTOCOLS).read_text())[str(identifier)]
    protected = ('teacher', 'student', 'dataset', 'method', 'epochs', 'batch_size',
                 'group_size', 'learning_rate', 'max_prompt_length', 'max_response_length',
                 'layer_map', 'layer_mapping', 'train_sampling', 'eval_sampling',
                 'eval_steps', 'checkpoint_steps', 'eval_samples', 'val_metric',
                 'math_verify_boxed_only', 'vllm_memory_fraction')
    if (protocol['world_size'] != (8 if identifier == 19 else 4) or
            protocol['total_steps'] != 264 or protocol['wandb_mode'] != 'disabled' or
            protocol['source_execution_root'] != str(root / 'reference_source') or
            (protocol.get('historical_initialization') != original['initialization']
             if active else protocol.get('initialization') != original.get('initialization')) or
            any(protocol.get(key) != original.get(key) for key in protected)):
        raise ValueError('Native mathematics bridge execution protocol changed')
    from .native_selection import verify
    verify(protocol)
    return protocol


def verify_resume(root, protocol, step):
    """Check original formal checkpoint, rank RNG and metric prefix before GPUs."""
    import torch

    root = Path(root).absolute()
    run = root / 'train'
    if (type(step) is not int or protocol['smoke_only'] or
            step not in protocol['checkpoint_steps'] or
            not 0 < step < protocol['total_steps'] or
            (run / 'complete.json').exists()):
        raise ValueError('Resume requires an incomplete formal run and saved validation step')
    checkpoint = run / 'checkpoints' / f'step-{step:04d}'
    if checkpoint.is_symlink():
        raise ValueError('Native resume cannot follow an external checkpoint link')
    metadata = json.loads((checkpoint / 'checkpoint.json').read_text())
    initial = json.loads((run / 'initialization.json').read_text())
    if (metadata.get('step') != step or
            metadata.get('initial_bridge_state_sha256') != initial['bridge_state_sha256'] or
            digest(checkpoint / 'bridge.pt') != metadata.get('bridge_sha256')):
        raise ValueError('Native bridge checkpoint identity differs')
    optimizer = torch.load(checkpoint / 'optimizer.pt', map_location='cpu',
                           weights_only=True, mmap=True)
    if (optimizer.get('step') != step or
            optimizer.get('initial_sha256') != initial['bridge_state_sha256'] or
            not isinstance(optimizer.get('optimizer'), dict)):
        raise ValueError('Native FP32 optimizer checkpoint identity differs')
    del optimizer
    for rank in range(protocol['world_size']):
        path = checkpoint / f'rng.rank-{rank:02d}.pt'
        value = torch.load(path, weights_only=True, map_location='cpu')
        if not torch.is_tensor(value.get('cpu_rng')) or not torch.is_tensor(value.get('cuda_rng')):
            raise ValueError('Native rank CPU/CUDA RNG snapshot is incomplete')
    for boundary in (0, *(number for number in protocol['eval_steps'] if number <= step)):
        summary = json.loads((run / 'evaluation' /
                              f'step-{boundary:04d}' / 'summary.json').read_text())
        if (summary.get('step') != boundary or not summary.get('complete') or
                summary.get('questions') != 500 or summary.get('responses') != 2000 or
                summary.get('scored_responses') != 2000 or
                summary.get('gold_missing') or summary.get('scoring_timeouts')):
            raise ValueError('Prior native 500x4 validation checkpoint is incomplete')
        if boundary == step and summary.get('bridge_state_sha256') != metadata.get('bridge_state_sha256'):
            raise ValueError('Native validation bridge differs from signed checkpoint')
    metrics = run / 'metrics.jsonl'
    raw = metrics.read_bytes()
    rows = []
    torn = False
    for index, line in enumerate(raw.splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if index != len(raw.splitlines()) - 1 or raw.endswith(b'\n'):
                raise RuntimeError('Committed native metric line is corrupt') from None
            torn = True
    if [row.get('step') for row in rows[:step]] != list(range(1, step + 1)):
        raise ValueError('Prior native optimizer metric steps have gaps')
    if torn:
        archive = run / 'resume_archive'
        archive.mkdir(exist_ok=True)
        shutil.copyfile(metrics, archive / f'metrics-torn-before-step-{step:04d}-{time.time_ns()}.jsonl')
        temporary = metrics.with_suffix('.pending')
        temporary.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        temporary.replace(metrics)
    return checkpoint


def worker(root, *, resume_step=None):
    root = Path(root).absolute()
    protocol = verify_prepared(root)
    frozen = root / 'reference_source'
    sys.path[:0] = [str(frozen / 'recipe/llm_communicate'), str(frozen / 'scripts'),
                    str(frozen / 'third_party/verl')]
    if protocol.get('native_experiment_id', 21) == 21:
        import qwen35_multinode_config as configuration

        def verify_sources_without_site_launcher():
            return verify_prepared(root)['source_manifest_sha256']

        configuration.verify_sources = verify_sources_without_site_launcher
    import train_qwen35_bridge_opd_vllm as trainer
    sys.argv = [str(frozen / 'recipe/llm_communicate/train_qwen35_bridge_opd_vllm.py'),
                '--root', str(root), '--objective', 'grpo']
    if protocol['smoke_only']:
        sys.argv += ['--smoke', '--stress']
    if resume_step is not None:
        sys.argv += ['--resume-step', str(resume_step)]
    from .native_selection import install
    install(trainer)
    trainer.main()


def execute(root, *, generation_python, scoring_python, resume_step=None):
    root = Path(root).absolute()
    protocol = verify_prepared(root)
    if resume_step is not None:
        prior = json.loads((root / 'native_process.json').read_text())
        if type(prior.get('pid')) is not int or prior['pid'] <= 1:
            raise ValueError('Prior native process group identity differs')
        try:
            os.killpg(prior['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Prior native training is still running')
        verify_resume(root, protocol, resume_step)
    for python in (generation_python, scoring_python):
        if not Path(python).is_file():
            raise FileNotFoundError('Native generation or scoring Python is missing')
    environment = dict(os.environ, WANDB_MODE='disabled', WANDB_DISABLED='true',
        PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        PLATOUR_BRIDGE_PYTHON=sys.executable, PLATOUR_GENERATION_PYTHON=str(generation_python),
        PLATOUR_SCORING_PYTHON=str(scoring_python), CUDA_VISIBLE_DEVICES=','.join(
            map(str, protocol['gpu_ids'])))
    source = root / 'reference_source'
    environment['PYTHONPATH'] = os.pathsep.join((str(source / 'recipe/llm_communicate'),
        str(source / 'scripts'), str(source / 'third_party/verl'), original_math_checker(),
        str(Path(__file__).resolve().parents[1]), os.environ.get('PYTHONPATH', '')))
    for key in ('WANDB_API_KEY', 'WANDB_RUN_ID', 'WANDB_ENTITY'):
        environment.pop(key, None)
    identifier = protocol.get('native_experiment_id', 21)
    port = int(environment.get('PLATOUR_NATIVE_MASTER_PORT', str(39100 + identifier)))
    if not 1024 <= port <= 65535:
        raise ValueError('Native Qwen3.5 rendezvous port must be in 1024..65535')
    environment['MASTER_ADDR'] = '127.0.0.1'
    environment['MASTER_PORT'] = str(port)
    command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--nnodes=1',
               f"--nproc-per-node={protocol['world_size']}", '--master-addr=127.0.0.1',
               f'--master-port={port}', '--max-restarts=0', '-m',
               'qwen35.native_training', 'worker', '--root', str(root)]
    if resume_step is not None:
        command += ['--resume-step', str(resume_step)]
        archive = root / 'recovery'
        archive.mkdir(exist_ok=True)
        shutil.copyfile(root / 'phase.json',
                        archive / f'phase-before-resume-{resume_step:04d}-{time.time_ns()}.json')
    atomic_json(root / 'phase.json', {'phase': 'smoke' if protocol['smoke_only'] else 'train',
                                      'time': time.time(), 'world_size': protocol['world_size'],
                                      'resumed_from_step': resume_step})
    try:
        from platour_exps import run_native_training
        folder = root / ('smoke' if protocol['smoke_only'] else 'train')
        run_native_training(command, cwd=source, environment=environment, output=root,
                            failure_directory=folder, smoke=protocol['smoke_only'],
                            resume_step=resume_step)
        completed = json.loads((folder / 'complete.json').read_text())
        if (not completed.get('complete') or
                completed.get('world_size') != protocol['world_size'] or
                not completed.get('replica_exact_weights') or
                completed.get('nonzero_gradient_updates', 0) < 1):
            raise RuntimeError('Native mathematics bridge training incomplete')
        if protocol['smoke_only']:
            if not completed.get('singleton_tail_verified') or completed.get('steps') != 2:
                raise RuntimeError('Native bridge two-update/tail qualification differs')
            summary = json.loads((folder / 'evaluation/step-0002/summary.json').read_text())
            questions, responses = (6, 12) if identifier == 19 else (500, 2000)
            if (not summary['complete'] or summary['responses'] != responses or
                    summary['questions'] != questions or
                    summary.get('scored_responses') != responses or
                    summary.get('scoring_timeouts') or summary.get('gold_missing')):
                raise RuntimeError('Original native smoke validation denominator differs')
            profiles = ([folder / 'worst_case_profile.json'] if identifier == 19 else
                        [folder / f'worst_case_profile.rank-{rank:02d}.json' for rank in range(4)])
            for path in profiles:
                profile = json.loads(path.read_text())
                if (not profile['passed'] or not profile['vllm_resident'] or
                        profile.get('response_tokens') != 16384):
                    raise RuntimeError('Native 16K GPU pressure differs')
            checkpoint = folder / 'checkpoints/step-0002'
            if not all((checkpoint / name).is_file() for name in
                       ('bridge.pt', 'optimizer.pt', 'checkpoint.json',
                        *(f'rng.rank-{rank:02d}.pt' for rank in range(protocol['world_size'])))):
                raise RuntimeError('Native bridge optimizer/rank RNG checkpoint is incomplete')
        elif (completed.get('steps') != protocol['total_steps'] or
              (completed.get('steps_run', completed['steps']) < protocol['total_steps'] and
               not completed.get('early_stopped'))):
            raise RuntimeError('Native bridge budget or original early-stop certificate differs')
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
        parser.error('execute requires --vllm-python and --scoring-python')
    else:
        execute(args.root, generation_python=args.vllm_python,
                scoring_python=args.scoring_python, resume_step=args.resume_step)


if __name__ == '__main__':
    main()
