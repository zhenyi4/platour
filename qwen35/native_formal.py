"""Audited private checkpoint/rolling I/O for new native 4B full and LoRA runs."""

import ast
import json
import os
from pathlib import Path
import shutil
import time

from .checkpoints import atomic_json, sha256


SOURCES = {
    'full': ('recipe/llm_communicate/full4b_train.py',),
    'lora': ('recipe/llm_communicate/lora4b_core.py',
             'recipe/llm_communicate/lora4b_train.py'),
}


def _replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Native full-budget checkpoint patch source differs')
    return text.replace(old, new, 1)


def patch_sources(kind, originals):
    """Modify only fresh private copies; no rollout, rewards or policy loss edits."""
    if kind not in SOURCES or set(originals) != set(SOURCES[kind]):
        raise ValueError('Native full/LoRA original source closure differs')
    patched = dict(originals)
    if kind == 'full':
        name = SOURCES[kind][0]
        source = patched[name]
        source = _replace(source,
            "        cpu={key:value.detach().cpu().contiguous().clone() for key,value in model.state_dict().items()}\n"
            "        save_file(cpu,str(destination)+'.tmp');Path(str(destination)+'.tmp').replace(destination)\n"
            "        del cpu\n"
            "        save(directory/'identity.json',dict(state_sha256=identity,file_sha256=sha(destination),step=step))",
            "        if destination.is_symlink():\n"
            "            raise ValueError('Native signed model snapshot cannot redirect')\n"
            "        if destination.exists():\n"
            "            original=read(directory/'identity.json')\n"
            "            if original.get('step')!=step or original.get('state_sha256')!=identity or sha(destination)!=original.get('file_sha256'):\n"
            "                raise RuntimeError('Native resumed model snapshot differs from its signed state')\n"
            "        else:\n"
            "            cpu={key:value.detach().cpu().contiguous().clone() for key,value in model.state_dict().items()}\n"
            "            save_file(cpu,str(destination)+'.tmp');Path(str(destination)+'.tmp').replace(destination)\n"
            "            del cpu\n"
            "            save(directory/'identity.json',dict(state_sha256=identity,file_sha256=sha(destination),step=step))")
        source = _replace(source,
            '    dist.barrier()\n\n\ndef valid_backend_logprobs',
            "    dist.barrier()\n    if rank==0:\n"
            "        from qwen35.native_formal import sign_checkpoint\n"
            "        sign_checkpoint(directory, 'full', step, world)\n"
            "    dist.barrier()\n\n\ndef valid_backend_logprobs")
        source = _replace(source,
            "            state=dict(best_step=0,best_accuracy=baseline['accuracy'],fails=0)",
            "            state=dict(best_step=0,best_accuracy=baseline['accuracy'],fails=0)\n"
            "            if rank==0 and not args.smoke:save(run/'early_stop.json',dict(state=state,last_step=0,reason=None))")
        source = _replace(source,
            '        checkpoint(model,optimizer,run,step0,weights,rank,world)',
            '        if step0==0:checkpoint(model,optimizer,run,step0,weights,rank,world)')
        source = _replace(source,
            '            if reason:break',
            "            if not args.smoke and protocol['per_step_checkpoints'] and step not in protocol['eval_steps']:\n"
            "                checkpoint(model,optimizer,run,step,weights,rank,world)\n"
            '            if reason:break')
        patched[name] = source
    else:
        core, trainer = SOURCES[kind]
        patched[core] = _replace(patched[core],
            '    dist.barrier()\n\n\ndef load_checkpoint',
            "    dist.barrier()\n    if rank == 0:\n"
            "        from qwen35.native_formal import sign_checkpoint\n"
            "        sign_checkpoint(directory, 'lora', step, world)\n"
            "    dist.barrier()\n\n\ndef load_checkpoint")
        source = patched[trainer]
        source = _replace(source,
            '        checkpoint(actor, optimizer, run, step0, state, rank, world, frozen, sha)',
            '        if step0 == 0:\n            checkpoint(actor, optimizer, run, step0, state, rank, world, frozen, sha)')
        source = _replace(source,
            '            if reason:\n                break',
            "            if not args.smoke and protocol['per_step_checkpoints'] and step not in protocol['eval_steps']:\n"
            "                checkpoint(actor, optimizer, run, step, state, rank, world, frozen, sha)\n"
            '            if reason:\n                break')
        patched[trainer] = source
    for name, content in patched.items():
        ast.parse(content, filename=name)
    return patched


def required_files(kind, step, world):
    if kind == 'full':
        return ('checkpoint.json', *(f'optimizer.rank-{i:02d}.pt' for i in range(world)),
                *(f'rng.rank-{i:02d}.pt' for i in range(world)))
    if kind == 'lora':
        return ('checkpoint.json', 'adapter.pt', 'optimizer.pt',
                *(f'rng.rank-{i:02d}.pt' for i in range(world)))
    raise ValueError('Unknown native checkpoint kind')


def _check_dir(directory, kind, step, world):
    directory = Path(directory)
    if (directory.is_symlink() or not directory.is_dir() or
            directory.name != f'step-{step:04d}' or
            directory.parent.name != 'checkpoints' or type(world) is not int or world != 8):
        raise ValueError('Native checkpoint must belong to a private eight-rank run')
    meta_path = directory / 'checkpoint.json'
    if meta_path.is_symlink() or not meta_path.is_file():
        raise FileNotFoundError('Native checkpoint metadata is missing')
    metadata = json.loads(meta_path.read_text())
    if (metadata.get('complete') is not True or metadata.get('step') != step or
            metadata.get('world_size') != world):
        raise RuntimeError('Native optimizer checkpoint identity differs')
    names = required_files(kind, step, world)
    if any((directory / name).is_symlink() or not (directory / name).is_file() or
           (directory / name).stat().st_size == 0 for name in names):
        raise RuntimeError('Native full model/adapter, optimizer or rank RNG is incomplete')
    return metadata, names


def sign_checkpoint(directory, kind, step, world):
    """Only rank zero signs after all rank optimizer/RNG file barriers."""
    directory = Path(directory)
    marker = directory / 'complete.json'
    if marker.exists():
        raise FileExistsError('Never replace a signed native optimizer step')
    metadata, names = _check_dir(directory, kind, step, world)
    hashes = {name: sha256(directory / name) for name in names}
    if kind == 'full':
        weight = Path(metadata['weights'])
        if (weight.is_symlink() or not weight.is_file() or
                weight.parent.name != directory.name or weight.parent.parent.name != 'weights' or
                weight.resolve() != (directory.parent.parent / 'weights' / directory.name / 'model.safetensors').resolve() or
                sha256(weight) != metadata['weights_sha256']):
            raise RuntimeError('Native full weight snapshot differs from optimizer checkpoint')
        hashes['model.safetensors'] = metadata['weights_sha256']
    elif hashes['adapter.pt'] != metadata['adapter_sha256'] or hashes['optimizer.pt'] != metadata['optimizer_sha256']:
        raise RuntimeError('Native LoRA adapter/FP32 optimizer differs from checkpoint metadata')
    atomic_json(marker, {'complete': True, 'step': step, 'kind': kind,
                         'world_size': world, 'files': hashes})
    if not valid_checkpoint(directory, kind, step, world):
        raise RuntimeError('Newly signed native checkpoint failed its own verification')
    protocol = json.loads((directory.parent.parent.parent / 'protocol.json').read_text())
    if protocol.get('per_step_checkpoints'):
        retire_predecessor(directory.parent.parent, kind, step,
                           {0, *protocol['eval_steps']})
    return marker


def valid_checkpoint(directory, kind, step, world=8):
    directory = Path(directory)
    try:
        metadata, names = _check_dir(directory, kind, step, world)
        marker = json.loads((directory / 'complete.json').read_text())
        expected = set(names) | ({'model.safetensors'} if kind == 'full' else set())
        if (marker.get('complete') is not True or marker.get('kind') != kind or
                marker.get('step') != step or marker.get('world_size') != world or
                set(marker.get('files', {})) != expected or any(
                    sha256(directory / name) != marker['files'][name] for name in names)):
            return False
        if kind == 'full':
            weight = Path(metadata['weights'])
            if (weight.is_symlink() or weight.resolve() !=
                    (directory.parent.parent / 'weights' / directory.name / 'model.safetensors').resolve() or
                    sha256(weight) != marker['files']['model.safetensors']):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return False


def newest_checkpoint(run, kind, total_steps):
    """Find nearest fully signed FP32 optimizer and every rank's RNG; skip torn tail."""
    run = Path(run)
    checkpoints = run / 'checkpoints'
    if not checkpoints.is_dir() or checkpoints.is_symlink():
        raise FileNotFoundError('Native run has no signed checkpoint directory')
    steps = sorted((int(p.name[5:]) for p in checkpoints.glob('step-[0-9][0-9][0-9][0-9]')
                    if p.is_dir() and p.name[5:].isdigit()), reverse=True)
    for step in steps:
        if 0 < step < total_steps and valid_checkpoint(checkpoints / f'step-{step:04d}', kind, step):
            return step
    raise RuntimeError('No complete native optimizer and eight-rank RNG step to resume')


def archive_resume_tail(run, step):
    """Preserve all later interrupted artifacts before resuming; never drop signed input."""
    run = Path(run)
    if (run.is_symlink() or not valid_checkpoint(run / 'checkpoints' / f'step-{step:04d}',
                                                 'full' if (run / 'checkpoints' / f'step-{step:04d}' / 'optimizer.rank-00.pt').exists() else 'lora', step)):
        raise RuntimeError('Cannot archive after an unsigned checkpoint')
    recovery = run / 'recovery' / f'from-step-{step:04d}-{time.time_ns()}'
    recovery.mkdir(parents=True)
    for folder in ('checkpoints', 'weights', 'rollouts', 'evaluation', 'decoder_audit'):
        base = run / folder
        if not base.is_dir() or base.is_symlink():
            continue
        for path in base.glob('step-[0-9][0-9][0-9][0-9]'):
            if int(path.name[5:]) > step:
                if path.is_symlink():
                    raise ValueError('Native interrupted tail may not redirect through symlinks')
                destination = recovery / folder
                destination.mkdir(exist_ok=True)
                path.rename(destination / path.name)
    for path in (*run.glob('failure.rank-*.json'), run / 'workers', run / 'scorer'):
        if path.exists():
            if path.is_symlink():
                raise ValueError('Native failure/worker artifact may not redirect through symlinks')
            path.rename(recovery / path.name)
    metrics = run / 'metrics.jsonl'
    if not metrics.is_file() or metrics.is_symlink():
        raise FileNotFoundError('Committed native training history is missing')
    data = metrics.read_bytes()
    lines = data.splitlines(keepends=True)
    prefix = []
    for line in lines:
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            if line != lines[-1] or data.endswith(b'\n'):
                raise ValueError('Committed native metric line is corrupt') from None
            break
        if row.get('step') == len(prefix) + 1 and row['step'] <= step:
            prefix.append(line)
        elif row.get('step') is None or row['step'] <= step:
            raise ValueError('Committed native metric history is not contiguous')
    if len(prefix) != step:
        raise ValueError('Signed native step has no corresponding complete metrics')
    (recovery / 'metrics-before-resume.jsonl').write_bytes(data)
    pending = metrics.with_suffix('.pending')
    if pending.exists():
        raise FileExistsError('Refusing existing pending native metric rewrite')
    pending.write_bytes(b''.join(prefix))
    pending.replace(metrics)
    atomic_json(recovery / 'archive.json', {'resume_step': step, 'preserved_later_files': True,
                                            'metric_count': len(prefix)})
    return recovery


def retire_predecessor(run, kind, step, protected):
    """Retain signed n/n+1 and protected validation steps in fresh private runs.

    Every signed, unprotected step up to n-2 is removed, so a crash between signing n and
    deleting n-2 cannot leave a stale checkpoint behind for the rest of the run.
    """
    base = Path(run)
    for checkpoint in sorted((base / 'checkpoints').glob('step-[0-9][0-9][0-9][0-9]')):
        previous = int(checkpoint.name[5:])
        if previous <= 0 or previous > step - 2 or previous in protected:
            continue
        if checkpoint.is_symlink() or not valid_checkpoint(checkpoint, kind, previous):
            if previous == step - 2:
                raise RuntimeError('Refusing to retire an unsigned or linked native predecessor')
            continue
        shutil.rmtree(checkpoint)
        if kind == 'full':
            weights = base / 'weights' / f'step-{previous:04d}'
            if weights.is_symlink():
                raise ValueError('Refusing linked native full model weight deletion')
            if weights.is_dir():
                shutil.rmtree(weights)
