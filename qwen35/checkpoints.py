"""Crash-safe GRPO checkpoints: durable state before validation and one rolling predecessor."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.pending')
    with temporary.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def rng_state():
    import random
    import torch
    return {'cpu': torch.get_rng_state(), 'python': random.getstate(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import random
    import torch
    torch.set_rng_state(state['cpu'])
    if 'python' in state:  # checkpoints signed before the Python RNG was recorded lack it
        random.setstate(state['python'])
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise RuntimeError('CUDA RNG device count changed across resume')
        torch.cuda.set_rng_state_all(state['cuda'])


def valid(directory, *, expected_step=None):
    directory = Path(directory)
    marker = directory / 'complete.json'
    if not marker.is_file():
        return False
    record = read(marker)
    if not record.get('complete') or (expected_step is not None and record['step'] != expected_step):
        return False
    names = {'model.safetensors', 'optimizer.pt', 'rng.pt', 'checkpoint.json'}
    if set(record.get('sha256', {})) != names:
        return False
    return all((directory / name).is_file() and sha256(directory / name) == digest
               for name, digest in record['sha256'].items())


def latest(output):
    checkpoints = Path(output) / 'checkpoints'
    if not checkpoints.is_dir():
        return None
    candidates = sorted((int(p.name[5:]), p) for p in checkpoints.glob('step-*')
                        if p.is_dir() and p.name[5:].isdigit())
    for step, path in reversed(candidates):
        if valid(path, expected_step=step):
            return path
    return None


def save_step(output, step, actor, optimizer, record, *, protected=(), identity=None,
              rank_rng=None, previous_step=None):
    """Commit state; an explicit verified previous_step permits sparse in-process saves.

    The sparse path hashes each newly written file once. Its caller must have
    verified the previous state on load or written it in this process. Full
    integrity checks still run in latest/load_step; retirement only inspects
    commit metadata. The default path preserves the consecutive-step contract.
    """
    import torch
    from safetensors.torch import save_file
    output = Path(output)
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    dest = checkpoints / f'step-{step:04d}'
    if step < 0 or (dest.exists() and (valid(dest) if previous_step is None else (dest / 'complete.json').exists())):
        raise FileExistsError(f'Cannot overwrite a signed step or use a negative step: {dest}')
    if previous_step is None:
        old = latest(output)
        if old is None and step != 0:
            raise RuntimeError('Missing previous signed optimizer state')
        if old is not None and int(old.name[5:]) != step - 1:
            raise RuntimeError('Only a consecutive signed optimizer step may be checkpointed')
    else:
        signed = sorted(int(p.name[5:]) for p in checkpoints.glob('step-*')
                        if p.name[5:].isdigit() and (p / 'complete.json').is_file())
        if (type(previous_step) is not int or previous_step < -1 or step <= previous_step or
                (signed[-1] if signed else -1) != previous_step):
            raise RuntimeError('Sparse checkpoint predecessor differs from the latest committed state')
        if previous_step >= 0:
            marker = read(checkpoints / f'step-{previous_step:04d}' / 'complete.json')
            if not marker.get('complete') or marker.get('step') != previous_step:
                raise RuntimeError('Sparse checkpoint predecessor is not committed')
    if dest.exists():
        archive = output / 'recovery'
        archive.mkdir(exist_ok=True)
        dest.rename(archive / f'unsigned-step-{step:04d}-{time.time_ns()}')
    stage = checkpoints / f'.step-{step:04d}.{os.getpid()}.{time.time_ns()}.pending'
    stage.mkdir()
    with torch.no_grad():
        tensors = {key: tensor.detach().cpu().contiguous().clone()
                   for key, tensor in actor.inference_state().items()}
    save_file(tensors, stage / 'model.safetensors', metadata={'format': 'pt'})
    torch.save(optimizer.state_dict(), stage / 'optimizer.pt')
    if rank_rng is not None and len(rank_rng) > 1:
        torch.save({'world_size': len(rank_rng), 'ranks': list(rank_rng)}, stage / 'rng.pt')
    else:
        torch.save(rng_state(), stage / 'rng.pt')
    files = ('model.safetensors', 'optimizer.pt', 'rng.pt')
    hashes = {name: sha256(stage / name) for name in files}
    atomic_json(stage / 'checkpoint.json', {'step': step, 'identity': identity,
                                             'metric': record, 'sha256': hashes})
    hashes['checkpoint.json'] = sha256(stage / 'checkpoint.json')
    stage.replace(dest)
    atomic_json(dest / 'complete.json', {'complete': True, 'step': step,
                                          'sha256': hashes, 'protected': step in protected})
    if previous_step is None and not valid(dest, expected_step=step):
        raise RuntimeError('New checkpoint did not pass its own integrity check')
    # Keep n and n-1 plus protected validation steps. Sweep every older signed step, so a
    # crash between signing n and deleting n-2 cannot leave a stale checkpoint behind.
    for retired in sorted(checkpoints.glob('step-*')):
        if not retired.name[5:].isdigit():
            continue
        old = int(retired.name[5:])
        cutoff = step - 1 if previous_step is None else previous_step
        if old < cutoff and old not in protected and not retired.is_symlink() and retired.is_dir():
            if previous_step is None:
                removable = valid(retired, expected_step=old)
            else:
                marker = retired / 'complete.json'
                try:
                    value = read(marker) if marker.is_file() else {}
                except (OSError, ValueError):
                    value = {}
                removable = value.get('complete') and value.get('step') == old
            if removable:
                shutil.rmtree(retired)
    return dest


def load_step(path, actor, optimizer, *, expected_identity=None, rank=0, world=1):
    import torch
    from safetensors.torch import load_file
    path = Path(path)
    if not valid(path):
        raise RuntimeError('Cannot resume from an unsigned or corrupt checkpoint')
    meta = read(path / 'checkpoint.json')
    if expected_identity is not None and meta['identity'] != expected_identity:
        raise RuntimeError('Checkpoint experiment/data identity differs')
    state = load_file(path / 'model.safetensors', device='cpu')
    actor.load_inference_state(state)
    optimizer.load_state_dict(torch.load(path / 'optimizer.pt', map_location='cpu', weights_only=True))
    saved = torch.load(path / 'rng.pt', map_location='cpu', weights_only=True)
    if 'ranks' in saved:
        if saved.get('world_size') != world or len(saved['ranks']) != world:
            raise RuntimeError('Resume must use the same torchrun world size as the checkpoint '
                               f"(saved with {saved.get('world_size')}, now {world})")
        saved = saved['ranks'][rank]
    elif world != 1:
        raise RuntimeError('Resume must use the same torchrun world size as the checkpoint '
                           f'(saved with 1, now {world})')
    restore_rng(saved)
    return read(path / 'complete.json')['step'], meta['metric']


def reconcile_history(output, signed_step, checkpoint_record):
    """Archive uncommitted lines and restore exactly the signed contiguous prefix."""
    output = Path(output)
    metrics = output / 'metrics.jsonl'
    rows = []
    torn_tail = False
    if metrics.is_file():
        raw = metrics.read_bytes()
        lines = raw.split(b'\n')
        for number, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line.decode('utf-8')))
            except (UnicodeDecodeError, json.JSONDecodeError):
                if number != len(lines) - 1 or raw.endswith(b'\n'):
                    raise RuntimeError('A committed history line is corrupt') from None
                torn_tail = True
    kept = [row for row in rows if 1 <= row.get('step', -1) <= signed_step]
    if [row['step'] for row in kept] != list(range(1, len(kept) + 1)):
        raise RuntimeError('The prior metric history has gaps; do not silently fabricate steps')
    if signed_step and len(kept) == signed_step - 1:
        if checkpoint_record is None or checkpoint_record.get('step') != signed_step:
            raise RuntimeError('Signed final step has no recoverable metric record')
        kept.append(checkpoint_record)
    if len(kept) != signed_step:
        raise RuntimeError('Cannot reconcile logs to the latest signed checkpoint')
    if metrics.is_file() and (len(kept) != len(rows) or torn_tail):
        recovery = output / 'recovery'
        recovery.mkdir(exist_ok=True)
        archive = recovery / f'metrics-before-resume-{time.time_ns()}.jsonl'
        shutil.copyfile(metrics, archive)
    tmp = metrics.with_name(metrics.name + f'.{os.getpid()}.pending')
    with tmp.open('x') as stream:
        for row in kept:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(metrics)
    return kept
