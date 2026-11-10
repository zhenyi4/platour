"""Read-only provenance and two-host gate for original 397B mathematics training."""

import json
from pathlib import Path

from .native_training import digest


ASSETS = Path(__file__).with_name('native_reference') / '20'
HASHES = Path(__file__).with_name('native_math20_source_sha256.json')


def verify_bundle():
    record = json.loads(HASHES.read_text())
    sources = record['files']
    actual = {path.relative_to(ASSETS).as_posix() for path in ASSETS.rglob('*')
              if path.is_file()}
    if (len(sources) != 42 or set(sources) != actual or
            record['original_science_files'] != 37 or
            record['historical_two_host_launcher_not_bundled'] is not True or
            record['original_private_ip_test_and_site_launcher_excluded'] is not True or
            any(Path(name).is_absolute() or '..' in Path(name).parts or
                (ASSETS / name).is_symlink() or digest(ASSETS / name) != value
                for name, value in sources.items())):
        raise ValueError('Frozen two-host 397B mathematics science source differs')
    return sources


def audit(case, signed, data_root):
    """Verify source, topology, and signed inputs; never contact a GPU or host."""
    if (case.identifier != 20 or case.task != 'dapo' or case.method != 'bridge' or
            case.sender != 'Qwen3.5-397B-A17B-FP8' or case.steps != 264 or
            case.val_samples != 4 or case.teacher_dim != 4096):
        raise ValueError('This audited two-host recipe applies only to mathematics ID20')
    files = verify_bundle()
    data_root = Path(data_root)
    manifest = json.loads((data_root / 'manifest.json').read_text())['dapo']
    for split, count in (('train', 16891), ('validation', 500)):
        path = data_root / 'dapo' / f'{split}.jsonl'
        if (manifest[split]['rows'] != count or
                digest(path) != manifest[split]['sha256'] or
                len(signed[split]) != count or
                [row['index'] for row in signed[split]] != list(range(count))):
            raise ValueError(f'Original mathematics {split} data identity differs')
        with path.open() as stream:
            for record, source in zip(signed[split], stream, strict=True):
                original = json.loads(source)
                if any(record[key] != original[key]
                       for key in ('index', 'gold', 'prompt_token_ids')):
                    raise ValueError(f'Original mathematics {split} token or gold differs')
    return {
        'experiment': 20, 'original_science_source_files_verified': 37,
        'total_bundled_files_verified': len(files),
        'training_questions': 16891, 'validation_questions': 500,
        'validation_samples_per_question': 4, 'total_steps': 264,
        'distributed_world_size': 4, 'teacher_tensor_parallel_size': 4,
        'required_hosts': 2, 'gpus_per_host': 8,
        'historical_397b_training_launcher_bundled': False,
        'portable_two_host_launcher': 'qwen35.native_multinode',
        'launch_enabled': False,  # This audit is read-only; see separate guarded launcher.
        'blockers': ['second authorized eight-GPU host and actual NCCL/shared-data proof',
                     'current 397B FP8 teacher/native vLLM 0.17.1 GPU qualification',
                     'complete math 500x4 validation and checkpoint replay'],
    }
