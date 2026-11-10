"""Read-only preflight for the frozen two-host 397B DeepCoder native recipe.

No training/judge launcher is provided: one eight-GPU host cannot run this recipe.
"""

import json
from pathlib import Path

from .native_training import digest
from .data import source_manifest, verify_materialized


ASSETS = Path(__file__).with_name('native_reference') / '25'
HASHES = Path(__file__).with_name('native_deepcoder_source_sha256.json')


def verify_bundle():
    record = json.loads(HASHES.read_text())
    sources = record['files']
    actual = {p.relative_to(ASSETS).as_posix() for p in ASSETS.rglob('*.py')}
    if (len(sources) != 41 or set(sources) != actual or
            record['excluded_external_host_paths'] != 6 or
            record['excluded_inactive_site_launcher_and_tests'] != 7 or
            any('..' in Path(name).parts or digest(ASSETS / name) != sha
                for name, sha in sources.items())):
        raise ValueError('Frozen 397B DeepCoder source closure differs')
    return sources


def audit(case, signed, data_root, *, case_root=None, verify_cases=False):
    """Validate public inputs and report missing resources without launching anything."""
    if (case.identifier != 25 or case.method != 'bridge' or
            case.task != 'deepcoder' or case.sender != 'Qwen3.5-397B-A17B-FP8' or
            case.steps != 295 or case.val_samples != 16):
        raise ValueError('This two-host source is specific to DeepCoder experiment 25')
    files = verify_bundle()
    manifest = json.loads((Path(data_root) / 'manifest.json').read_text())
    original = manifest['deepcoder']
    root = Path(data_root) / 'deepcoder'
    if (len(signed['train']) != 18873 or len(signed['validation']) != 201 or
            any(digest(root / (split + '.jsonl')) != original[split]['sha256']
                for split in ('train', 'validation')) or
            [row['index'] for row in signed['train']] != list(range(18873)) or
            [row['index'] for row in signed['validation']] != list(range(201))):
        raise ValueError('Original DeepCoder signed training/validation data differs')
    source_manifest(data_root)
    if verify_cases:
        if case_root is None:
            raise ValueError('Rebuilt DeepCoder case directory required')
        verify_materialized(data_root, case_root)
    return {
        'experiment': 25, 'source_files_verified': len(files),
        'training_questions': 18873, 'validation_questions': 201,
        'teacher_tp': 4, 'distributed_world_size': 4,
        'required_hosts': 2, 'gpus_per_host': 8, 'total_steps': 295,
        'judge_cases_sha256_verified': verify_cases,
        'judge': 'original isolated Docker code judge plus a pinned private CPU service',
        'portable_two_host_launcher': 'qwen35.native_multinode',
        'launch_enabled': False,  # This evidence audit cannot launch training.
        'blockers': ['second authorized eight-GPU host and actual NCCL/shared-data proof',
                     'original judge image/checker/case service supporting ID25',
                     'same-source four-rank resident 16K smoke and full 201x16 startup VAL'],
    }
