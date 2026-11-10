"""Replace the original qualification receipts with same-run evidence."""

import json
from pathlib import Path

from .checkpoints import sha256


def verify_evidence(root, protocol=None):
    """Require actual two-host NCCL + resident 16K FP8 smoke before formal 25."""
    root = Path(root).resolve()
    probe_path = root / 'probe_audit.json'
    smoke_path = root / 'smoke_audit.json'
    source_path = root / 'source_manifest.json'
    if not all(path.is_file() and not path.is_symlink()
               for path in (probe_path, smoke_path, source_path)):
        raise FileNotFoundError('Native DeepCoder needs same-source two-host probe and smoke receipts')
    probe, smoke = json.loads(probe_path.read_text()), json.loads(smoke_path.read_text())
    source = sha256(source_path)
    if (not probe.get('passed') or not probe.get('nccl_validated') or
            probe.get('cpu_simulation') or not probe.get('shared_storage_verified') or
            len(set(probe.get('hosts', ()))) != 2 or
            probe.get('source_manifest_sha256') != source or
            not smoke.get('passed') or smoke.get('source_manifest_sha256') != source):
        raise RuntimeError('Actual two-host NCCL, source or independent smoke audit differs')
    record = smoke.get('structural', {})
    profiles = smoke.get('worst_cases', ())
    if (not record.get('complete') or record.get('world_size') != 4 or
            not record.get('replica_exact_weights') or
            record.get('nonzero_gradient_updates', 0) < 1 or
            len(profiles) != 4 or any(not item.get('passed') or
                not item.get('vllm_resident') or item.get('response_tokens') != 16384
                for item in profiles)):
        raise RuntimeError('Original four-rank resident native DeepCoder smoke incomplete')
    proof = {'paths': {'probe': str(probe_path), 'smoke': str(smoke_path)},
             'sha256': {'probe': sha256(probe_path), 'smoke': sha256(smoke_path)},
             'components_passed': True, 'live_fp8_and_full_validation_required': True,
             'previous_launcher_exit_codes_observed': True,
             'procedure': 'Same-source two-host NCCL, rank-complete resident 16K smoke, '
                          'then trainer live FP8 stress and all 201x16 original validation answers'}
    if protocol is not None:
        if (protocol.get('world_size') != 4 or protocol.get('dataset') != 'deepcoder' or
                protocol.get('qualification_evidence') != proof):
            raise RuntimeError('Native full-budget startup evidence differs from signed protocol')
    return proof
