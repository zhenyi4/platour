"""Publish and restore frozen IT inputs without model/GPU imports."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

from platour_data import _install, safe_path, verify


PREFIX = 'it/v1'
RELEASE = Path(__file__).with_name('qwen3') / 'it_release.json'
DEFAULT_ROOT = Path(__file__).with_name('data') / 'it/qwen3/archived-it-v1'


def release():
    value = json.loads(RELEASE.read_text())
    if (value['profile'] != 'archived-it-v1' or value['prefix'] != PREFIX
            or not re.fullmatch(r'[0-9a-f]{40}', value['revision'])):
        raise ValueError('Invalid pinned IT release')
    raw = (json.dumps(value['manifest'], ensure_ascii=False, indent=2) + '\n').encode()
    if hashlib.sha256(raw).hexdigest() != value['manifest_sha256']:
        raise ValueError('Pinned IT manifest differs')
    return value


def supported_experiments():
    return sorted(int(name[:2]) for name in release()['manifest']['jobs'])


def ensure_inputs(experiments, root=DEFAULT_ROOT, *, cache_dir=None, local_files_only=False):
    """Download only selected fixed pairings plus shared supplement provenance."""
    from huggingface_hub import hf_hub_download
    root = Path(root)
    spec = release()
    manifest = spec['manifest']
    requested = set(experiments)
    available = {int(name[:2]): name for name in manifest['jobs']}
    if not requested or not requested <= available.keys():
        raise ValueError('Frozen Qwen3 IT supports only experiments ' + str(sorted(available)))
    def fetch(name):
        return hf_hub_download(repo_id=spec['repo_id'], repo_type='dataset',
            revision=spec['revision'], filename=f'{PREFIX}/{name}', cache_dir=cache_dir,
            local_files_only=local_files_only)
    for experiment in sorted(requested):
        job = available[experiment]
        for filename in ('rows.json', 'pairs.json', 'config.json'):
            name = f'inputs/{job}/{filename}'
            _install(root/job/filename, manifest['files'][name], lambda n=name: fetch(n))
    name = 'supplements/rephrases.jsonl'
    _install(root/name, manifest['files'][name], lambda: fetch(name))
    return [root/available[i] for i in sorted(requested)]


def load_inputs(experiment, root=DEFAULT_ROOT, *, test_rows=None, **options):
    """Return the frozen test and effective historical donors, with a coverage audit.

    Custom tests must match indices, full prompts and scoring fields exactly.
    Duplicate full prompts retain the archived last-pairing-wins behavior.
    """
    folder, = ensure_inputs([experiment], root, **options)
    spec = release()
    rows, pairs, config = [json.loads((folder/n).read_text())
                           for n in ('rows.json', 'pairs.json', 'config.json')]
    indices = [r['index'] for r in rows]
    if (len(rows) != config['questions'] or len(set(indices)) != len(rows)
            or len(pairs) != len(rows) or {p['test_index'] for p in pairs} != set(indices)):
        raise ValueError('Incomplete frozen IT coverage')
    fields = ('prompt_token_ids', 'gold', 'labels', 'valid_labels')
    if test_rows is not None:
        supplied = {r['index']: r for r in test_rows}
        if (len(test_rows) != len(rows) or set(supplied) != set(indices)
                or any(any(r.get(k) != supplied[r['index']].get(k) for k in fields) for r in rows)):
            raise ValueError('Test prompts/gold differ from archived-it-v1; do not apply donors by index or length alone')
    if (config['experiment'] != experiment or config['samples'] < 1 or config['cap'] < 1
            or config['seed_stride'] != config['samples']
            or config['mapping'] != ('last_aligned' if experiment <= 12 else 'uniform_endpoints')):
        raise ValueError('Frozen IT protocol differs')
    supplements = [json.loads(s) for s in (Path(root)/'supplements/rephrases.jsonl').read_text().splitlines()]
    keys = {(s['test_index'], tuple(s['teacher_prompt_token_ids'])) for s in supplements}
    by_index = {r['index']: r for r in rows}
    from collections import Counter
    counts = Counter()
    for pair in pairs:
        a, b = pair['student_prompt_ids'], pair['teacher_prompt_ids']
        if (a != by_index[pair['test_index']]['prompt_token_ids'] or len(a) != len(b)
                or a == b or any(type(t) is not int or t < 0 for t in a+b)):
            raise ValueError('Invalid frozen IT donor')
        counts[pair['pairing_kind']] += 1
        if 'rephrased' in pair['pairing_kind'] and (pair['test_index'], tuple(b)) not in keys:
            raise ValueError('Missing IT rewrite provenance')
    lookup = {tuple(p['student_prompt_ids']): p['teacher_prompt_ids'] for p in pairs}
    donors = {r['index']: lookup[tuple(r['prompt_token_ids'])] for r in rows}
    changed = sum(p['teacher_prompt_ids'] != donors[p['test_index']] for p in pairs)
    if (config['duplicate_test_prompt_rows'] != len(rows)-len(lookup)
            or config['rows_using_later_duplicate_donor'] != changed):
        raise ValueError('Historical duplicate-prompt policy differs')
    audit = dict(profile=spec['profile'], repo_id=spec['repo_id'], revision=spec['revision'],
        manifest_sha256=spec['manifest_sha256'], experiment=experiment, questions=len(rows),
        coverage=1.0, pairing_counts=dict(counts), duplicate_test_prompt_rows=len(rows)-len(lookup),
        rows_using_later_duplicate_donor=changed, donor_lookup_policy=config['donor_lookup_policy'],
        rows_sha256=sha(folder/'rows.json'), pairs_sha256=sha(folder/'pairs.json'),
        semantic_audit=spec['manifest']['semantic_audit'], scope=config['scope'])
    return rows, donors, config, audit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def validate(root):
    """Verify hashes, exact index coverage, donor lengths and supplement linkage."""
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['schema_version'] != 1 or manifest['kind'] != 'frozen_qwen3_it':
        raise ValueError('Unsupported IT bundle')
    for name, spec in manifest['files'].items():
        safe_path(name)
        verify(root / name, spec)
    supplements = [json.loads(s) for s in (root / 'supplements/rephrases.jsonl').read_text().splitlines()]
    if len(supplements) != manifest['supplement_records']:
        raise ValueError('Supplement count differs')
    supplement_keys = set()
    for r in supplements:
        ids = r['teacher_prompt_token_ids']
        if len(ids) != r['target_prompt_tokens']:
            raise ValueError('Supplement length differs')
        supplement_keys.add((r['test_index'], tuple(ids)))
    for job, spec in manifest['jobs'].items():
        if not re.fullmatch(r'\d{2}_[a-z0-9_]+', job):
            raise ValueError('Invalid job name')
        values = []
        for filename in ('rows.json', 'pairs.json', 'config.json'):
            name = f'inputs/{job}/{filename}'
            if name not in manifest['files']:
                raise ValueError('Unverified input: ' + name)
            values.append(json.loads((root / name).read_text()))
        rows, pairs, config = values
        indices = [r['index'] for r in rows]
        if (len(rows) != spec['questions'] or len(set(indices)) != len(rows)
                or len(pairs) != len(rows) or {p['test_index'] for p in pairs} != set(indices)
                or config['questions'] != len(rows) or config['samples'] != spec['samples']
                or config['experiment'] != int(job[:2])):
            raise ValueError('IT question coverage differs: ' + job)
        by_index = {r['index']: r for r in rows}
        for p in pairs:
            a, b = p['student_prompt_ids'], p['teacher_prompt_ids']
            if (a != by_index[p['test_index']]['prompt_token_ids'] or a == b
                    or len(a) != len(b) or not a
                    or any(type(t) is not int or t < 0 for t in a + b)):
                raise ValueError('IT prompt mismatch: ' + job)
            if 'rephrased' in p['pairing_kind'] and (p['test_index'], tuple(b)) not in supplement_keys:
                raise ValueError('Missing supplement provenance: ' + job)
    return manifest


def restore(bundle, output, experiments=None):
    """Restore exactly the evaluated rows/pairs/config; never overwrite differences."""
    bundle, output = Path(bundle), Path(output)
    manifest = validate(bundle)
    selected = set(experiments or [int(j[:2]) for j in manifest['jobs']])
    available = {int(j[:2]) for j in manifest['jobs']}
    if not selected or not selected <= available:
        raise ValueError('Requested experiment is absent from the IT bundle')
    restored = []
    for job in manifest['jobs']:
        if int(job[:2]) not in selected:
            continue
        for name in ('rows.json', 'pairs.json', 'config.json'):
            source = f'inputs/{job}/{name}'
            _install(output / job / name, manifest['files'][source], lambda s=source: bundle / s)
        restored.append(job)
    write_json(output / 'it_bundle_receipt.json', {
        'manifest_sha256': sha(bundle / 'manifest.json'), 'jobs': restored,
        'semantic_audit': manifest['semantic_audit']})
    _install(output / 'it_manifest.json', {'size': (bundle/'manifest.json').stat().st_size,
        'sha256': sha(bundle/'manifest.json')}, lambda: bundle/'manifest.json')
    return restored


def prepared_jobs(root, experiments):
    """Load downloaded inputs for the existing evaluation runners, without archives."""
    root = Path(root)
    manifest = json.loads((root/'it_manifest.json').read_text())
    receipt = json.loads((root/'it_bundle_receipt.json').read_text())
    if sha(root/'it_manifest.json') != receipt['manifest_sha256']:
        raise ValueError('Restored manifest changed')
    result = []
    for experiment in experiments:
        names = [n for n in manifest['jobs'] if int(n[:2]) == experiment]
        if len(names) != 1 or names[0] not in receipt['jobs']:
            raise ValueError('IT experiment was not restored: ' + str(experiment))
        job = names[0]
        for name in ('rows.json', 'pairs.json', 'config.json'):
            verify(root/job/name, manifest['files'][f'inputs/{job}/{name}'])
        result.append((root/job, json.loads((root/job/'config.json').read_text())))
    return result


def publish(bundle, repo_id, receipt):
    from huggingface_hub import HfApi, get_token, hf_hub_download
    bundle = Path(bundle)
    manifest = validate(bundle)
    receipt = Path(receipt)
    if receipt.exists():
        raise FileExistsError('Preserve existing publication receipt: ' + str(receipt))
    if not get_token():
        raise RuntimeError('HF write authentication missing. Run hf auth login in your terminal.')
    api = HfApi()
    # Existing dataset only; this command never creates a new public repository.
    info = api.repo_info(repo_id, repo_type='dataset')
    existing = api.list_repo_files(repo_id, repo_type='dataset', revision=info.sha)
    if any(name.startswith(PREFIX + '/') for name in existing):
        raise FileExistsError('Remote it/v1 already exists; inspect it before publishing a new version')
    commit = api.upload_folder(repo_id=repo_id, repo_type='dataset', folder_path=bundle,
        path_in_repo=PREFIX, parent_commit=info.sha,
        allow_patterns=['manifest.json', *manifest['files']],
        commit_message='Add frozen Qwen3 IT pairings and length-matched supplements')
    revision = commit.oid
    # Save the immutable revision immediately, even if subsequent network verification fails.
    result = dict(repo_id=repo_id, revision=revision, prefix=PREFIX,
                  manifest_sha256=sha(bundle / 'manifest.json'), remote_verified=False)
    write_json(receipt, result)
    for name, spec in {'manifest.json': {'size': (bundle/'manifest.json').stat().st_size,
                                         'sha256': result['manifest_sha256']}, **manifest['files']}.items():
        downloaded = hf_hub_download(repo_id, repo_type='dataset', revision=revision,
                                     filename=f'{PREFIX}/{name}')
        verify(downloaded, spec)
    result['remote_verified'] = True
    write_json(receipt, result)
    print(json.dumps(result, indent=2))


def download(repo_id, revision, expected_manifest_sha, output, experiments=None,
             local_files_only=False):
    from huggingface_hub import hf_hub_download
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Use an immutable 40-character HF commit, not main')
    if not re.fullmatch(r'[0-9a-f]{64}', expected_manifest_sha):
        raise ValueError('Supply the manifest SHA-256 from the publication receipt')
    def fetch(name):
        return Path(hf_hub_download(repo_id, repo_type='dataset', revision=revision,
            filename=f'{PREFIX}/{name}', local_files_only=local_files_only))
    remote = fetch('manifest.json')
    if sha(remote) != expected_manifest_sha:
        raise ValueError('IT manifest SHA-256 differs')
    manifest = json.loads(remote.read_text())
    with tempfile.TemporaryDirectory(prefix='platour-it-') as temporary:
        bundle = Path(temporary)
        (bundle / 'manifest.json').write_bytes(remote.read_bytes())
        for name, spec in manifest['files'].items():
            safe_path(name)
            _install(bundle / name, spec, lambda n=name: fetch(n))
        jobs = restore(bundle, output, experiments)
    write_json(Path(output) / 'it_hf_receipt.json', dict(repo_id=repo_id, revision=revision,
        prefix=PREFIX, manifest_sha256=expected_manifest_sha, jobs=jobs))
    print('Restored IT inputs:', ', '.join(jobs))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    v = sub.add_parser('verify'); v.add_argument('--bundle', type=Path, required=True)
    r = sub.add_parser('restore'); r.add_argument('--bundle', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--experiments', nargs='+', type=int)
    u = sub.add_parser('publish'); u.add_argument('--bundle', type=Path, required=True)
    u.add_argument('--repo-id', required=True); u.add_argument('--receipt', type=Path, required=True)
    d = sub.add_parser('download'); d.add_argument('--receipt', type=Path, required=True)
    d.add_argument('--output', type=Path, required=True)
    d.add_argument('--experiments', nargs='+', type=int)
    d.add_argument('--local-files-only', action='store_true')
    a = p.parse_args(argv)
    if a.action == 'verify':
        m = validate(a.bundle); print(f"Verified {len(m['jobs'])} jobs and {m['supplement_records']} supplement records")
    elif a.action == 'restore':
        print(restore(a.bundle, a.output, a.experiments))
    elif a.action == 'publish':
        publish(a.bundle, a.repo_id, a.receipt)
    else:
        receipt = json.loads(a.receipt.read_text())
        if receipt.get('prefix') != PREFIX:
            raise ValueError('Unexpected remote IT path')
        download(receipt['repo_id'], receipt['revision'], receipt['manifest_sha256'],
                 a.output, a.experiments, a.local_files_only)


if __name__ == '__main__':
    main()
