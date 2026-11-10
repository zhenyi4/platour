"""Prepare and publish fixed TRAIN/VAL splits for the six Qwen3 GRPO datasets."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import shutil

from platour_data import safe_path, verify

POLICY = 'published_train_holdout_v2'
REPO = 'zen-E/PLatour-Qwen3-Data'
ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        return dict(size=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def partition(rows, test, task, seed=42):
    """Materialize the split once for publication; never called by PACT evaluation."""
    count = len(test) if task == 'chemistry_cot' else 500
    if task not in ('chemistry_cot', 'gsm8k', 'math500') or not 0 < count < len(rows):
        raise ValueError('Invalid GRPO task or validation size')
    if any('original_train_index' in row for row in rows):
        raise ValueError('Input was already partitioned; do not split TRAIN again')
    if len({r['index'] for r in rows}) != len(rows):
        raise ValueError('Original TRAIN indices must be unique')
    selected = set(random.Random(seed).sample(range(len(rows)), count))
    def select(heldout):
        chosen = [r for i, r in enumerate(rows) if (i in selected) == heldout]
        return [dict(r, index=i, original_train_index=r['index']) for i, r in enumerate(chosen)]
    train, val = select(False), select(True)
    groups = [{tuple(r['prompt_token_ids']) for r in split} for split in (train, val, test)]
    if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('TRAIN/VAL/TEST prompts overlap')
    split = dict(policy=POLICY, seed=seed, original_train_count=len(rows),
        train_count=len(train), validation_count=len(val), test_count=len(test),
        train_source_indices=[r['original_train_index'] for r in train],
        validation_source_indices=[r['original_train_index'] for r in val],
        algorithm='python.random.Random.sample; retained source order; contiguous split indices',
        test_unchanged=True)
    return train, val, split


def prepare(output, checkout=ROOT):
    output, checkout = Path(output), Path(checkout)
    if output.exists():
        raise FileExistsError('Use a fresh prepared release directory')
    table = json.loads((checkout/'data/hf_manifest.json').read_text())
    if table['repositories']['qwen3']['repo_id'] != REPO:
        raise ValueError('Unexpected Qwen3 dataset repository')
    output.mkdir(parents=True)
    write_json(output/'registry-before.json', table)
    plan = dict(policy=POLICY, repo_id=REPO, source_revision=table['repositories']['qwen3']['revision'],
                seed=42, jobs={}, files={})
    for experiment in range(13, 19):
        names = [n for n in table['files'] if n.startswith(f'qwen3/{experiment:02d}_')]
        if len(names) != 3:
            raise ValueError('Expected exactly three original splits')
        folder = Path(names[0]).parent
        source = checkout/'data'/folder
        manifest = json.loads((source/'manifest.json').read_text())
        if manifest.get('validation_split', {}).get('policy') == POLICY:
            raise ValueError('This release already contains the published validation split')
        for name in names:
            verify(checkout/'data'/name, table['files'][name])
            dest = output/'source'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checkout/'data'/name, dest)
        shutil.copyfile(source/'manifest.json', output/'source'/folder/'manifest.json')
        rows, test = read_rows(source/'train.jsonl'), read_rows(source/'test.jsonl')
        train, validation, split = partition(rows, test, manifest['task'])
        split['source_train_sha256'] = table['files'][str(folder/'train.jsonl')]['sha256']
        split['source_revision'] = plan['source_revision']
        split['test_sha256'] = table['files'][str(folder/'test.jsonl')]['sha256']
        dest = output/'upload'/folder
        dest.mkdir(parents=True)
        for name, values in [('train', train), ('validation', validation)]:
            path = dest/(name+'.jsonl')
            path.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':'))+'\n' for r in values))
            manifest['counts'][name] = len(values)
            manifest['sha256'][name] = identity(path)['sha256']
        manifest['validation_split'] = dict(policy=POLICY, seed=42, source='original TRAIN only',
            source_train_sha256=split['source_train_sha256'], source_revision=plan['source_revision'],
            original_train_count=len(rows), validation_count=len(validation), test_unchanged=True)
        write_json(dest/'manifest.json', manifest)
        write_json(dest/'split.json', split)
        plan['jobs'][str(experiment)] = dict(folder=folder.as_posix(), task=manifest['task'], counts=manifest['counts'])
    note = '# Fixed Qwen3 GRPO validation splits\n\n'
    note += 'Experiments 13–18: validation is drawn once from original TRAIN with seed 42.\n'
    note += 'Chemistry: TRAIN 1680 / VAL 210 / TEST 210.\nGSM8K: TRAIN 6973 / VAL 500 / TEST 1319.\nMATH500: TRAIN 7000 / VAL 500 / TEST 500.\n\n'
    note += 'TEST files, SFT data, and archived it/v1 pairings are unchanged. Each split.json records source indices and checksums. Consumers read TRAIN and VAL directly; no second split is required. The it/v1 donors retain their original training-pool provenance. Historical pretrained weights are unchanged.\n'
    (output/'upload/GRPO_SPLITS.md').write_text(note)
    for path in (output/'upload').rglob('*'):
        if path.is_file():
            plan['files'][path.relative_to(output/'upload').as_posix()] = identity(path)
    write_json(output/'plan.json', plan)
    validate(output)
    return plan


def validate(bundle):
    bundle = Path(bundle)
    plan = json.loads((bundle/'plan.json').read_text())
    before = json.loads((bundle/'registry-before.json').read_text())
    if plan['policy'] != POLICY or plan['repo_id'] != REPO or set(plan['jobs']) != set(map(str, range(13, 19))):
        raise ValueError('Invalid prepared GRPO release')
    expected = {'GRPO_SPLITS.md'}
    for job in plan['jobs'].values():
        folder = safe_path(job['folder'])
        expected.update(str(folder/name) for name in ('train.jsonl', 'validation.jsonl', 'manifest.json', 'split.json'))
        source = bundle/'source'/folder
        for part in ('train', 'validation', 'test'):
            verify(source/(part+'.jsonl'), before['files'][str(folder/(part+'.jsonl'))])
        original, test = read_rows(source/'train.jsonl'), read_rows(source/'test.jsonl')
        train, val, split = partition(original, test, job['task'], plan['seed'])
        target = bundle/'upload'/folder
        if read_rows(target/'train.jsonl') != train or read_rows(target/'validation.jsonl') != val:
            raise ValueError('Prepared rows differ from the recorded source split')
        actual_split = json.loads((target/'split.json').read_text())
        if any(actual_split[k] != v for k, v in split.items()):
            raise ValueError('Prepared split index record differs')
        manifest = json.loads((target/'manifest.json').read_text())
        counts = dict(train=len(train), validation=len(val), test=len(test))
        if manifest['counts'] != counts or job['counts'] != counts:
            raise ValueError('Prepared split counts differ')
        for part in counts:
            path = source/'test.jsonl' if part == 'test' else target/(part+'.jsonl')
            if identity(path)['sha256'] != manifest['sha256'][part]:
                raise ValueError('Prepared split checksum differs')
    if set(plan['files']) != expected:
        raise ValueError('Prepared upload scope differs')
    for name, spec in plan['files'].items():
        verify(bundle/'upload'/safe_path(name), spec)
    return plan, before


def activate(bundle, checkout=ROOT):
    """Install only a remotely verified release; retain old registry until publication succeeds."""
    bundle, checkout = Path(bundle), Path(checkout)
    plan, before = validate(bundle)
    receipt = json.loads((bundle/'hf-receipt.json').read_text())
    if (receipt.get('remote_verified') is not True or receipt.get('repo_id') != REPO or
            not re.fullmatch(r'[0-9a-f]{40}', receipt.get('revision', '')) or
            receipt.get('plan_sha256') != identity(bundle/'plan.json')['sha256']):
        raise ValueError('A verified immutable HF publication receipt is required')
    after = copy.deepcopy(before)
    after['repositories']['qwen3']['revision'] = receipt['revision']
    after['repositories']['qwen3']['split_policy'] = POLICY
    for name, spec in plan['files'].items():
        if name == 'GRPO_SPLITS.md':
            continue
        if name.endswith('.jsonl'):
            after['files'][name].update(spec)
        else:
            after['protocol_files'][name] = spec
    current = json.loads((checkout/'data/hf_manifest.json').read_text())
    if current not in (before, after):
        raise ValueError('Local registry changed since preparation; do not overwrite it')
    # Check every destination before writing. Never overwrite caller-modified data.
    destinations = []
    for name, spec in plan['files'].items():
        if name == 'GRPO_SPLITS.md':
            continue
        dest = checkout/'data'/name
        old = before['files'].get(name, before['protocol_files'].get(name))
        if dest.exists() and identity(dest) not in (spec, {k: old[k] for k in ('size','sha256')} if old else None):
            raise ValueError('Local input changed: '+name)
        destinations.append((bundle/'upload'/name, dest))
    for source, dest in destinations:
        dest.parent.mkdir(parents=True, exist_ok=True)
        pending = dest.with_suffix(dest.suffix+'.grpo-v2-pending')
        shutil.copyfile(source, pending)
        pending.replace(dest)
    receipt = dict(receipt, payload_bytes=sum(v['size'] for v in after['files'].values()),
                   payload_files=len(after['files']), protocol_files=len(after['protocol_files']))
    write_json(checkout/'data/hf_manifest.json', after)
    write_json(checkout/'docs/qwen3_grpo_split_release.json', receipt)
    return after


def publish(bundle, checkout=ROOT):
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    bundle = Path(bundle)
    plan, before = validate(bundle)
    receipt_path = bundle/'hf-receipt.json'
    api = HfApi()
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('plan_sha256') != identity(bundle/'plan.json')['sha256'] or receipt.get('repo_id') != REPO:
            raise ValueError('Existing publication receipt belongs to another plan')
        revision = receipt['revision']
    else:
        parent = api.repo_info(REPO, repo_type='dataset', timeout=15).sha
        # Guard all published Qwen3 payloads against concurrent dataset edits.
        for name, spec in before['files'].items():
            if spec['group'] == 'qwen3':
                path = hf_hub_download(REPO, name, repo_type='dataset', revision=parent)
                verify(path, spec)
        result = api.create_commit(repo_id=REPO, repo_type='dataset', parent_commit=parent,
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(bundle/'upload'/name))
                        for name in sorted(plan['files'])],
            commit_message='Fix Qwen3 GRPO validation: Chemistry 210, GSM8K/MATH 500 from TRAIN')
        revision = result.oid
        receipt = dict(repo_id=REPO, revision=revision, parent_revision=parent,
            policy=POLICY, plan_sha256=identity(bundle/'plan.json')['sha256'],
            remote_verified=False, jobs=plan['jobs'])
        write_json(receipt_path, receipt)
    expected = {name: spec for name, spec in before['files'].items() if spec['group'] == 'qwen3'}
    expected.update(plan['files'])
    for name, spec in expected.items():
        verify(hf_hub_download(REPO, name, repo_type='dataset', revision=revision), spec)
    receipt['remote_verified'] = True
    write_json(receipt_path, receipt)
    activate(bundle, checkout)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--checkout', type=Path, default=ROOT)
    for name in ('verify', 'publish', 'activate'):
        command = sub.add_parser(name); command.add_argument('--bundle', type=Path, required=True)
        if name != 'verify': command.add_argument('--checkout', type=Path, default=ROOT)
    a = p.parse_args(argv)
    if a.action == 'prepare': result = prepare(a.output, a.checkout)
    elif a.action == 'verify': result, _ = validate(a.bundle)
    elif a.action == 'publish': result = publish(a.bundle, a.checkout)
    else: result = activate(a.bundle, a.checkout)
    print(json.dumps(result.get('jobs', result.get('repositories')), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
