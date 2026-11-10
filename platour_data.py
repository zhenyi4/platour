"""Restore frozen experiment inputs from checksum-pinned public HF datasets.

This module deliberately has no torch/model imports. Run from an editable checkout.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile


DATA = Path(__file__).resolve().parent / 'data'
REGISTRY = DATA / 'hf_manifest.json'
LFS_HEADER = b'version https://git-lfs.github.com/spec/v1\n'


def registry():
    value = json.loads(REGISTRY.read_text())
    if value.get('schema_version') != 1 or set(value['repositories']) != {
            'qwen3', 'dapo', 'deepcoder'}:
        raise ValueError('Invalid frozen dataset registry')
    for spec in value['repositories'].values():
        if (not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', spec['repo_id']) or
                not re.fullmatch(r'[0-9a-f]{40}', spec['revision'])):
            raise ValueError('Dataset repository must use a pinned commit')
    for section in ('files', 'protocol_files'):
        for name, spec in value[section].items():
            safe_path(name)
            if (type(spec['size']) is not int or spec['size'] < 1 or
                    not re.fullmatch(r'[0-9a-f]{64}', spec['sha256'])):
                raise ValueError('Invalid dataset size or SHA-256: ' + name)
            if section == 'files':
                group = spec['group']
                expected = name if group == 'qwen3' else name.removeprefix('qwen35/')
                prefix = 'qwen3/' if group == 'qwen3' else 'qwen35/' + group + '/'
                if (group not in value['repositories'] or not name.startswith(prefix) or
                        spec['filename'] != expected):
                    raise ValueError('Invalid dataset repository mapping: ' + name)
                safe_path(spec['filename'])
    return value


def safe_path(name):
    path = PurePosixPath(name)
    if (not isinstance(name, str) or path.is_absolute() or '..' in path.parts or
            not path.parts or path.as_posix() != name or '\\' in name):
        raise ValueError('Invalid dataset path: ' + str(name))
    return path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(path, spec):
    path = Path(path)
    if (not path.is_file() or path.stat().st_size != spec['size'] or
            digest(path) != spec['sha256']):
        raise ValueError('Frozen data size/SHA-256 differs: ' + str(path))


def is_lfs_pointer(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 1024:
        return False
    with path.open('rb') as stream:
        return stream.read(len(LFS_HEADER)) == LFS_HEADER


def _existing(path, spec):
    if not path.exists() and not path.is_symlink():
        return False
    if is_lfs_pointer(path):
        expected = (LFS_HEADER.decode() + 'oid sha256:' + spec['sha256'] +
                    '\nsize ' + str(spec['size']) + '\n')
        if path.read_text() != expected:
            raise ValueError('LFS pointer identity differs: ' + str(path))
        return False
    verify(path, spec)
    return True


def _install(path, spec, source):
    """Serialize ranks and publish only a complete verified copy."""
    path = Path(path)
    if _existing(path, spec):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    locks = path.parent / '.cache' / 'platour-data'
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / (path.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _existing(path, spec):
            return path
        downloaded = Path(source())
        verify(downloaded, spec)
        fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.',
                                         suffix='.pending', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as target, downloaded.open('rb') as origin:
                shutil.copyfileobj(origin, target)
            verify(temporary, spec)
            # A caller changing a target while we download must not lose its data.
            if _existing(path, spec):
                return path
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return path


def ensure_file(name, destination=None, *, cache_dir=None, local_files_only=False):
    """Reuse valid local data; hydrate absent files or matching old LFS pointers."""
    table = registry()
    spec = table['files'][name]
    repository = table['repositories'][spec['group']]

    def source():
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repository['repo_id'], repo_type='dataset',
            revision=repository['revision'], filename=spec['filename'],
            cache_dir=cache_dir, local_files_only=local_files_only)

    return _install(DATA / name if destination is None else destination, spec, source)


def experiment_files(experiment, purpose='all'):
    if type(experiment) is not int or not 1 <= experiment <= 30:
        raise ValueError('Experiment must be between 1 and 30')
    if purpose not in ('all', 'train', 'evaluate', 'judge'):
        raise ValueError('Unknown dataset purpose')
    files = registry()['files']
    if experiment <= 18:
        names = [name for name in files if name.startswith(f'qwen3/{experiment:02d}_')]
        if purpose == 'evaluate':
            names = [name for name in names if name.endswith('/test.jsonl')]
        elif purpose == 'train':
            names = [name for name in names if not name.endswith('/test.jsonl')]
        return sorted(names)
    group = 'dapo' if experiment <= 24 else 'deepcoder'
    names = [name for name, spec in files.items() if spec['group'] == group]
    if purpose == 'train':
        names = [name for name in names if Path(name).stem in ('train', 'validation')]
    elif purpose == 'evaluate':
        stems = ('aime', 'hmmt', 'imo') if group == 'dapo' else (
            ('lcb', 'humaneval_template_adapted', 'mbpp_template_adapted')
            if experiment == 28 else ('lcb', 'humaneval', 'mbpp'))
        names = [name for name in names if Path(name).stem in stems]
    elif purpose == 'judge':
        names = [name for name in names if Path(name).stem in ('train', 'validation', 'lcb')]
    return sorted(names)


def _protocol(name, destination):
    spec = registry()['protocol_files'][name]
    return _install(destination, spec, lambda: DATA / name)


def ensure_qwen35_data(experiment, data_root, purpose, **options):
    """Support published inputs at any root without changing custom manifests."""
    root = Path(data_root)
    spec = registry()['protocol_files']['qwen35/manifest.json']
    manifest = root / 'manifest.json'
    if manifest.exists() and (manifest.stat().st_size != spec['size'] or
                              digest(manifest) != spec['sha256']):
        if root.resolve() == (DATA / 'qwen35').resolve():
            raise ValueError('Frozen Qwen3.5 manifest differs')
        return  # Caller-owned custom data retains its existing audit path.
    _protocol('qwen35/manifest.json', manifest)
    if experiment <= 24:
        _protocol('qwen35/dapo/clean80.json', root / 'dapo/clean80.json')
    return [ensure_file(name, root / name.removeprefix('qwen35/'), **options)
            for name in experiment_files(experiment, purpose)]


def download(experiments, output=DATA, **options):
    """Restore payloads and Git protocol files in the usual data/ layout."""
    table = registry()
    names = sorted({name for experiment in experiments for name in experiment_files(experiment)})
    parents = {str(PurePosixPath(name).parent) for name in names}
    for name in table['protocol_files']:
        if str(PurePosixPath(name).parent) in parents or (
                name == 'qwen35/manifest.json' and any(p.startswith('qwen35/') for p in parents)):
            _protocol(name, Path(output) / name)
    return [ensure_file(name, Path(output) / name, **options) for name in names]


def release_inventory():
    """Release exporters include HF payloads even though Git no longer tracks them."""
    table = registry()
    return sorted('data/' + name for section in ('files', 'protocol_files')
                  for name in table[section])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Download checksum-pinned public experiment data')
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--experiment', type=int, choices=range(1, 31))
    choice.add_argument('--all', action='store_true')
    parser.add_argument('--output', type=Path, default=DATA, help='Root of the restored data layout')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--include-it', action='store_true',
                        help='Also restore supported frozen Qwen3 IT pairings and supplements')
    args = parser.parse_args(argv)
    it_experiments = []
    if args.include_it:
        from platour_it import supported_experiments, ensure_inputs
        it_experiments = supported_experiments() if args.all else [args.experiment]
        if not set(it_experiments) <= set(supported_experiments()):
            parser.error('--include-it supports only Qwen3 experiments 07–12 and 16–18')
    paths = download(range(1, 31) if args.all else [args.experiment], args.output,
                     cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    print(f'Verified {len(paths)} data files in {args.output}')
    if args.include_it:
        restored = ensure_inputs(it_experiments, args.output/'it/qwen3/archived-it-v1',
            cache_dir=args.cache_dir, local_files_only=args.local_files_only)
        print(f'Verified {len(restored)} frozen IT configurations (archived-it-v1)')


if __name__ == '__main__':
    main()
