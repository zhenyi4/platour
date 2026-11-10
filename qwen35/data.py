"""Rebuild byte-identical DeepCoder TRAIN/VAL judge cases from pinned public data."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .checkpoints import sha256

UPSTREAM = 'agentica-org/DeepCoder-Preview-Dataset'
REVISION = '177913a7bd43791646ef6a43645caa3c871ab3db'
SPLIT_COUNTS = {'train': 18873, 'validation': 201}
CASE_NAME = re.compile(r'deepcoder__(taco|primeintellect|lcbv5)__(\d{6})\.json\Z')
SOURCE_NAME = re.compile(r'(taco|primeintellect|lcbv5)/train-\d{5}-of-\d{5}\.parquet\Z')


def expected_records(data_root, *, cache_dir=None, local_files_only=False):
    """The frozen question selection and per-case hashes remain part of the release."""
    root = Path(data_root)
    from platour_data import ensure_qwen35_data
    ensure_qwen35_data(26, root, 'train', cache_dir=cache_dir,
                       local_files_only=local_files_only)
    metadata = json.loads((root / 'manifest.json').read_text())
    expected = {}
    for split, count in SPLIT_COUNTS.items():
        source = root / 'deepcoder' / (split + '.jsonl')
        if sha256(source) != metadata['deepcoder'][split]['sha256']:
            raise ValueError('Original DeepCoder prompt SHA differs: ' + split)
        rows = 0
        with source.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                judge = row['judge']
                name = judge['key'] + '.json'
                if (row['index'] != rows or not CASE_NAME.fullmatch(name) or
                        name in expected or judge['kind'] != 'deepcoder' or
                        not re.fullmatch(r'[0-9a-f]{64}', judge['sample_sha256']) or
                        type(judge['tests']) is not int or judge['tests'] < 1):
                    raise ValueError('Unexpected/duplicate original unit-test identity')
                expected[name] = judge
                rows += 1
        if rows != count:
            raise ValueError('The original DeepCoder question count differs')
    return expected


def expected_cases(data_root):
    return {name: spec['sample_sha256']
            for name, spec in expected_records(data_root).items()}


def source_manifest(data_root):
    metadata = json.loads((Path(data_root) / 'manifest.json').read_text())['samples']
    if metadata['upstream'] != UPSTREAM or metadata['revision'] != REVISION:
        raise ValueError('DeepCoder public source revision differs')
    files = metadata['files']
    if (not files or any(not SOURCE_NAME.fullmatch(name) or
            type(spec['size']) is not int or spec['size'] < 1 or
            type(spec['rows']) is not int or spec['rows'] < 1 or
            not re.fullmatch(r'[0-9a-f]{64}', spec['sha256'])
            for name, spec in files.items()) or
            {name.split('/')[0] for name in files} != {'taco', 'primeintellect', 'lcbv5'}):
        raise ValueError('Invalid pinned DeepCoder source file manifest')
    return files


def case_bytes(source, row):
    """Serialize data only; never execute upstream tests or solutions."""
    tests = json.loads(row['tests'])
    if isinstance(tests, list):
        sample = {'inputs': [test['input'] for test in tests],
                  'outputs': [test['output'] for test in tests]}
        if source == 'primeintellect' and tests and tests[0].get('type') == 'function_call':
            function = tests[0]['fn_name']
            if any(test.get('fn_name') != function for test in tests):
                raise ValueError('Inconsistent upstream function names')
            sample = {
                'inputs': ['\n'.join(json.dumps(arg, ensure_ascii=False)
                                     for arg in test['input']) for test in tests],
                'outputs': [json.dumps(test['output'][0], ensure_ascii=False) for test in tests],
                'fn_name': function}
        elif (row.get('metadata') or {}).get('func_name'):
            sample['fn_name'] = row['metadata']['func_name']
    elif isinstance(tests, dict):
        sample = {part: ['\n'.join(map(str, value)) if isinstance(value, list) else value
                         for value in tests[part]] for part in ('inputs', 'outputs')}
        if tests.get('fn_name'):
            sample = {
                'inputs': ['\n'.join(json.dumps(arg, ensure_ascii=False) for arg in args)
                           for args in tests['inputs']],
                'outputs': [json.dumps(value[0], ensure_ascii=False) for value in tests['outputs']],
                'fn_name': tests['fn_name']}
    else:
        raise ValueError('Unsupported upstream DeepCoder tests format')
    if (len(sample['inputs']) != len(sample['outputs']) or not sample['inputs'] or
            any(not isinstance(value, str) for part in ('inputs', 'outputs')
                for value in sample[part])):
        raise ValueError('Malformed upstream DeepCoder checker inputs/outputs')
    encoded = json.dumps({'input_output': json.dumps(sample, ensure_ascii=False)},
                         ensure_ascii=False) + '\n'
    return encoded.encode('utf-8'), len(sample['inputs'])


def verify_materialized(data_root, case_root):
    expected = expected_cases(data_root)
    root = Path(case_root)
    for name, digest in expected.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise RuntimeError('Original per-question TRAIN/VAL case SHA differs: ' + name)
    return len(expected)


def rebuild(data_root, output, *, source_root=None, cache_dir=None, local_files_only=False):
    """Fetch pinned parquet files (or use an offline snapshot), then check every case."""
    import pyarrow.parquet as pq

    output = Path(output)
    if output.exists():
        raise FileExistsError('Reconstruction requires a fresh empty destination')
    expected = expected_records(data_root, cache_dir=cache_dir, local_files_only=local_files_only)
    pinned = source_manifest(data_root)
    sources = {}
    for name, spec in sorted(pinned.items()):
        if source_root is None:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(repo_id=UPSTREAM, repo_type='dataset',
                revision=REVISION, filename=name, cache_dir=cache_dir,
                local_files_only=local_files_only))
        else:
            root = Path(source_root).resolve()
            path = root / name
        if not path.is_file() or path.stat().st_size != spec['size'] or sha256(path) != spec['sha256']:
            raise ValueError('Pinned upstream parquet size/SHA differs: ' + name)
        sources[name] = path
    output.mkdir(parents=True)
    offsets = Counter()
    seen = set()
    counts = Counter()
    for name, path in sources.items():
        source = name.split('/')[0]
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != pinned[name]['rows']:
            raise ValueError('Pinned upstream parquet row count differs: ' + name)
        columns = ['tests'] + (['metadata'] if source == 'lcbv5' else [])
        for batch in parquet.iter_batches(batch_size=2, columns=columns, use_threads=False):
            for row in batch.to_pylist():
                filename = f'deepcoder__{source}__{offsets[source]:06d}.json'
                offsets[source] += 1
                if filename not in expected:
                    continue
                payload, tests = case_bytes(source, row)
                spec = expected[filename]
                if (hashlib.sha256(payload).hexdigest() != spec['sample_sha256'] or
                        tests != spec['tests'] or filename in seen):
                    raise RuntimeError('Reconstructed original case SHA/count differs: ' + filename)
                pending = output / (filename + '.pending')
                pending.write_bytes(payload)
                pending.replace(output / filename)
                seen.add(filename)
                counts[source] += 1
        print(f'Rebuilt {len(seen)}/{len(expected)} cases ({name})', flush=True)
    if seen != set(expected):
        raise RuntimeError('Missing referenced training/validation unit tests')
    return {'repo': UPSTREAM, 'revision': REVISION, 'cases': len(seen),
            'per_question_sha256_verified': True, 'by_source': dict(counts),
            'source_files_sha256': {name: spec['sha256'] for name, spec in pinned.items()}}


def materialize(data_root, output, **kwargs):
    return rebuild(data_root, output, **kwargs)['cases']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[1] / 'data/qwen35')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, help='Offline snapshot root containing taco/, primeintellect/ and lcbv5/')
    parser.add_argument('--cache-dir', type=Path, help='Hugging Face download cache')
    parser.add_argument('--local-files-only', action='store_true', help='Use only previously cached upstream files')
    args = parser.parse_args()
    count = materialize(args.data_root, args.output, source_root=args.source_root,
                        cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    print(f'Verified and rebuilt {count} original train/validation unit tests')


if __name__ == '__main__':
    main()
