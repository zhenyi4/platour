"""Rebuild pinned DeepCoder and LCB v6 judge cases without executing code."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import pickletools
import re
import shutil
import tempfile
import zlib

from .checkpoints import atomic_json, sha256
from .cli import DATA, signed_rows
from .data import expected_cases, rebuild

SAFE = re.compile(r'[a-z0-9_]+\.json\Z')
PINNED = Path(__file__).with_name('lcb_v6_cases_sha256.json')
MAX_PICKLE_BYTES = 64 * 1024 * 1024


def _original_test_rows(value):
    """Read the revision-pinned LCB JSON-string pickle without executing pickle code."""
    try:
        rows = json.loads(value)
    except json.JSONDecodeError:
        binary = base64.b64decode(value, validate=True)
        decompressor = zlib.decompressobj()
        serialized = decompressor.decompress(binary, MAX_PICKLE_BYTES + 1)
        if (len(serialized) > MAX_PICKLE_BYTES or not decompressor.eof or
                decompressor.unconsumed_tail or decompressor.unused_data):
            raise ValueError('Unsafe or oversized original LCB private tests')
        operations = list(pickletools.genops(serialized))
        names = [op.name for op, _, _ in operations]
        if (names not in (['PROTO', 'FRAME', 'BINUNICODE', 'MEMOIZE', 'STOP'],
                          ['PROTO', 'BINUNICODE', 'MEMOIZE', 'STOP']) or
                operations[-1][2] != len(serialized) - 1):
            raise ValueError('LCB private tests are not a plain JSON string pickle')
        rows = json.loads(next(argument for op, argument, _ in operations
                               if op.name == 'BINUNICODE'))
    if (not isinstance(rows, list) or any(not isinstance(test, dict) or
            type(test.get('input')) is not str or type(test.get('output')) is not str or
            test.get('testtype') not in ('stdin', 'functional') for test in rows)):
        raise ValueError('Original LCB test rows are invalid')
    return rows


def _public_v6_case(row):
    """Reproduce the original CodeGenerationProblem.get_evaluation_sample byte-for-byte."""
    if not isinstance(row, dict):
        raise ValueError('Original LCB question is not a JSON object')
    metadata = json.loads(row['metadata'])
    if not isinstance(metadata, dict):
        raise ValueError('Original LCB problem metadata differs')
    tests = _original_test_rows(row['public_test_cases']) + _original_test_rows(
        row['private_test_cases'])
    if not tests:
        raise ValueError('Original LCB question has no test cases')
    sample = {'input_output': json.dumps({'inputs': [test['input'] for test in tests],
                                          'outputs': [test['output'] for test in tests],
                                          'fn_name': metadata.get('func_name')})}
    return json.dumps(sample).encode()


def materialize_public_v6(raw_test6, destination):
    """Derive exactly the 175 pinned originals from the public revision's test6.jsonl."""
    raw_test6, destination = Path(raw_test6), Path(destination)
    if destination.exists():
        raise FileExistsError('LCB public case extraction needs a fresh private root')
    pinned = json.loads(PINNED.read_text())
    if (not raw_test6.is_file() or raw_test6.is_symlink() or
            sha256(raw_test6) != pinned['test6_sha256']):
        raise RuntimeError('Pinned public LiveCodeBench test6.jsonl SHA differs')
    expected = pinned['case_sha256']
    destination.mkdir(mode=0o700)
    seen = set()
    with raw_test6.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            name = f"{row['platform']}__{row['question_id']}.json"
            if not SAFE.fullmatch(name) or name not in expected or name in seen:
                raise ValueError('Unexpected or duplicate public LCB v6 problem key')
            sample = _public_v6_case(row)
            if hashlib.sha256(sample).hexdigest() != expected[name]:
                raise RuntimeError('Original public LCB v6 per-question case SHA differs')
            (destination / name).write_bytes(sample)
            seen.add(name)
    if seen != set(expected) or len(seen) != 175:
        raise RuntimeError('Pinned public LCB v6 incremental coverage differs')
    atomic_json(destination / 'complete.json', {
        'complete': True, 'cases': len(seen), 'source_sha256': pinned['test6_sha256'],
        'livecodebench_commit': pinned['livecodebench_commit'],
        'per_question_sha256_verified': True})
    return len(seen)


def prepare(data_root, lcb_cases, checker, output, *, deepcoder_source_root=None,
            cache_dir=None, local_files_only=False):
    """Never overwrite prior results; sign only after all original cases match SHA."""
    data_root, lcb_cases, checker, output = map(Path,
        (data_root, lcb_cases, checker, output))
    if output.exists():
        raise FileExistsError('Use a fresh private code-judge case root')
    if lcb_cases.is_symlink() or checker.is_symlink():
        raise ValueError('Authorized judge sources may not redirect through symlinks')
    pinned = json.loads(PINNED.read_text())
    official = pinned['case_sha256']
    extractor = checker.parent / 'utils/extraction_utils.py'
    if (len(official) != 175 or not (lcb_cases.is_dir() and checker.is_dir()) or
            (checker / 'testing_util.py').is_symlink() or extractor.is_symlink() or
            not extractor.is_file() or
            sha256(checker / 'testing_util.py') != pinned['checker_sha256'] or
            sha256(extractor) != pinned['extractor_sha256']):
        raise RuntimeError('Official checker, code extractor or LCB v6 manifest differs')
    from platour_data import ensure_qwen35_data
    ensure_qwen35_data(26, data_root, 'judge', cache_dir=cache_dir,
                       local_files_only=local_files_only)
    manifest = json.loads((data_root / 'manifest.json').read_text())
    data = data_root / 'deepcoder'
    wanted = expected_cases(data_root)
    test_file = data / 'lcb.jsonl'
    signature = manifest['benchmark']['lcb']
    if sha256(test_file) != signature['sha256']:
        raise RuntimeError('Original LCB v6 test prompt SHA differs')
    names = {row['judge']['key'] + '.json' for row in signed_rows(
        test_file, expected=signature['rows'])}
    if names != set(official) or names & set(wanted):
        raise RuntimeError('LCB v6 cases are incomplete or overlap with TRAIN/VAL')
    for name in names:
        source = lcb_cases / name
        if (source.is_symlink() or not source.is_file() or
                sha256(source) != official[name]):
            raise RuntimeError('Authorized original LCB v6 test case SHA differs')
    output.mkdir(mode=0o700, parents=True)
    cases = output / 'cases'
    rebuilt = rebuild(data_root, cases, source_root=deepcoder_source_root,
                      cache_dir=cache_dir, local_files_only=local_files_only)
    if rebuilt['cases'] != len(wanted):
        raise RuntimeError('Original TRAIN/VAL case coverage is incomplete')
    for name in names:
        shutil.copyfile(lcb_cases / name, cases / name)
        if sha256(cases / name) != official[name]:
            raise RuntimeError('Copied official LCB v6 case differs')
    proof = {'complete': True, 'training_validation_cases': rebuilt['cases'],
             'lcb_v6_cases': len(names), 'per_question_sha256_verified': True,
             'deepcoder_source': rebuilt, 'checker_sha256': pinned['checker_sha256'],
             'extractor_sha256': pinned['extractor_sha256'],
             'lcb_case_sha256': official, 'docker_image_verified': False,
             'docker_execution_performed': False}
    atomic_json(output / 'complete.json', proof)
    return proof


def prepare_from_public_v6(data_root, raw_test6, checker, output, **kwargs):
    """Use only pinned public data; never require a private redistribution."""
    output, checker, raw_test6 = Path(output), Path(checker), Path(raw_test6).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh original judge output root')
    pinned = json.loads(PINNED.read_text())
    if (checker.is_symlink() or not (checker / 'testing_util.py').is_file() or
            sha256(checker / 'testing_util.py') != pinned['checker_sha256']):
        raise RuntimeError('Original LiveCodeBench checker SHA differs')
    if sha256(raw_test6) != pinned['test6_sha256']:
        raise RuntimeError('Original public release_v6 test6.jsonl SHA differs')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='platour-lcb-v6-', dir=output.parent) as scratch:
        materialize_public_v6(raw_test6, scratch + '/cases')
        return prepare(data_root, scratch + '/cases', checker, output, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--lcb-cases', type=Path,
                        help='Previously verified 175 original LCB v6 JSON files')
    source.add_argument('--lcb-test6', type=Path,
                        help='Pinned public code_generation_lite test6.jsonl; safely materialize all 175 cases')
    source.add_argument('--fetch-public-lcb-v6', action='store_true',
                        help='Explicitly fetch the pinned public test6.jsonl from Hugging Face')
    parser.add_argument('--checker', type=Path, required=True,
                        help='Pinned original lcb_runner/evaluation directory')
    parser.add_argument('--output', type=Path, required=True,
                        help='New private destination; includes cases/ and complete.json')
    parser.add_argument('--deepcoder-source-root', type=Path, help='Offline pinned DeepCoder snapshot root')
    parser.add_argument('--cache-dir', type=Path, help='Hugging Face download cache')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    options = dict(deepcoder_source_root=args.deepcoder_source_root,
                   cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    if args.lcb_cases is not None:
        result = prepare(args.data_root, args.lcb_cases, args.checker, args.output, **options)
    else:
        raw = args.lcb_test6
        if args.fetch_public_lcb_v6:
            from huggingface_hub import hf_hub_download
            pinned = json.loads(PINNED.read_text())
            raw = Path(hf_hub_download(repo_id=pinned['dataset_repo'],
                filename='test6.jsonl', revision=pinned['dataset_revision'],
                repo_type='dataset', cache_dir=args.cache_dir,
                local_files_only=args.local_files_only)).resolve()
        result = prepare_from_public_v6(args.data_root, raw, args.checker, args.output, **options)
    print(json.dumps({key: result[key] for key in
                      ('complete', 'training_validation_cases', 'lcb_v6_cases')}))


if __name__ == '__main__':
    main()
