"""Pinned public DeepCoder reconstruction, corruption checks and judge integration."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from qwen35 import data


def encoded(sample):
    return (json.dumps({'input_output': json.dumps(sample, ensure_ascii=False)},
                       ensure_ascii=False) + '\n').encode()


class DeepCoderDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = self.root / 'upstream'
        self.inputs = self.root / 'data'
        (self.inputs / 'deepcoder').mkdir(parents=True)
        self.counts = patch.object(data, 'SPLIT_COUNTS', {'train': 3, 'validation': 1})
        self.counts.start()
        self.addCleanup(self.counts.stop)
        fixtures = [
            ('taco', {'tests': json.dumps({'inputs': [['héllo', '']], 'outputs': [['世界', '']]})},
             {'inputs': ['héllo\n'], 'outputs': ['世界\n']}),
            ('taco', {'tests': json.dumps({'fn_name': 'f', 'inputs': [[[1, 2], 'é']], 'outputs': [[True]]})},
             {'inputs': ['[1, 2]\n"é"'], 'outputs': ['true'], 'fn_name': 'f'}),
            ('primeintellect', {'tests': json.dumps([{'type': 'function_call', 'fn_name': 'g',
                'input': [[1], '中'], 'output': [['é']]}])},
             {'inputs': ['[1]\n"中"'], 'outputs': ['["é"]'], 'fn_name': 'g'}),
            ('lcbv5', {'tests': json.dumps([{'input': '1\n', 'output': '2\n', 'testtype': 'functional'}]),
                       'metadata': {'func_name': 'solve'}},
             {'inputs': ['1\n'], 'outputs': ['2\n'], 'fn_name': 'solve'})]
        rows = {'taco': [], 'primeintellect': [], 'lcbv5': []}
        self.payloads = {}
        prompts = []
        for source, original, sample in fixtures:
            index = len(rows[source])
            rows[source].append(original)
            name = f'deepcoder__{source}__{index:06d}'
            payload = encoded(sample)
            self.payloads[name + '.json'] = payload
            prompts.append({'index': 0, 'prompt_token_ids': [1, 2], 'gold': '',
                'judge': {'kind': 'deepcoder', 'key': name, 'tests': 1,
                          'sample_sha256': hashlib.sha256(payload).hexdigest()}})
            self.assertEqual(data.case_bytes(source, original), (payload, 1))
        self.manifest = {'deepcoder': {}, 'samples': {'upstream': data.UPSTREAM,
            'revision': data.REVISION, 'files': {}}}
        for source, records in rows.items():
            path = self.sources / source / 'train-00000-of-00001.parquet'
            path.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(records), path)
            self.manifest['samples']['files'][path.relative_to(self.sources).as_posix()] = {
                'size': path.stat().st_size, 'sha256': data.sha256(path), 'rows': len(records)}
        for split, records in [('train', prompts[:3]), ('validation', prompts[3:])]:
            for i, row in enumerate(records):
                row['index'] = i
            path = self.inputs / 'deepcoder' / (split + '.jsonl')
            path.write_text(''.join(json.dumps(row) + '\n' for row in records))
            self.manifest['deepcoder'][split] = {'sha256': data.sha256(path)}
        self.save_manifest()

    def save_manifest(self):
        (self.inputs / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_offline_rebuild_preserves_exact_bytes_and_detects_tampering(self):
        output = self.root / 'cases'
        proof = data.rebuild(self.inputs, output, source_root=self.sources)
        self.assertEqual(proof['cases'], 4)
        self.assertEqual(proof['revision'], data.REVISION)
        self.assertEqual(data.verify_materialized(self.inputs, output), 4)
        for name, payload in self.payloads.items():
            self.assertEqual((output / name).read_bytes(), payload)
        name = next(iter(self.payloads))
        (output / name).write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'case SHA differs'):
            data.verify_materialized(self.inputs, output)
        with self.assertRaises(FileExistsError):
            data.rebuild(self.inputs, output, source_root=self.sources)

    def test_download_uses_only_pinned_train_files_and_honors_offline_cache(self):
        with patch('huggingface_hub.hf_hub_download', side_effect=lambda **kw:
                   str(self.sources / kw['filename'])) as download:
            self.assertEqual(data.materialize(self.inputs, self.root / 'cached',
                cache_dir=self.root / 'cache', local_files_only=True), 4)
        self.assertEqual(download.call_count, 3)
        for call in download.call_args_list:
            self.assertEqual(call.kwargs['revision'], data.REVISION)
            self.assertEqual(call.kwargs['repo_id'], data.UPSTREAM)
            self.assertTrue(call.kwargs['local_files_only'])
            self.assertIn('/train-', call.kwargs['filename'])

    def test_rejects_changed_prompts_sources_and_unsafe_manifest_before_output(self):
        output = self.root / 'never-created'
        path = self.inputs / 'deepcoder/train.jsonl'
        original = path.read_bytes()
        path.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'prompt SHA differs'):
            data.rebuild(self.inputs, output, source_root=self.sources)
        path.write_bytes(original)
        source = next(self.sources.rglob('*.parquet'))
        source.write_bytes(source.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'parquet size/SHA differs'):
            data.rebuild(self.inputs, output, source_root=self.sources)
        self.assertFalse(output.exists())
        self.manifest['samples']['files']['../unsafe.parquet'] = {
            'size': 1, 'sha256': 'a' * 64, 'rows': 1}
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'source file manifest'):
            data.source_manifest(self.inputs)

    def test_judge_preparation_rebuilds_cases_without_archives(self):
        from qwen35 import judge_cases
        checker = self.root / 'checker/evaluation'
        checker.mkdir(parents=True)
        (checker / 'testing_util.py').write_text('checker fixture')
        extractor = checker.parent / 'utils/extraction_utils.py'
        extractor.parent.mkdir()
        extractor.write_text('extractor fixture')
        lcb = self.root / 'lcb'
        lcb.mkdir()
        case_payload = b'{"input_output":"{}"}'
        case_hashes = {f'platform__{i:03d}.json': hashlib.sha256(case_payload).hexdigest()
                       for i in range(175)}
        records = []
        for i, name in enumerate(case_hashes):
            (lcb / name).write_bytes(case_payload)
            records.append({'index': i, 'prompt_token_ids': [1], 'gold': '',
                            'judge': {'kind': 'lcb', 'key': name[:-5]}})
        test_file = self.inputs / 'deepcoder/lcb.jsonl'
        test_file.write_text(''.join(json.dumps(row) + '\n' for row in records))
        self.manifest['benchmark'] = {'lcb': {'sha256': data.sha256(test_file), 'rows': 175}}
        self.save_manifest()
        pinned = self.root / 'pinned.json'
        pinned.write_text(json.dumps({'case_sha256': case_hashes,
            'checker_sha256': data.sha256(checker / 'testing_util.py'),
            'extractor_sha256': data.sha256(extractor)}))
        with patch.object(judge_cases, 'PINNED', pinned):
            output = self.root / 'prepared'
            proof = judge_cases.prepare(self.inputs, lcb, checker, output,
                                       deepcoder_source_root=self.sources)
        self.assertEqual(proof['training_validation_cases'], 4)
        self.assertEqual(proof['lcb_v6_cases'], 175)
        self.assertTrue(proof['per_question_sha256_verified'])
        self.assertNotIn('archive_sha256', proof)
        self.assertEqual(len(list((output / 'cases').glob('*.json'))), 179)
        self.assertTrue((output / 'complete.json').is_file())


if __name__ == '__main__':
    unittest.main()
