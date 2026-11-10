import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import platour_it as it
from platour_it import sha

class ITFixture(unittest.TestCase):
    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root/'bundle'
        self.job = '16_chemistry_cot'
        inputs = self.bundle/'inputs'/self.job
        self.write_json(inputs/'rows.json', [{'index': 0, 'prompt_token_ids': [1, 2, 3], 'gold': 'A'}])
        self.write_json(inputs/'pairs.json', [{'test_index': 0, 'student_prompt_ids': [1, 2, 3],
            'teacher_prompt_ids': [1, 4, 3], 'pairing_kind': 'rephrased_training_question'}])
        self.write_json(inputs/'config.json', {'experiment': 16, 'questions': 1, 'samples': 8})
        supplement = self.bundle/'supplements/rephrases.jsonl'
        supplement.parent.mkdir()
        supplement.write_text(json.dumps({'test_index': 0, 'target_prompt_tokens': 3,
            'teacher_prompt_token_ids': [1, 4, 3]})+'\n')
        self.manifest = dict(schema_version=1, kind='frozen_qwen3_it', supplement_records=1,
            semantic_audit='inherited', jobs={self.job: {'questions': 1, 'samples': 8}},
            files={p.relative_to(self.bundle).as_posix(): {'size': p.stat().st_size, 'sha256': sha(p)}
                   for p in self.bundle.rglob('*') if p.is_file()})
        self.save_manifest()

    def save_manifest(self):
        self.write_json(self.bundle/'manifest.json', self.manifest)

    def replace_file(self, relative, value):
        p = self.bundle/relative
        self.write_json(p, value)
        self.manifest['files'][relative] = dict(size=p.stat().st_size, sha256=sha(p))
        self.save_manifest()


class PinnedITTests(ITFixture):
    def setUp(self):
        super().setUp()
        self.replace_file(f'inputs/{self.job}/config.json', dict(experiment=16, questions=1,
            samples=8, cap=2048, seed_stride=8, mapping='uniform_endpoints', task='chemistry_cot',
            sampling={'temperature': .7, 'top_p': .8, 'top_k': 20}, duplicate_test_prompt_rows=0,
            rows_using_later_duplicate_donor=0, donor_lookup_policy='last pairing wins', scope='fixture'))
        self.spec = dict(repo_id='fixture/data', revision='a'*40, profile='archived-it-v1',
            manifest=self.manifest, manifest_sha256=sha(self.bundle/'manifest.json'))
        self.release_patch = patch.object(it, 'release', return_value=self.spec)
        self.release_patch.start(); self.addCleanup(self.release_patch.stop)
        def fetch(**kwargs):
            self.assertEqual(kwargs['revision'], 'a'*40)
            self.assertEqual(kwargs['repo_type'], 'dataset')
            return str(self.bundle/kwargs['filename'].removeprefix('it/v1/'))
        self.fetch_patch = patch('huggingface_hub.hf_hub_download', side_effect=fetch)
        self.fetch = self.fetch_patch.start(); self.addCleanup(self.fetch_patch.stop)
        self.destination = self.root/'downloaded'

    def test_selected_download_is_pinned_and_reuses_verified_local_files(self):
        it.ensure_inputs([16], self.destination, local_files_only=True, cache_dir=self.root/'cache')
        self.assertEqual(self.fetch.call_count, 4)
        for call in self.fetch.call_args_list:
            self.assertTrue(call.kwargs['local_files_only'])
            self.assertEqual(call.kwargs['cache_dir'], self.root/'cache')
        self.fetch.reset_mock()
        it.ensure_inputs([16], self.destination)
        self.fetch.assert_not_called()

    def test_fixed_donor_load_and_custom_template_rejection(self):
        rows, donors, config, audit = it.load_inputs(16, self.destination)
        self.assertEqual(donors, {0: [1, 4, 3]})
        self.assertEqual(audit['coverage'], 1.0)
        self.assertEqual(audit['pairing_counts'], {'rephrased_training_question': 1})
        it.load_inputs(16, self.destination, test_rows=rows)
        for changed in [dict(rows[0], prompt_token_ids=[1, 9, 3]), dict(rows[0], gold='B')]:
            with self.assertRaisesRegex(ValueError, 'Test prompts/gold differ'):
                it.load_inputs(16, self.destination, test_rows=[changed])

    def test_historical_duplicate_prompt_behavior_preserved(self):
        self.replace_file(f'inputs/{self.job}/rows.json', [
            {'index': i, 'prompt_token_ids': [1, 2, 3], 'gold': 'A'} for i in range(2)])
        self.replace_file(f'inputs/{self.job}/pairs.json', [
            {'test_index': i, 'student_prompt_ids': [1, 2, 3], 'teacher_prompt_ids': [1, 4+i, 3],
             'pairing_kind': 'original_exact_length_training_question'} for i in range(2)])
        p = self.bundle/f'inputs/{self.job}/config.json'
        config = json.loads(p.read_text())
        config.update(questions=2, duplicate_test_prompt_rows=1, rows_using_later_duplicate_donor=1)
        self.replace_file(f'inputs/{self.job}/config.json', config)
        self.manifest['jobs'][self.job]['questions'] = 2
        rows, donors, _, audit = it.load_inputs(16, self.destination)
        self.assertEqual(donors, {0: [1, 5, 3], 1: [1, 5, 3]})
        self.assertEqual(audit['rows_using_later_duplicate_donor'], 1)


    def test_modified_payload_is_rejected(self):
        it.ensure_inputs([16], self.destination)
        (self.destination/self.job/'rows.json').write_text('[]')
        with self.assertRaises(ValueError):
            it.load_inputs(16, self.destination)

    def test_invalid_donor_length_is_rejected(self):
        self.replace_file(f'inputs/{self.job}/pairs.json', [{'test_index': 0,
            'student_prompt_ids': [1, 2, 3], 'teacher_prompt_ids': [1, 4],
            'pairing_kind': 'original_exact_length_training_question'}])
        with self.assertRaisesRegex(ValueError, 'Invalid frozen IT donor'):
            it.load_inputs(16, self.destination)

    def test_missing_rewrite_provenance_is_rejected(self):
        self.replace_file('supplements/rephrases.jsonl', {'test_index': 99,
            'teacher_prompt_token_ids': [1, 4, 3]})
        p = self.bundle/'supplements/rephrases.jsonl'
        p.write_text(json.dumps(json.loads(p.read_text()))+'\n')
        self.manifest['files']['supplements/rephrases.jsonl'] = dict(size=p.stat().st_size, sha256=sha(p))
        with self.assertRaisesRegex(ValueError, 'Missing IT rewrite provenance'):
            it.load_inputs(16, self.destination)
