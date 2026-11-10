"""Refresh published usage while preserving artifact and dataset provenance."""
import ast
import re
import unittest

from qwen35.release_card import (PROJECT_URL, update_dataset_usage,
                                 update_usage_sections, usage_sections)
from scripts.prepare_hf_release import (HARDWARE, PRIVATE_KEY, dataset_card,
                                        experiment_records)


class ReleaseCardTests(unittest.TestCase):
    def test_all_published_model_formats_preserve_metadata_and_provenance(self):
        for record in experiment_records():
            standard = record['id'] in (23, 24, 29, 30)
            with self.subTest(experiment=record['id']):
                prefix = '---\nlicense: apache-2.0\n---\n# Checkpoint\n\nOriginal paper notice.\n\n'
                heading = 'Export dependency and scope' if standard else 'Evaluation provenance'
                suffix = f'## {heading}\n\nOriginal provenance and version record.\n\n## Citation\n\nCitation text.\n'
                old_usage = usage_sections(record, standard_format=standard).replace(
                    'python3.11 scripts/install_environment.py', 'python3.11 -m venv .venv-vllm')
                card = prefix + old_usage.rstrip() + '\n\n' + suffix
                updated = update_usage_sections(card, record)
                self.assertTrue(updated.startswith(prefix))
                self.assertTrue(updated.endswith(suffix))
                self.assertIn('python3.11 scripts/install_environment.py', updated)
                self.assertNotIn('.venv-vllm', updated)
                self.assertNotIn('vllm_python=', updated)
                self.assertNotIn('PLATOUR_VLLM_PYTHON', updated)
                self.assertEqual(update_usage_sections(updated, record), updated)
                self.assertIsNone(HARDWARE.search(updated))
                self.assertIsNone(PRIVATE_KEY.search(updated))
                for code in re.findall(r'```python\n(.*?)```', updated, re.S):
                    ast.parse(code)

    def test_legacy_model_layout_remains_supported(self):
        for record in (experiment_records()[0], experiment_records()[22]):
            standard = record['id'] == 23
            start = 'Load' if standard else 'Loading'
            end = 'Export dependency and scope' if standard else 'Reproducibility and licensing'
            provenance = '' if standard else '## Evaluation provenance\n\nOriginal notice.\n\n'
            card = f'# Checkpoint\n\n{provenance}## {start}\n\nOld instructions.\n\n## {end}\n\nOriginal terms.\n'
            updated = update_usage_sections(card, record)
            self.assertIn('scripts/install_environment.py', updated)
            self.assertIn('Original terms.', updated)
            self.assertEqual(updated.count('Original notice.'), 0 if standard else 1)
            self.assertEqual(update_usage_sections(updated, record), updated)

    def test_dataset_updates_preserve_front_matter_splits_and_source_notices(self):
        for group, identifier in (('qwen3', 11), ('dapo', 19), ('deepcoder', 26)):
            with self.subTest(group=group):
                prefix = ('---\nconfigs:\n  - config_name: original\n'
                          '    data_files:\n      - split: test\n        path: test.jsonl\n'
                          '---\n# Frozen data\n\nOriginal split notice.\n\n')
                suffix = '## Sources and use\n\nOriginal source and license notices.\n'
                updated = update_dataset_usage(prefix + suffix, group)
                self.assertTrue(updated.startswith(prefix))
                self.assertTrue(updated.endswith(suffix))
                self.assertIn(f'platour-exps download-data --experiment {identifier}', updated)
                self.assertEqual(update_dataset_usage(updated, group), updated)
                self.assertEqual(updated.count('## Use with PLatour\n'), 1)
                self.assertIsNone(HARDWARE.search(updated))
                self.assertIsNone(PRIVATE_KEY.search(updated))

    def test_new_dataset_exports_include_current_instructions(self):
        tracked = ['data/qwen3/11_example/train.jsonl']
        for group in ('qwen3', 'dapo', 'deepcoder'):
            card = dataset_card(group, 'owner', tracked)
            self.assertIn(f'{PROJECT_URL}#getting-started', card)
            self.assertIn('## Use with PLatour\n', card)
            self.assertIn('## Sources and use\n', card)

    def test_ambiguous_or_reordered_sections_are_rejected(self):
        record = experiment_records()[0]
        for card in ('# Unknown layout\n',
                     f'## Evaluation provenance\n\n[Code]({PROJECT_URL})\n',
                     f'[Code]({PROJECT_URL})\n[Code]({PROJECT_URL})\n## Evaluation provenance\n'):
            with self.subTest(card=card), self.assertRaises(ValueError):
                update_usage_sections(card, record)
        for card in ('# Missing sources\n',
                     '## Sources and use\n\n## Sources and use\n',
                     '## Sources and use\n\n## Use with PLatour\n'):
            with self.subTest(card=card), self.assertRaises(ValueError):
                update_dataset_usage(card, 'qwen3')


if __name__ == '__main__':
    unittest.main()
