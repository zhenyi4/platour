"""Frozen downloads, corruption handling and model-free CLI integration."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import platour_data as data


class PublishedDataTests(unittest.TestCase):
    def test_complete_inventory_matches_signed_protocols_and_public_commits(self):
        table = data.registry()
        self.assertEqual(len(table['files']), 68)
        revised = table['repositories']['qwen3'].get('split_policy') == 'published_train_holdout_v2'
        self.assertEqual(len(table['protocol_files']), 38 if revised else 32)
        split_receipt = json.loads((data.DATA.parent / 'docs/qwen3_grpo_split_release.json').read_text()) if revised else {}
        self.assertEqual(sum(spec['size'] for spec in table['files'].values()),
                         split_receipt['payload_bytes'] if revised else 528675845)
        release = json.loads((data.DATA.parent / 'docs/hf_public_release.json').read_text())
        commits = {r['repo_id']: r['revision'] for r in release['repositories']}
        for repo in table['repositories'].values():
            expected = split_receipt['revision'] if revised and repo['repo_id'] == split_receipt['repo_id'] else commits[repo['repo_id']]
            self.assertEqual(repo['revision'], expected)
        for name, spec in table['protocol_files'].items():
            data.verify(data.DATA / name, spec)
        for experiment in range(1, 19):
            names = data.experiment_files(experiment)
            self.assertEqual(len(names), 3)
            manifest = json.loads((data.DATA / Path(names[0]).parent / 'manifest.json').read_text())
            for name in names:
                self.assertEqual(table['files'][name]['sha256'], manifest['sha256'][Path(name).stem])
        manifest = json.loads((data.DATA / 'qwen35/manifest.json').read_text())
        for name, spec in table['files'].items():
            if not name.startswith('qwen35/'):
                continue
            path = Path(name)
            if path.stem in ('train', 'validation'):
                expected = manifest[path.parent.name][path.stem][
                    'parquet_sha256' if path.suffix == '.parquet' else 'sha256']
            else:
                expected = manifest['benchmark'][path.stem]['sha256']
            self.assertEqual(spec['sha256'], expected)

    def test_selection_preserves_split_roles_and_virtual_sender_templates(self):
        for experiment in range(1, 19):
            self.assertEqual([Path(n).stem for n in data.experiment_files(experiment, 'evaluate')], ['test'])
        self.assertEqual(len(data.experiment_files(19, 'train')), 4)
        self.assertEqual(len(data.experiment_files(26, 'train')), 2)
        self.assertEqual({Path(n).stem for n in data.experiment_files(28, 'evaluate')},
                         {'lcb', 'humaneval_template_adapted', 'mbpp_template_adapted'})
        self.assertEqual({Path(n).stem for n in data.experiment_files(29, 'evaluate')},
                         {'lcb', 'humaneval', 'mbpp'})
        self.assertEqual(len(set(n for i in range(1, 31) for n in data.experiment_files(i))), 68)

    def test_export_inventory_keeps_all_payloads_without_git_tracking(self):
        inventory = data.release_inventory()
        revised = data.registry()['repositories']['qwen3'].get('split_policy') == 'published_train_holdout_v2'
        self.assertEqual(len(inventory), 106 if revised else 100)
        self.assertEqual(sum(Path(n).suffix in ('.jsonl', '.parquet') for n in inventory), 68)
        self.assertNotIn('data/hf_manifest.json', inventory)

    def test_qwen3_cli_defaults_signed_paths_without_starting_models(self):
        from qwen3 import cli
        for action, experiment in (('evaluate', 1), ('train', 1), ('train', 13)):
            with self.subTest(action=action, experiment=experiment):
                argv = ['platour-exps', action, '--experiment', str(experiment),
                        '--sender', 'sender', '--receiver', 'receiver', '--output', 'unused',
                        '--backend' if action == 'evaluate' else '--eval-backend', 'vllm']
                with patch('platour_exps.require_vllm_python'), patch('sys.argv', argv), patch.object(cli, 'audit_inputs',
                        side_effect=RuntimeError('stop before models')) as audit:
                    with self.assertRaisesRegex(RuntimeError, 'stop before models'):
                        cli.main()
                paths = audit.call_args.args[1]
                parent = cli.ROOT / 'data/qwen3' / cli.select(experiment)['tag']
                expected = ('test',) if action == 'evaluate' else (
                    ('train', 'validation') if experiment == 1 else ('train', 'test'))
                if action == 'train' and cli.select(experiment).get('published_grpo_validation'):
                    expected = ('train', 'validation', 'test')
                self.assertEqual(paths, {split: parent / (split + '.jsonl') for split in expected})


class DownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source_root = self.root / 'checkout/data'
        self.source_root.mkdir(parents=True)
        self.table = data.registry()
        self.payload = b'{"index":0,"prompt_token_ids":[1],"gold":"1"}\n'
        self.source = self.root / 'cached-payload'
        self.source.write_bytes(self.payload)
        for spec in self.table['files'].values():
            spec.update(size=len(self.payload), sha256=hashlib.sha256(self.payload).hexdigest())
        for name, spec in self.table['protocol_files'].items():
            path = self.source_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'{"fixture_protocol":true}\n')
            spec.update(size=path.stat().st_size, sha256=data.digest(path))
        self.manifest = self.source_root / 'hf_manifest.json'
        self.save_registry()
        patches = (patch.object(data, 'DATA', self.source_root),
                   patch.object(data, 'REGISTRY', self.manifest),
                   patch('huggingface_hub.hf_hub_download', return_value=str(self.source)))
        for p in patches:
            result = p.start()
            self.addCleanup(p.stop)
        self.fetch = result
        self.name = next(name for name in self.table['files'] if name.startswith('qwen3/01_'))
        self.destination = self.root / 'restored' / self.name

    def save_registry(self):
        self.manifest.write_text(json.dumps(self.table))

    def test_download_pins_revision_and_honors_offline_cache_then_reuses_local_file(self):
        data.ensure_file(self.name, self.destination, local_files_only=True, cache_dir=self.root / 'cache')
        self.assertEqual(self.destination.read_bytes(), self.payload)
        kwargs = self.fetch.call_args.kwargs
        self.assertEqual(kwargs['revision'], self.table['repositories']['qwen3']['revision'])
        self.assertEqual(kwargs['repo_type'], 'dataset')
        self.assertEqual(kwargs['filename'], self.name)
        self.assertTrue(kwargs['local_files_only'])
        self.assertEqual(kwargs['cache_dir'], self.root / 'cache')
        data.ensure_file(self.name, self.destination)
        self.assertEqual(self.fetch.call_count, 1)

    def test_damaged_local_file_is_preserved_without_network(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b'user-edited')
        with self.assertRaisesRegex(ValueError, 'SHA-256 differs'):
            data.ensure_file(self.name, self.destination)
        self.assertEqual(self.destination.read_bytes(), b'user-edited')
        self.fetch.assert_not_called()

    def test_same_size_corrupt_cache_is_never_installed(self):
        self.source.write_bytes(b'x' * len(self.payload))
        with self.assertRaisesRegex(ValueError, 'SHA-256 differs'):
            data.ensure_file(self.name, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.destination.parent.glob('*.pending')), [])

    def test_failed_offline_download_leaves_no_payload(self):
        self.fetch.side_effect = FileNotFoundError('offline cache miss')
        with self.assertRaisesRegex(FileNotFoundError, 'offline cache miss'):
            data.ensure_file(self.name, self.destination, local_files_only=True)
        self.assertFalse(self.destination.exists())

    def test_matching_lfs_pointer_is_restored_but_wrong_identity_is_preserved(self):
        spec = self.table['files'][self.name]
        pointer = data.LFS_HEADER + f"oid sha256:{spec['sha256']}\nsize {spec['size']}\n".encode()
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(pointer)
        self.source.write_bytes(b'bad-cache')
        with self.assertRaises(ValueError):
            data.ensure_file(self.name, self.destination)
        self.assertEqual(self.destination.read_bytes(), pointer)
        self.source.write_bytes(self.payload)
        data.ensure_file(self.name, self.destination)
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.destination.write_bytes(pointer.replace(spec['sha256'].encode(), b'0' * 64))
        with self.assertRaisesRegex(ValueError, 'pointer identity differs'):
            data.ensure_file(self.name, self.destination)

    def test_concurrent_downloaders_share_one_atomic_install(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(lambda _: data.ensure_file(self.name, self.destination), range(8)))
        self.assertEqual(paths, [self.destination] * 8)
        self.fetch.assert_called_once()
        self.assertEqual(self.destination.read_bytes(), self.payload)

    def test_all_download_deduplicates_and_copies_protocols_to_custom_root(self):
        output = self.root / 'all-data'
        paths = data.download(range(1, 31), output, local_files_only=True)
        self.assertEqual(len(paths), 68)
        self.assertEqual(self.fetch.call_count, 68)
        for name in self.table['protocol_files']:
            self.assertEqual((output / name).read_bytes(), (self.source_root / name).read_bytes())
        data.download(range(1, 31), output, local_files_only=True)
        self.assertEqual(self.fetch.call_count, 68)

    def test_qwen35_custom_protocol_is_not_overwritten_or_downloaded(self):
        output = self.root / 'custom'
        output.mkdir()
        manifest = output / 'manifest.json'
        manifest.write_text('{"caller_owned":true}')
        data.ensure_qwen35_data(19, output, 'train')
        self.assertEqual(manifest.read_text(), '{"caller_owned":true}')
        self.fetch.assert_not_called()

    def test_qwen35_empty_custom_root_restores_only_requested_benchmarks(self):
        output = self.root / 'evaluation'
        paths = data.ensure_qwen35_data(28, output, 'evaluate', local_files_only=True)
        self.assertEqual(len(paths), 3)
        self.assertTrue((output / 'manifest.json').is_file())
        self.assertFalse((output / 'deepcoder/train.jsonl').exists())
        self.assertFalse((output / 'deepcoder/humaneval.jsonl').exists())

    def test_registry_rejects_unpinned_revisions_and_path_traversal(self):
        self.table['repositories']['qwen3']['revision'] = 'main'
        self.save_registry()
        with self.assertRaisesRegex(ValueError, 'pinned commit'):
            data.ensure_file(self.name, self.destination)
        self.table['repositories']['qwen3']['revision'] = 'a' * 40
        self.table['files']['../escape'] = self.table['files'].pop(self.name)
        self.save_registry()
        with self.assertRaisesRegex(ValueError, 'Invalid dataset path'):
            data.registry()
        self.fetch.assert_not_called()

    def test_qwen3_audit_hydrates_missing_signed_split(self):
        from qwen3 import cli
        config = cli.select(1)
        name = f"qwen3/{config['tag']}/test.jsonl"
        manifest = self.source_root / f"qwen3/{config['tag']}/manifest.json"
        manifest.write_text(json.dumps({'experiment_id': 1, 'method': 'sft',
            'sha256': {'test': self.table['files'][name]['sha256']}}))
        output = self.root / 'qwen3-test.jsonl'
        with patch.object(cli, 'ROOT', self.source_root.parent):
            cli.audit_inputs(config, {'test': output})
        self.assertEqual(output.read_bytes(), self.payload)
        self.fetch.assert_called_once()

    def test_qwen35_eval_hydrates_before_judge_or_model_loading(self):
        from qwen35 import cli
        from qwen35.catalog import select
        from types import SimpleNamespace
        args = SimpleNamespace(data_root=self.root / 'eval-root')
        with patch.object(cli, 'verify_code_evaluation_environment',
                          side_effect=RuntimeError('stop before model')):
            with self.assertRaisesRegex(RuntimeError, 'stop before model'):
                cli.run_eval(select(26), args)
        self.assertEqual(self.fetch.call_count, 3)
        self.assertTrue((args.data_root / 'deepcoder/lcb.jsonl').is_file())

    def test_command_dispatch_needs_no_model_runtime(self):
        import platour_exps
        with patch('sys.argv', ['platour-exps', 'download-data', '--all']), \
                patch.object(data, 'main') as main:
            platour_exps.main()
        main.assert_called_once_with(['--all'])


if __name__ == '__main__':
    unittest.main()
