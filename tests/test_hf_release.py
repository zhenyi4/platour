"""Release integrity tests using tiny weight fixtures and mocked Hub downloads."""
from contextlib import chdir
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('platour_release_prepare', ROOT / 'scripts/prepare_hf_release.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
import platour_exps


class HFReleaseTests(unittest.TestCase):
    def test_export_preserves_source_and_tensor_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / 'source.safetensors', root / 'public.safetensors'
            tensor = b'\x00\x00\x80\x3f'
            original_header = {'x': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]},
                '__metadata__': {'format': 'pt', 'family': 'qwen3', 'selected_step': '17',
                    'gpu_type': 'private-accelerator', 'training_duration': 'private-duration'}}
            encoded = json.dumps(original_header).encode()
            encoded += b' ' * (-len(encoded) % 8)
            original = struct.pack('<Q', len(encoded)) + encoded + tensor
            source.write_bytes(original)
            audit = prepare.public_weight(source, target)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(target.read_bytes()[8 + len(encoded):], tensor)
            self.assertEqual(prepare.header(target)[1]['__metadata__'], {'format': 'pt', 'family': 'qwen3'})
            self.assertTrue(audit['tensor_payload_unchanged'])
            self.assertEqual(audit['public_sha256'], hashlib.sha256(target.read_bytes()).hexdigest())
            # Mutating the export cannot affect the original, including on APFS.
            with target.open('r+b') as stream:
                stream.seek(-1, 2)
                stream.write(b'\x00')
            self.assertEqual(source.read_bytes(), original)

    def test_card_scope_and_operational_metadata_rejection(self):
        record = prepare.experiment_records()[0]
        card = prepare.model_card(record, 'owner/data')
        self.assertIn('without benchmark scores', card)
        self.assertNotIn('selected using test-set', card)
        self.assertNotIn('validation-only selection', card)
        config = {'family': 'qwen3', 'evaluation': {'unbiased_heldout_result_claimed': False}}
        prepare.check_public_config(config)
        for name in ['selected_checkpoint_step', 'training_steps', 'gpu_type', 'training_duration']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                prepare.check_public_config(dict(config, **{name: 'private'}))

    def fixture(self, root, experiment=1):
        manifest = json.loads((ROOT / 'models/manifest.json').read_text())
        identifier = f'{experiment:02d}'
        payload = b'tiny-weight-fixture'
        filename = manifest['files'][identifier][0]
        weight = {'filename': filename, 'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
        config = {'schema_version': 1, 'experiment_id': experiment, 'weight': weight}
        encoded = json.dumps(config).encode()
        manifest['files'][identifier] = [filename, len(payload), weight['sha256']]
        manifest['configs'][identifier] = hashlib.sha256(encoded).hexdigest()
        (root / 'manifest.json').write_text(json.dumps(manifest))
        return manifest['repositories'][identifier], payload, filename, encoded

    def test_real_config_is_loaded_before_weight_and_cached(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository, payload, filename, encoded = self.fixture(root)
            calls = []
            def download(*, repo_id, filename, local_dir):
                self.assertEqual(repo_id, repository)
                calls.append(filename)
                path = Path(local_dir) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(encoded if filename == 'config.json' else payload)
                return str(path)
            with patch.object(platour_exps, 'MODEL_DIR', root), patch('huggingface_hub.hf_hub_download', download):
                self.assertEqual(platour_exps.ensure_weight(1), root / filename)
                self.assertEqual(platour_exps.ensure_weight(1), root / filename)
            self.assertEqual(calls, ['config.json', filename])

    def test_tampered_config_prevents_weight_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, _ = self.fixture(root)
            def download(*, repo_id, filename, local_dir):
                self.assertEqual(filename, 'config.json')
                path = Path(local_dir) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
                return str(path)
            with patch.object(platour_exps, 'MODEL_DIR', root), patch('huggingface_hub.hf_hub_download', download):
                with self.assertRaisesRegex(ValueError, 'configuration SHA differs'):
                    platour_exps.ensure_weight(1)

    def test_manifest_cannot_redirect_config_to_another_weight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, encoded = self.fixture(root)
            config = json.loads(encoded)
            config['experiment_id'] = 2
            encoded = json.dumps(config).encode()
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest['configs']['01'] = hashlib.sha256(encoded).hexdigest()
            (root / 'manifest.json').write_text(json.dumps(manifest))
            cached = root / '.cache/platour-configs/01' / manifest['configs']['01'] / 'config.json'
            cached.parent.mkdir(parents=True)
            cached.write_bytes(encoded)
            with patch.object(platour_exps, 'MODEL_DIR', root):
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    platour_exps.ensure_weight(1)

    def test_local_weight_inputs_do_not_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / 'owner' / 'model'
            existing.parent.mkdir()
            existing.write_bytes(b'local')
            with patch('platour_exps.ensure_weight') as fetch, chdir(root):
                for value in ('owner/model', './owner/missing', '../owner/missing',
                              '/owner/missing', 'owner/model.safetensors'):
                    self.assertEqual(platour_exps.resolve_weight_input(1, value), Path(value))
                with patch('pathlib.Path.expanduser', return_value=root / 'expanded'):
                    self.assertEqual(platour_exps.resolve_weight_input(1, '~/weights'), root / 'expanded')
            fetch.assert_not_called()

    def test_explicit_hub_repository_is_checked_even_with_cached_weight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository, payload, filename, encoded = self.fixture(root)
            calls = []
            def download(*, repo_id, filename, local_dir):
                calls.append((repo_id, filename))
                path = Path(local_dir) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(encoded if repo_id in (repository, 'mirror/copy') else b'{}')
                return str(path)
            (root / filename).write_bytes(payload)
            with patch.object(platour_exps, 'MODEL_DIR', root), patch('huggingface_hub.hf_hub_download', download):
                self.assertEqual(platour_exps.resolve_weight_input(1, repository), root / filename)
                self.assertEqual(platour_exps.resolve_weight_input(1, 'mirror/copy'), root / filename)
                with self.assertRaisesRegex(ValueError, 'configuration SHA differs'):
                    platour_exps.resolve_weight_input(1, 'mirror/wrong-experiment')
            self.assertEqual(calls, [(repository, 'config.json'), ('mirror/copy', 'config.json'),
                                    ('mirror/wrong-experiment', 'config.json')])

    def test_both_evaluation_clis_resolve_hub_id_before_loading_models(self):
        # Exercise real argument parsing, routing, download and integrity checks.
        # Stop before loading backbones or producing any benchmark result.
        for experiment, stop_at in ((1, 'qwen3.cli.audit_published_weight'),
                                    (19, 'qwen35.cli._model')):
            with self.subTest(experiment=experiment), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                repository, payload, filename, encoded = self.fixture(root, experiment)
                def download(*, repo_id, filename, local_dir):
                    self.assertEqual(repo_id, repository)
                    path = Path(local_dir) / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(encoded if filename == 'config.json' else payload)
                    return str(path)
                def stop(*args):
                    resolved = args[1] if experiment == 1 else args[1].weights
                    self.assertEqual(resolved, root / filename)
                    raise RuntimeError('stop before backbone loading')
                argv = ['platour-exps', 'evaluate', '--experiment', str(experiment),
                        '--weights', repository, '--sender', str(root / 'sender'),
                        '--receiver', str(root / 'receiver'),
                        '--output', str(root / 'evaluation'), '--backend', 'vllm']
                (root / 'sender').mkdir()
                (root / 'receiver').mkdir()
                if experiment == 1:
                    argv += ['--test-data', '/test.jsonl']
                with patch.object(platour_exps, 'MODEL_DIR', root), \
                     patch('huggingface_hub.hf_hub_download', download), \
                     patch('qwen3.cli.audit_inputs'), patch(stop_at, side_effect=stop), \
                     patch('platour_exps.require_vllm_python'), patch.object(sys, 'argv', argv):
                    with self.assertRaisesRegex(RuntimeError, 'stop before backbone loading'):
                        platour_exps.main()

    def test_backbone_local_paths_take_precedence_and_missing_paths_do_not_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'owner' / 'model').mkdir(parents=True)
            with chdir(root), patch('huggingface_hub.snapshot_download') as download:
                self.assertEqual(platour_exps.resolve_model_input('owner/model'), Path('owner/model'))
                for value in ('./owner/missing', '../owner/missing', '/missing/model', 'missing'):
                    with self.subTest(value=value), self.assertRaises(FileNotFoundError):
                        platour_exps.resolve_model_input(value)
            download.assert_not_called()

    def test_standard_full_and_peft_downloads_use_pinned_config_and_separate_weight_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = json.loads((ROOT / 'models/manifest.json').read_text())
            contents, calls = {}, []
            for identifier in ('23', '24', '29', '30'):
                full = identifier in ('23', '29')
                filename = 'model.safetensors' if full else 'adapter_model.safetensors'
                config_filename = 'config.json' if full else 'adapter_config.json'
                config = ({'model_type': 'qwen3_5_text', 'tie_word_embeddings': False} if full else
                          {'peft_type': 'LORA', 'r': 128, 'lora_alpha': 128,
                           'base_model_name_or_path': 'Qwen/Qwen3.5-4B'})
                payload, encoded = identifier.encode(), json.dumps(config).encode()
                manifest['files'][identifier] = [filename, len(payload), hashlib.sha256(payload).hexdigest()]
                manifest['configs'][identifier] = hashlib.sha256(encoded).hexdigest()
                contents[manifest['repositories'][identifier]] = {filename: payload, config_filename: encoded}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            def download(*, repo_id, filename, local_dir):
                calls.append((repo_id, filename))
                path = Path(local_dir) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents[repo_id][filename])
                return str(path)
            with patch.object(platour_exps, 'MODEL_DIR', root), patch('huggingface_hub.hf_hub_download', download):
                for identifier in ('23', '24', '29', '30'):
                    path = platour_exps.ensure_weight(int(identifier))
                    self.assertEqual(path.read_bytes(), identifier.encode())
                    self.assertIn(identifier, path.parts)
                    self.assertEqual(platour_exps.ensure_weight(int(identifier)), path)
            self.assertEqual(len(calls), 8)

    def test_backbone_snapshot_download_includes_weights_and_tokenizer_and_checks_config(self):
        import fnmatch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'config.json').write_text('{}')
            with patch('huggingface_hub.snapshot_download', return_value=str(root)) as download:
                self.assertEqual(platour_exps.resolve_model_input('owner/backbone'), root)
                self.assertEqual(download.call_args.kwargs['repo_id'], 'owner/backbone')
                patterns = download.call_args.kwargs['allow_patterns']
                for name in ('config.json', 'model.safetensors.index.json',
                             'model-00001-of-00002.safetensors', 'tokenizer.json',
                             'tokenizer_config.json', 'tokenizer.model', 'vocab.json',
                             'merges.txt', 'chat_template.jinja'):
                    self.assertTrue(any(fnmatch.fnmatch(name, pattern) for pattern in patterns), name)
                (root / 'config.json').unlink()
                with self.assertRaisesRegex(ValueError, 'no config.json'):
                    platour_exps.resolve_model_input('owner/adapter-only')

    def test_both_evaluation_clis_resolve_hub_backbones_before_model_loading(self):
        for experiment, stop_at in ((1, 'qwen3.cli.load_bridge'), (19, 'qwen35.cli._model')):
            with self.subTest(experiment=experiment), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = {name: root / name for name in ('sender', 'receiver')}
                for path in paths.values():
                    path.mkdir()
                    (path / 'config.json').write_text('{}')
                calls = []
                def snapshot(*, repo_id, allow_patterns):
                    calls.append(repo_id)
                    return str(paths[repo_id.split('/')[1]])
                def stop(*args, **kwargs):
                    sender, receiver = ((args[0], args[1]) if experiment == 1 else
                                        (args[1].sender, args[1].receiver))
                    self.assertEqual((Path(sender), Path(receiver)), (paths['sender'], paths['receiver']))
                    raise RuntimeError('stop before backbone loading')
                argv = ['platour-exps', 'evaluate', '--experiment', str(experiment),
                        '--weights', str(root / 'local.safetensors'), '--sender', 'owner/sender',
                        '--receiver', 'owner/receiver', '--output', str(root / 'evaluation'),
                        '--backend', 'vllm']
                if experiment == 1:
                    argv += ['--test-data', '/test.jsonl']
                with patch('huggingface_hub.snapshot_download', side_effect=snapshot), \
                     patch('qwen3.cli.audit_inputs'), patch('qwen3.cli.audit_published_weight'), \
                     patch(stop_at, side_effect=stop), patch('platour_exps.require_vllm_python'), patch.object(sys, 'argv', argv):
                    with self.assertRaisesRegex(RuntimeError, 'stop before backbone loading'):
                        platour_exps.main()
                self.assertCountEqual(calls, ['owner/sender', 'owner/receiver'])


if __name__ == '__main__':
    unittest.main()
