"""Regression coverage for private weights, resume identity and run evaluation."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from qwen3 import runner
from qwen3.cli import selected_run_state
from tests.test_qwen3 import _portable_sft_run, _sft_config, _sft_rows, SFTToyBridge
from tests import test_vllm_training as vllm_fixtures


class CheckpointContracts(unittest.TestCase):
    def test_q35_fit_reaches_real_decoder_for_initial_and_final_validation(self):
        fixture = vllm_fixtures.VLLMTrainingTests()
        fixture.setUp()
        from qwen35.vllm_decode import VLLMDecoder
        for method in ('bridge', 'random', 'full', 'lora'):
            with self.subTest(method=method), TemporaryDirectory() as tmp:
                session = fixture.fresh()
                session.case.method = method
                del session.evaluate  # Exercise the real API -> decoder path.
                outputs = [dict(index=0, sample_index=0, sampling_seed=1234,
                                tokens=1, response=r'\boxed{3}', response_token_ids=[3])]
                with patch('platour_exps.require_vllm_python'), \
                     patch('qwen35.vllm_decode.subprocess.Popen') as spawn, \
                     patch.object(VLLMDecoder, '_wait', return_value={'complete': True}), \
                     patch.object(VLLMDecoder, 'generate', return_value=outputs), \
                     patch('qwen35.vllm_decode.export_lora'), \
                     patch('qwen35.cli.train_step', return_value={} ), \
                     patch('qwen35.model.verify_standard_weight') as release_check:
                    receipt = session.fit(fixture.train, fixture.val, output=Path(tmp)/'run',
                        objective='grpo', validation_samples=1, smoke=True, max_steps=1, batch_size=2,
                        response_cap=4, vllm_python=Path(sys.executable))
                    self.assertEqual(receipt['selected_step'], 0)
                    self.assertEqual(spawn.call_count, 2)
                    release_check.assert_not_called()
                    weight = session._current
                    weight.write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError, 'SHA differs'):
                        session._decoder(Path(tmp)/'tampered', 4, Path(sys.executable), 2)

    def test_q35_release_still_requires_published_identity(self):
        from qwen35.vllm_decode import VLLMDecoder
        with TemporaryDirectory() as tmp, \
             patch('qwen35.model.verify_standard_weight', side_effect=ValueError('release mismatch')) as verify:
            weight = Path(tmp)/'model.safetensors'
            with self.assertRaisesRegex(ValueError, 'release mismatch'):
                VLLMDecoder(SimpleNamespace(method='full'), Path(tmp), weight, Path(tmp)/'out',
                            worker_python=Path(sys.executable))
            verify.assert_called_once()

    def test_portable_resume_rejects_changed_recipe_data_or_foreign_state(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            _portable_sft_run(root)
            checkpoint = root/'checkpoints/step-0004'
            for changes in ({'learning_rate': .123}, {'warmup_steps': 2}, {'sender': 'another-model'}):
                with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError, 'protocol changed'):
                    runner.train(dict(_sft_config(), **changes), SFTToyBridge(), None,
                                 _sft_rows(8, 0), _sft_rows(3, 50), root, resume=checkpoint,
                                 final_test=_sft_rows(2, 90))
            with self.assertRaisesRegex(RuntimeError, 'protocol changed'):
                runner.train(_sft_config(), SFTToyBridge(), None, _sft_rows(8, 100),
                             _sft_rows(3, 50), root, resume=checkpoint, final_test=_sft_rows(2, 90))
            with self.assertRaisesRegex(ValueError, 'newest complete'):
                _portable_sft_run(root, resume=root/'checkpoints/step-0002')
            other = Path(tmp)/'other'
            other.mkdir()
            shutil.copy(root/'training_protocol.json', other/'training_protocol.json')
            with self.assertRaisesRegex(ValueError, 'this output root'):
                _portable_sft_run(other, resume=checkpoint)
            # A valid foreign checkpoint copied into the right directory must
            # still fail its protocol identity, even with compatible tensor shapes.
            fresh = Path(tmp)/'fresh'
            _portable_sft_run(fresh)
            shutil.copytree(fresh/'checkpoints/step-0004', checkpoint, dirs_exist_ok=True)
            with self.assertRaisesRegex(RuntimeError, 'another training protocol'):
                _portable_sft_run(root, resume=checkpoint)

    def test_backbone_contents_are_part_of_resume_identity(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp)/'model'
            source.mkdir()
            file = source/'model.safetensors'
            file.write_bytes(b'original')
            bridge = SFTToyBridge()
            bridge.training_sources = {'sender': str(source)}
            before = runner.training_identity({}, bridge, [], [], None, tmp)
            file.write_bytes(b'replaced')
            after = runner.training_identity({}, bridge, [], [], None, tmp)
            self.assertNotEqual(before['models'], after['models'])

    def test_selected_checkpoint_requires_commit_marker_and_hash(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            _portable_sft_run(root)
            marker = root/'checkpoints/step-0004/complete.json'
            original = json.loads(marker.read_text())
            marker.unlink()
            with self.assertRaisesRegex(ValueError, 'complete marker'):
                selected_run_state(root)
            for replacement in (dict(original, complete=False), dict(original, step=2),
                                dict(original, sha256={})):
                marker.write_text(json.dumps(replacement))
                with self.assertRaises(ValueError):
                    selected_run_state(root)
            marker.write_text(json.dumps(original))
            self.assertEqual(selected_run_state(root)[1]['selected_step'], 4)

    def test_native_sft_requires_terminal_budget_even_without_selection(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root/'checkpoints/step-0002'
            directory.mkdir(parents=True)
            torch.save({'P': {'weight': torch.ones(1)}}, directory/'bridge.pt')
            digest = hashlib.sha256((directory/'bridge.pt').read_bytes()).hexdigest()
            (directory/'complete.json').write_text(json.dumps(dict(complete=True, step=2,
                metric={'heldout_nll': 1.0}, sha256={'bridge.pt': digest})))
            for budget in (None, 6):
                if budget:
                    (root/'protocol.json').write_text(json.dumps({'config': {'total_steps': budget}}))
                with self.assertRaisesRegex(ValueError, 'Final SFT checkpoint'):
                    selected_run_state(root)
            (root/'protocol.json').write_text(json.dumps({'config': {'total_steps': 2}}))
            self.assertEqual(selected_run_state(root)[1]['selected_step'], 2)

    def test_native_grpo_snapshot_checks_validation_and_weight_digest(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root/'evaluation/step-0002'
            directory.mkdir(parents=True)
            weight = directory/'bridge.pt'
            torch.save({'P': {'weight': torch.ones(1)}}, weight)
            (directory/'checkpoint.json').write_text(json.dumps(dict(step=2,
                bridge_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(), bridge_state_sha256='state')))
            (directory/'summary.json').write_text(json.dumps(dict(complete=True,
                evaluation_role='validation', bridge_state_sha256='state')))
            (root/'selection.json').write_text(json.dumps(dict(selected_step=2,
                policy='train_holdout_best_validation_v1', test_used_for_selection=False,
                selected_checkpoint=str(weight), selected_bridge_state_sha256='state')))
            self.assertEqual(selected_run_state(root)[1]['selected_step'], 2)
            weight.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA differs'):
                selected_run_state(root)
