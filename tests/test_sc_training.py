"""SC full-epoch coverage and validation-selected bridge recovery."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

import platour_api
from platour_api import PLatourSession
from qwen35.checkpoints import sha256


class SCTrainingTests(unittest.TestCase):
    def test_self_only_loader_shares_one_backbone_and_preserves_identity(self):
        model = nn.Linear(2, 2)
        model.device = torch.device('cpu')
        model.config = SimpleNamespace(num_hidden_layers=2, hidden_size=2,
                                       pad_token_id=0, _name_or_path='/receiver')
        tokenizer = SimpleNamespace(pad_token_id=0)
        with patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=model) as loader, \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer):
            session = PLatourSession.load_self_communication('/receiver', device_map='cpu')
        loader.assert_called_once_with('/receiver', torch_dtype=torch.bfloat16, local_files_only=True)
        self.assertIs(session.bridge.sender, model)
        self.assertIs(session.bridge.receiver, model)
        self.assertIs(session.tokenizer, tokenizer)
        self.assertEqual(session.model_identity, 'self:/receiver')
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(parameter.requires_grad and not parameter.count_nonzero()
                            for parameter in session.bridge.P.parameters()))

    def setUp(self):
        self.train = [dict(index=i, prompt_token_ids=[1, 10+i], gold='3',
                           response_token_ids=[3]) for i in range(5)]
        self.val = [dict(index=0, prompt_token_ids=[1, 99], gold='3', response_token_ids=[3])]

    def fresh(self):
        bridge = nn.Module()
        bridge.P = nn.ModuleList([nn.Linear(1, 1, bias=False)])
        nn.init.zeros_(bridge.P[0].weight)
        bridge.sender = SimpleNamespace()
        bridge.receiver = SimpleNamespace(device=torch.device('cpu'))
        session = PLatourSession(bridge, None, model_identity='toy-self-communication')
        def evaluate(rows, **kwargs):
            self.assertEqual(rows, self.val)
            self.assertEqual(kwargs['samples'], 2)
            step = int(kwargs['output'].name.split('-')[1])
            correct = 2 if step in (1, 2) else 0
            return dict(complete=True, questions=1, responses=2, correct=correct,
                        accuracy=correct/2, backend='vllm')
        session.evaluate = evaluate
        return session

    @staticmethod
    def sft_update(bridge, optimizer, train, indices, micro_batch, tokens):
        loss = (bridge.P[0].weight - 1).square().sum()
        loss.backward()
        return float(loss.detach()), tokens

    @staticmethod
    def grpo_update(bridge, tokenizer, train, indices, config, step, count, **kwargs):
        loss = (bridge.P[0].weight - 1).square().sum()
        loss.backward()
        return float(loss.detach()), count * 4

    def fit(self, session, root, objective):
        with patch('platour_api.sft_update', side_effect=self.sft_update), \
             patch('platour_api.grpo_update', side_effect=self.grpo_update), \
             patch('platour_api.heldout_nll', return_value=(1., 1)):
            return session.fit(self.train, self.val, output=root, control='SC',
                objective=objective, task='gsm8k', batch_size=2,
                validation_backend='hf' if objective == 'sft' else 'vllm', validation_samples=2, validation_schedule='step')

    def test_sft_restores_last_and_grpo_restores_validation_best(self):
        for objective in ('sft', 'grpo'):
            with self.subTest(objective=objective), TemporaryDirectory() as folder:
                root = Path(folder) / 'run'
                session = self.fresh()
                receipt = self.fit(session, root, objective)
                protocol = json.loads((root / 'protocol.json').read_text())
                self.assertEqual((protocol['epochs'], protocol['steps_per_epoch'], protocol['max_steps']), (3, 3, 9))
                metrics = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
                expected_lrs = [0., 1e-4, 7e-4/8, 6e-4/8, 5e-4/8, 4e-4/8, 3e-4/8, 2e-4/8, 1e-4/8] if objective == 'sft' else [1e-5]*9
                for row, lr in zip(metrics, expected_lrs):
                    self.assertAlmostEqual(row['learning_rate'], lr)
                self.assertEqual(protocol.get('scheduler'), 'linear' if objective == 'sft' else None)
                for epoch in (1, 2, 3):
                    rows = [r for r in metrics if r['epoch'] == epoch]
                    self.assertEqual([len(r['train_indices']) for r in rows], [2, 2, 1])
                    self.assertEqual(sorted(i for r in rows for i in r['train_indices']), list(range(5)))
                selection = json.loads((root / 'selection.json').read_text())
                selected = 9 if objective == 'sft' else 1
                self.assertEqual(selection['best_step'], selected)  # SFT ignores declining VAL scores.
                self.assertFalse(selection['test_used_for_selection'])
                self.assertEqual(receipt['selected_step'], selected)
                self.assertEqual(receipt['training_steps'], 9)
                best = torch.load(root / f'checkpoints/step-{selected:04d}/bridge.pt', weights_only=True)['P']['0.weight']
                last = torch.load(root / 'checkpoints/step-0009/bridge.pt', weights_only=True)['P']['0.weight']
                self.assertEqual(torch.equal(best, last), objective == 'sft')
                torch.testing.assert_close(session.bridge.P[0].weight, best)
                self.assertEqual({p.name for p in (root / 'checkpoints').iterdir()},
                                 ({'step-0000', 'step-0008', 'step-0009'} if objective == 'sft'
                                  else {'step-0000', 'step-0001', 'step-0008', 'step-0009'}))
                restored = self.fresh()
                with patch.object(restored, 'evaluate', side_effect=AssertionError('already validated')):
                    again = self.fit(restored, root, objective)
                self.assertEqual(receipt, again)
                torch.testing.assert_close(restored.bridge.P[0].weight, best)
                (root / f'checkpoints/step-{selected:04d}/bridge.pt').write_bytes(b'corrupt')
                with self.assertRaisesRegex(RuntimeError, 'missing or corrupt'):
                    self.fit(self.fresh(), root, objective)

    def test_validation_failure_resumes_to_identical_best_and_final_bridge(self):
        with TemporaryDirectory() as folder:
            root, direct = Path(folder) / 'crash', Path(folder) / 'direct'
            original = platour_api.atomic_json
            def interrupt(value, path):
                if str(path).endswith('evaluation/step-0005.json'):
                    raise RuntimeError('SC validation interrupted')
                return original(value, path)
            with patch('platour_api.atomic_json', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'SC validation interrupted'):
                    self.fit(self.fresh(), root, 'grpo')
            self.assertTrue((root / 'checkpoints/step-0005/complete.json').is_file())
            self.fit(self.fresh(), root, 'grpo')
            self.fit(self.fresh(), direct, 'grpo')
            for step in (1, 9):
                relative = f'checkpoints/step-{step:04d}/bridge.pt'
                self.assertEqual(sha256(root / relative), sha256(direct / relative))
            self.assertEqual([json.loads(line)['step'] for line in
                (root / 'metrics.jsonl').read_text().splitlines()], list(range(1, 10)))

    def test_completed_run_rejects_changed_budget_or_validation_scores(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / 'run'
            self.fit(self.fresh(), root, 'sft')
            with self.assertRaises((RuntimeError, ValueError)):
                self.fresh().fit(self.train, self.val, output=root, control='SC',
                    objective='sft', task='gsm8k', epochs=2, batch_size=2,
                    validation_backend='vllm', validation_samples=2, validation_schedule='step')
            path = root / 'evaluation/step-0009.json'
            value = json.loads(path.read_text())
            value.update(heldout_nll=2.)
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, 'completed selection differs'):
                self.fit(self.fresh(), root, 'sft')

    def test_epoch_sft_uses_final_despite_worse_loss_and_resumes(self):
        seen = []
        def loss(bridge, rows, micro_batch):
            self.assertEqual(rows, self.val)
            # Validation order is independent of the optimizer's LR schedule.
            step = len(seen) * 3
            seen.append(step)
            return {0: 0., 3: 1., 6: 1., 9: 2.}[step], 1
        def fit(root):
            session = self.fresh()
            with patch('platour_api.sft_update', side_effect=self.sft_update), \
                 patch('platour_api.heldout_nll', side_effect=loss), \
                 patch.object(session, 'evaluate', side_effect=AssertionError('SFT VAL must not generate')):
                receipt = session.fit(self.train, self.val, output=root, control='SC',
                    objective='sft', task=None, batch_size=2,
                    validation_backend='hf', validation_schedule='epoch')
            return receipt
        with TemporaryDirectory() as folder:
            direct, resumed = Path(folder)/'direct', Path(folder)/'resumed'
            receipt = fit(direct)
            self.assertEqual(seen, [0, 3, 6, 9])
            self.assertEqual(receipt['selected_step'], 9)  # Final loss is worst; selection still uses final.
            self.assertEqual([r['step'] for r in json.loads((direct/'selection.json').read_text())['candidates']],
                             [3, 6, 9])
            self.assertEqual(json.loads((direct/'evaluation/step-0000.json').read_text())['evaluation_role'], 'initial')
            self.assertEqual(json.loads((direct/'protocol.json').read_text())['checkpoint_steps'], [3, 6, 9])
            self.assertEqual({p.name for p in (direct/'checkpoints').iterdir()},
                             {'step-0006', 'step-0009'})
            seen.clear()
            original = platour_api.save_step
            def interrupt(root, step, *args, **kwargs):
                original(root, step, *args, **kwargs)
                if step == 3:
                    raise RuntimeError('interrupt between validations')
            with patch('platour_api.save_step', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'between validations'):
                    fit(resumed)
            again = fit(resumed)
            self.assertEqual(seen, [0, 3, 6, 9])
            self.assertEqual(again['selected_step'], 9)
            for step in (6, 9):
                relative = f'checkpoints/step-{step:04d}/bridge.pt'
                self.assertEqual(sha256(direct/relative), sha256(resumed/relative))
            seen.clear()
            fit(resumed)
            self.assertEqual(seen, [])

    def test_grpo_initial_only_smoke_keeps_final_weights_without_model_selection(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)/'smoke'
            session = self.fresh()
            with patch('platour_api.grpo_update', side_effect=self.grpo_update), \
                 patch.object(session, 'evaluate', wraps=session.evaluate) as evaluate:
                receipt = session.fit(self.train, self.val, output=root, control='SC',
                    objective='grpo', task='gsm8k', batch_size=2, smoke=True, max_steps=10,
                    validation_backend='vllm', validation_samples=2, validation_schedule='initial')
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(receipt['training_steps'], 10)
            self.assertEqual(receipt['selected_step'], 10)
            self.assertTrue(receipt['training_only'])
            self.assertFalse(receipt['selection_performed'])
            self.assertEqual(sorted(p.name for p in (root/'evaluation').iterdir()), ['step-0000.json'])
            self.assertEqual(json.loads((root/'selection.json').read_text())['candidates'], [])
            self.assertEqual({p.name for p in (root/'checkpoints').iterdir()}, {'step-0010'})
            final = torch.load(root/'checkpoints/step-0010/bridge.pt', weights_only=True)['P']['0.weight']
            torch.testing.assert_close(session.bridge.P[0].weight, final)
            with self.assertRaisesRegex(ValueError, 'restricted'):
                self.fresh().fit(self.train, self.val, output=Path(folder)/'formal',
                    objective='grpo', task='gsm8k', validation_schedule='initial')

    def test_sparse_resume_discards_unsaved_updates_then_replays_them(self):
        def fit(root, fail=False):
            session = self.fresh()
            updates = 0
            def update(*args):
                nonlocal updates
                updates += 1
                if fail and updates == fail:
                    raise RuntimeError('unsaved update interrupted')
                return self.sft_update(*args)
            def loss(bridge, rows, micro):
                return float((bridge.P[0].weight.detach()-1).square().sum()), 1
            with patch('platour_api.sft_update', side_effect=update), \
                 patch('platour_api.heldout_nll', side_effect=loss):
                return session.fit(self.train, self.val, output=root, objective='sft', control='SC',
                    batch_size=2, validation_backend='hf', validation_schedule='epoch')
        with TemporaryDirectory() as folder:
            direct, resumed = Path(folder)/'direct', Path(folder)/'resumed'
            fit(direct)
            with self.assertRaisesRegex(RuntimeError, 'unsaved update'):
                fit(resumed, fail=5)
            self.assertEqual(len((resumed/'metrics.jsonl').read_text().splitlines()), 4)
            self.assertFalse((resumed/'checkpoints/step-0004').exists())
            fit(resumed)
            self.assertEqual(sha256(direct/'checkpoints/step-0009/bridge.pt'),
                             sha256(resumed/'checkpoints/step-0009/bridge.pt'))
            self.assertEqual((direct/'metrics.jsonl').read_text(), (resumed/'metrics.jsonl').read_text())
            self.assertTrue(list((resumed/'recovery').glob('metrics-before-resume-*.jsonl')))
            # Before any checkpoint exists, restart from fresh weights and remeasure VAL.
            resumed = Path(folder)/'before-first-save'
            with self.assertRaisesRegex(RuntimeError, 'unsaved update'):
                fit(resumed, fail=2)
            self.assertFalse((resumed/'checkpoints').exists())
            fit(resumed)
            self.assertEqual(sha256(direct/'checkpoints/step-0009/bridge.pt'),
                             sha256(resumed/'checkpoints/step-0009/bridge.pt'))
            self.assertEqual((direct/'metrics.jsonl').read_text(), (resumed/'metrics.jsonl').read_text())
            self.assertTrue(list((resumed/'recovery').glob('step-0000.json-before-restart-*')))
            self.assertIsNone(json.loads((resumed/'evaluation/step-0000.json').read_text())['checkpoint_sha256'])


if __name__ == '__main__':
    unittest.main()
