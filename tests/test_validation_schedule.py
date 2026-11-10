"""Default validation cadence, scheduled selection and recovery between VAL nodes."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import torch
from safetensors.torch import load_file
from training_protocol import validation_steps


class ValidationScheduleTests(TestCase):
    def test_final_only_is_explicit_grpo_smoke_option(self):
        for family in ('qwen3', 'qwen35'):
            self.assertEqual(validation_steps('final', 'grpo', 'vllm', 'dapo', 4, 12,
                smoke=True, family=family), [12])
            with self.assertRaisesRegex(ValueError, 'smoke'):
                validation_steps('final', 'grpo', 'vllm', 'dapo', 4, 12, family=family)
        with self.assertRaisesRegex(ValueError, 'GRPO'):
            validation_steps('final', 'sft', 'hf', 'gsm8k', 4, 12, smoke=True)

    def test_final_only_apis_validate_once_resume_and_select_terminal(self):
        from contextlib import ExitStack
        from test_sc_training import SCTrainingTests
        from test_tr_training import TRTrainingTests
        from test_vllm_training import VLLMTrainingTests, decoder
        for kind in (SCTrainingTests, TRTrainingTests, VLLMTrainingTests):
            fixture = kind(); fixture.setUp()
            with self.subTest(api=kind.__name__), TemporaryDirectory() as folder, ExitStack() as stack:
                if kind is SCTrainingTests:
                    stack.enter_context(patch('platour_api.grpo_update', side_effect=fixture.grpo_update))
                if kind is VLLMTrainingTests:
                    stack.enter_context(patch('qwen35.vllm_decode.training_decoder', decoder))
                    stack.enter_context(patch('qwen35.cli._grade', side_effect=lambda c,r,text,tokens,*a,**kw: int(tokens == [3])))
                root = Path(folder)/'run'
                kwargs = dict(output=root, objective='grpo', batch_size=2, smoke=True,
                              validation_schedule='final', response_cap=2,
                              validation_samples=4 if kind is TRTrainingTests else 2)
                if kind is not VLLMTrainingTests:
                    kwargs.update(task='gsm8k', checkpoint_interval=2)
                final = 3 if kind is VLLMTrainingTests else 9
                session = fixture.fresh()
                # A failed terminal VAL must resume without replaying signed updates.
                with patch.object(session, 'evaluate', side_effect=RuntimeError('terminal VAL interrupted')) as evaluate:
                    with self.assertRaisesRegex(RuntimeError, 'terminal VAL interrupted'):
                        session.fit(fixture.train, fixture.val, **kwargs)
                    self.assertEqual(evaluate.call_count, 1)
                    self.assertIn(f'step-{final:04d}', str(evaluate.call_args.kwargs['output']))
                before = (root/'metrics.jsonl').read_text()
                restored = fixture.fresh()
                with patch.object(restored, 'evaluate', wraps=restored.evaluate) as evaluate:
                    receipt = restored.fit(fixture.train, fixture.val, **kwargs)
                    self.assertEqual(evaluate.call_count, 1)
                self.assertEqual((root/'metrics.jsonl').read_text(), before)
                self.assertEqual(receipt['selected_step'], final)
                self.assertEqual(sorted(p.name for p in (root/'evaluation').glob('*.json')), [f'step-{final:04d}.json'])
                with patch.object(restored, 'evaluate', side_effect=AssertionError('already validated')):
                    restored.fit(fixture.train, fixture.val, **kwargs)
                with self.assertRaisesRegex(RuntimeError, '[Rr]esume'):
                    restored.fit(fixture.train, fixture.val, **dict(kwargs, validation_schedule='step'))

    def test_qwen3_half_epochs_and_sft_epochs(self):
        for per_epoch, expected in ((53, [0,27,53,80,106,133,159]),
                                    (109, [0,55,109,164,218,273,327]),
                                    (110, [0,55,110,165,220,275,330])):
            self.assertEqual(validation_steps('auto', 'grpo', 'vllm', 'gsm8k', per_epoch, per_epoch*3), expected)
            self.assertEqual(validation_steps('auto', 'sft', 'hf', None, per_epoch, per_epoch*3),
                             [0, per_epoch, per_epoch*2, per_epoch*3])
        self.assertEqual(validation_steps('auto', 'grpo', 'vllm', 'gsm8k', 109, 10, smoke=True), [0,10])

    def test_qwen35_all_published_recipes_and_short_custom_budget(self):
        from qwen35.catalog import EXPERIMENTS
        for case in EXPERIMENTS:
            self.assertEqual(validation_steps('auto', 'grpo', 'vllm', case.task, case.steps, case.steps,
                family='qwen35', case=case), [0, *case.validation_steps])
            self.assertEqual(validation_steps('auto', 'sft', 'vllm', case.task, case.steps, case.steps,
                family='qwen35', case=case), [0, case.steps])
        from types import SimpleNamespace
        for method, dim, extra in [('bridge', 2048, True), ('bridge', 2560, False),
                                   ('bridge', 4096, False), ('random', 2560, True),
                                   ('full', 2560, False), ('lora', 2560, False)]:
            steps = validation_steps('auto', 'grpo', 'vllm', 'deepcoder', 295, 295,
                family='qwen35', case=SimpleNamespace(method=method, teacher_dim=dim))
            self.assertEqual(294 in steps, extra)
        self.assertEqual(validation_steps('auto','grpo','vllm','dapo',3,3,family='qwen35'), [0,3])

    def test_qwen3_api_defaults_restrict_selection_to_scheduled_nodes(self):
        from test_sc_training import SCTrainingTests
        from test_tr_training import TRTrainingTests
        for fixture_type in (SCTrainingTests, TRTrainingTests):
            fixture = fixture_type(); fixture.setUp()
            cases = [('sft',[0,3,6,9]), ('grpo',[0,2,3,5,6,8,9])] if fixture_type is SCTrainingTests else [('grpo',[0,2,3,5,6,8,9])]
            for objective, expected in cases:
                with self.subTest(api=fixture_type.__name__, objective=objective), TemporaryDirectory() as tmp:
                    session = fixture.fresh()
                    train = [dict(r, response_token_ids=[3]) for r in fixture.train]
                    val = [dict(r, response_token_ids=[3]) for r in fixture.val]
                    fixture.val = val
                    kwargs = dict(output=Path(tmp)/'run', batch_size=2, objective=objective,
                                  task='gsm8k', validation_backend='hf' if objective == 'sft' else 'vllm',
                                  validation_samples=2 if fixture_type is SCTrainingTests else 4)
                    if fixture_type is SCTrainingTests:
                        with patch('platour_api.sft_update', side_effect=fixture.sft_update), \
                             patch('platour_api.grpo_update', side_effect=fixture.grpo_update), \
                             patch('platour_api.heldout_nll', return_value=(1., 1)):
                            receipt = session.fit(train, val, **kwargs)
                    else:
                        receipt = session.fit(train, val, **kwargs)
                    root = kwargs['output']
                    protocol = json.loads((root/'protocol.json').read_text())
                    self.assertEqual(protocol['validation_steps'], expected)
                    self.assertEqual(sorted(int(p.stem.split('-')[1]) for p in (root/'evaluation').glob('*.json')), expected)
                    self.assertEqual(receipt['selected_step'], 9 if objective == 'sft' else 2)

    def test_qwen35_sparse_validation_resume_and_best_checkpoint(self):
        from test_vllm_training import VLLMTrainingTests
        from qwen35 import checkpoints
        fixture = VLLMTrainingTests(); fixture.setUp()
        train = [dict(index=i, prompt_token_ids=[1, 100+i], gold='3', response_token_ids=[3]) for i in range(70)]
        seen = []
        def fresh():
            session = fixture.fresh()
            def evaluate(rows, **kw):
                step = int(kw['output'].name.split('-')[1]); seen.append(step)
                correct = 2 if step == 33 else 0
                return dict(complete=True, questions=1, responses=2, correct=correct, accuracy=correct/2, backend='vllm')
            session.evaluate = evaluate
            return session
        def update(actor, optimizer, *args, **kwargs):
            optimizer.zero_grad(); loss=(actor.student.weight-1).square().sum()
            loss.backward(); optimizer.step()
            return {'loss':float(loss.detach())}
        def fit(root):
            session = fresh()
            with patch('qwen35.cli.train_step', side_effect=update):
                result = session.fit(train, fixture.val, output=root, batch_size=1, validation_samples=2)
            return session, result
        with TemporaryDirectory() as folder:
            direct = Path(folder)/'direct'
            _, expected = fit(direct)
            self.assertEqual(seen, [0,33,66,70]); self.assertEqual(expected['selected_step'],33)
            for interrupted in (33,34):
                root = Path(folder)/f'interrupted-{interrupted}'
                original = checkpoints.save_step
                def stop(output, step, *args, **kwargs):
                    result = original(output, step, *args, **kwargs)
                    if step == interrupted: raise RuntimeError('interrupted commit')
                    return result
                seen.clear()
                with patch.object(checkpoints,'save_step',side_effect=stop):
                    with self.assertRaisesRegex(RuntimeError,'interrupted commit'): fit(root)
                _, result = fit(root)
                self.assertEqual(seen,[0,33,66,70])
                self.assertEqual(result['selected_step'],33)
                self.assertEqual(sorted(int(p.stem.split('-')[1]) for p in (root/'evaluation').glob('*.json')), [0,33,66,70])
                for step in (33,70):
                    a=load_file(root/f'checkpoints/step-{step:04d}/model.safetensors')
                    b=load_file(direct/f'checkpoints/step-{step:04d}/model.safetensors')
                    torch.testing.assert_close(a['weight'],b['weight'],atol=0,rtol=0)
                seen.clear(); fit(root); self.assertEqual(seen,[])

    def test_qwen35_sft_default_validates_initial_and_epoch_end(self):
        from test_vllm_training import VLLMTrainingTests
        fixture=VLLMTrainingTests(); fixture.setUp()
        with TemporaryDirectory() as folder:
            root=Path(folder)/'run'
            session=fixture.fresh()
            receipt=session.fit(fixture.train,fixture.val,output=root,batch_size=2,objective='sft',validation_samples=2)
            self.assertEqual(receipt['selected_step'],3)
            self.assertEqual(json.loads((root/'protocol.json').read_text())['validation_steps'],[0,3])
            self.assertEqual(sorted(p.name for p in (root/'evaluation').glob('*.json')),['step-0000.json','step-0003.json'])
            # Validation schedules are signed and cannot change during resume.
            with self.assertRaisesRegex(RuntimeError,'resume'):
                session.fit(fixture.train,fixture.val,output=root,batch_size=2,objective='sft',validation_samples=2,validation_schedule='step')
