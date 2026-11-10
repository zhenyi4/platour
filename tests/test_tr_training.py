"""TR epoch coverage, native rollout contract, and TEST-blind checkpoint selection."""
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import load_file

from platour_api import ReceiverSession
from qwen35 import checkpoints


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.student.weight)
        self.student.device = torch.device('cpu')
        self.experiment = SimpleNamespace(method='full', lora_rank=128, lora_alpha=128)

    def trainable(self):
        return tuple(self.parameters())

    def inference_state(self):
        return self.student.state_dict()

    def load_inference_state(self, state):
        self.student.load_state_dict(state)

    def generate(self, *_args, **_kwargs):
        raise AssertionError('TR training must never fall back to HF generation')

    def response_log_probs(self, prompt, answer):
        return (-1 + self.student.weight.flatten() * (1 if answer == [3] else -1))


class TRTrainingTests(unittest.TestCase):
    def setUp(self):
        self.train = [dict(index=i, prompt_token_ids=[1, 10+i], gold='3') for i in range(5)]
        self.val = [dict(index=0, prompt_token_ids=[1, 99], gold='3')]
        self.sessions = []

    def fresh(self):
        session = ReceiverSession.__new__(ReceiverSession)
        session.actor = Actor()
        session.method = 'full'
        session.model_identity = 'toy-128'
        session.model_path = None
        session.tokenizer = SimpleNamespace(decode=lambda ids: r'\boxed{' + str(ids[0]) + '}')
        calls = []
        @contextmanager
        def decoder(output, cap, prompt_length, python, memory_fraction):
            snapshot = session.actor.student.weight.detach().clone()
            step = int(Path(output).name[5:])
            def generate(rows, *, count, cap, sampling, logprobs):
                self.assertTrue(logprobs)
                self.assertEqual(count, 8)
                self.assertEqual(sampling, dict(temperature=1., top_p=1., top_k=-1))
                calls.append((step, [r['index'] for r in rows], snapshot))
                return [dict(index=row['index'], sample_index=sample,
                    response_token_ids=[3 if sample % 2 else 4],
                    response=r'\boxed{' + str(3 if sample % 2 else 4) + '}',
                    log_probs=[-1 + float(snapshot) * (1 if sample % 2 else -1)])
                    for row in rows for sample in range(count)]
            yield SimpleNamespace(generate=generate)
        session._decoder = decoder
        def evaluate(rows, **kw):
            self.assertEqual(rows, self.val)  # TEST never enters fit.
            self.assertEqual(kw['samples'], 4)
            self.assertEqual(kw['backend'], 'vllm')
            step = int(kw['output'].name.split('-')[1])
            correct = 4 if step in (1, 2) else 0
            return dict(complete=True, questions=1, responses=4, correct=correct,
                        accuracy=correct/4, backend='vllm')
        session.evaluate = evaluate
        session.rollout_calls = calls
        self.sessions.append(session)
        return session

    def fit(self, session, root):
        return session.fit(self.train, self.val, output=root, batch_size=2,
            objective='grpo', task='gsm8k', validation_samples=4, response_cap=2, validation_schedule='step')

    def test_three_epochs_cover_every_row_and_restore_earliest_validation_best(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / 'run'
            session = self.fresh()
            receipt = self.fit(session, root)
            protocol = json.loads((root / 'protocol.json').read_text())
            self.assertEqual((protocol['epochs'], protocol['steps_per_epoch'], protocol['max_steps']), (3, 3, 9))
            records = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
            for epoch in (1, 2, 3):
                epoch_rows = [r for r in records if r['epoch'] == epoch]
                self.assertEqual([len(r['train_indices']) for r in epoch_rows], [2, 2, 1])
                self.assertEqual(sorted(i for r in epoch_rows for i in r['train_indices']), list(range(5)))
            selection = json.loads((root / 'selection.json').read_text())
            self.assertEqual(selection['best_step'], 1)  # step 2 ties; later steps lose.
            self.assertFalse(selection['test_used_for_selection'])
            self.assertEqual(receipt['selected_step'], 1)
            self.assertEqual(receipt['training_steps'], 9)
            self.assertEqual(len(session.rollout_calls), 9)
            snapshots = [float(item[2]) for item in session.rollout_calls]
            self.assertEqual(len(set(snapshots)), 9)  # Current weights exported on every update.
            best = load_file(str(root / 'checkpoints/step-0001/model.safetensors'))['weight']
            last = load_file(str(root / 'checkpoints/step-0009/model.safetensors'))['weight']
            self.assertFalse(torch.equal(best, last))
            torch.testing.assert_close(session.actor.student.weight, best)
            self.assertTrue(checkpoints.valid(root / 'checkpoints/step-0001'))
            restored = self.fresh()
            with patch.object(restored, '_decoder', side_effect=AssertionError('already trained')), \
                 patch.object(restored, 'evaluate', side_effect=AssertionError('already validated')):
                again = self.fit(restored, root)
            self.assertEqual(again, receipt)
            torch.testing.assert_close(restored.actor.student.weight, best)
            with self.assertRaises((RuntimeError, ValueError)):
                restored.fit(self.train, self.val, output=root, epochs=2, batch_size=2,
                    objective='grpo', task='gsm8k', validation_samples=4, response_cap=2, validation_schedule='step')
            (root / 'checkpoints/step-0001/model.safetensors').write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError, 'missing or corrupt'):
                self.fit(self.fresh(), root)

    def test_resume_after_validation_failure_matches_uninterrupted_selected_and_final_weights(self):
        with TemporaryDirectory() as folder:
            root, direct = Path(folder) / 'crash', Path(folder) / 'direct'
            original = checkpoints.atomic_json
            def interrupt(path, data):
                if str(path).endswith('evaluation/step-0005.json'):
                    raise RuntimeError('validation interrupted')
                return original(path, data)
            with patch.object(checkpoints, 'atomic_json', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'validation interrupted'):
                    self.fit(self.fresh(), root)
            self.assertEqual(checkpoints.latest(root).name, 'step-0005')
            self.fit(self.fresh(), root)
            self.fit(self.fresh(), direct)
            for step in (1, 9):
                name = f'checkpoints/step-{step:04d}/model.safetensors'
                self.assertEqual(checkpoints.sha256(root / name), checkpoints.sha256(direct / name))
            rows = [json.loads(line)['step'] for line in (root / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(rows, list(range(1, 10)))

    def test_missing_vllm_logprobs_fail_without_updating_or_falling_back(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / 'bad'
            session = self.fresh()
            from contextlib import nullcontext
            def generate(rows, **kwargs):
                return [dict(index=r['index'], sample_index=i, response_token_ids=[3], response='3')
                        for r in rows for i in range(8)]
            session._decoder = lambda *_a: nullcontext(SimpleNamespace(generate=generate))
            with self.assertRaisesRegex(RuntimeError, 'finite token log probabilities'):
                self.fit(session, root)
            self.assertEqual(checkpoints.latest(root).name, 'step-0000')
            self.assertFalse((root / 'complete.json').exists())

    def test_grpo_initial_only_smoke_resumes_without_revalidating_and_keeps_step_ten(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)/'smoke'
            calls = []
            def fit():
                session = self.fresh()
                with patch.object(session, 'evaluate', wraps=session.evaluate) as evaluate:
                    try:
                        receipt = session.fit(self.train, self.val, output=root, control='TR',
                            objective='grpo', task='gsm8k', batch_size=2, smoke=True, max_steps=10,
                            validation_samples=4, validation_schedule='initial', checkpoint_interval=4)
                    finally:
                        calls.extend(evaluate.call_args_list)
                return session, receipt
            original = checkpoints.save_step
            def interrupt(root, step, *args, **kwargs):
                original(root, step, *args, **kwargs)
                if step == 4:
                    raise RuntimeError('interrupt smoke')
            with patch.object(checkpoints, 'save_step', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'interrupt smoke'):
                    fit()
            session, receipt = fit()
            self.assertEqual(len(calls), 1)
            self.assertEqual(receipt['training_steps'], 10)
            self.assertFalse(receipt['selection_performed'])
            self.assertTrue(receipt['training_only'])
            self.assertEqual(receipt['selected_step'], 10)
            self.assertFalse((root/'checkpoints/step-0000').exists())
            self.assertIsNone(json.loads((root/'evaluation/step-0000.json').read_text())['checkpoint_sha256'])
            torch.testing.assert_close(session.actor.student.weight,
                load_file(str(root/'checkpoints/step-0010/model.safetensors'))['weight'])
            self.assertEqual(sorted(p.name for p in (root/'evaluation').iterdir()), ['step-0000.json'])
            fit()
            self.assertEqual(len(calls), 1)

    def test_checkpoint_sequence_crosses_four_digit_boundary(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            actor = Actor()
            optimizer = torch.optim.AdamW(actor.trainable())
            checkpoints.save_step(root, 0, actor, optimizer, None)
            old = root / 'checkpoints/step-0000'
            high = root / 'checkpoints/step-9999'
            old.rename(high)
            meta = json.loads((high / 'checkpoint.json').read_text())
            meta['step'] = 9999
            checkpoints.atomic_json(high / 'checkpoint.json', meta)
            marker = json.loads((high / 'complete.json').read_text())
            marker['step'] = 9999
            marker['sha256']['checkpoint.json'] = checkpoints.sha256(high / 'checkpoint.json')
            checkpoints.atomic_json(high / 'complete.json', marker)
            checkpoints.save_step(root, 10000, actor, optimizer, {'step': 10000})
            self.assertEqual(checkpoints.latest(root).name, 'step-10000')
            checkpoints.save_step(root, 10001, actor, optimizer, {'step': 10001})
            self.assertFalse(high.exists())
            self.assertTrue(checkpoints.valid(root / 'checkpoints/step-10001'))

    def test_sparse_checkpoint_hashes_new_payload_once_and_verifies_on_restore(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            actor = Actor()
            optimizer = torch.optim.AdamW(actor.trainable())
            previous = -1
            for step in (0, 10, 20, 30, 40):
                with patch.object(checkpoints, 'sha256', wraps=checkpoints.sha256) as digest:
                    checkpoints.save_step(root, step, actor, optimizer, {'step': step},
                        protected={0, 10}, identity='sparse', previous_step=previous)
                    self.assertEqual(digest.call_count, 4)
                    self.assertTrue(all('.pending' in str(call.args[0]) for call in digest.call_args_list))
                previous = step
            self.assertEqual({p.name for p in (root/'checkpoints').iterdir()},
                             {'step-0000', 'step-0010', 'step-0030', 'step-0040'})
            with self.assertRaisesRegex(RuntimeError, 'predecessor'):
                checkpoints.save_step(root, 50, actor, optimizer, None, previous_step=20)
            (root/'checkpoints/step-0040/model.safetensors').write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError, 'corrupt'):
                checkpoints.load_step(root/'checkpoints/step-0040', actor, optimizer, expected_identity='sparse')
            self.assertEqual(checkpoints.latest(root).name, 'step-0030')

    def test_full_and_lora_sft_default_rates_and_linear_schedule(self):
        class Student(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(4))
                self.device = torch.device('cpu')
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.weight.expand(*input_ids.shape, 4))
        train = [dict(row, response_token_ids=[3]) for row in self.train]
        val = [dict(row, response_token_ids=[3]) for row in self.val]
        for method, peak in (('full', 1e-5), ('lora', 1e-4)):
            with self.subTest(method=method), TemporaryDirectory() as folder:
                root = Path(folder)/'run'
                session = self.fresh()
                session.method = method
                session.actor.student = Student()
                session.fit(train, val, output=root, batch_size=2,
                            validation_backend='hf')
                protocol = json.loads((root/'protocol.json').read_text())
                self.assertEqual(protocol['learning_rate'], peak)
                self.assertEqual(protocol['warmup_steps'], 1)
                records = [json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
                for row, factor in zip(records, [0., 1., 7/8, 6/8, 5/8, 4/8, 3/8, 2/8, 1/8]):
                    self.assertAlmostEqual(row['learning_rate'], peak*factor)
                optimizer = torch.load(root/'checkpoints/step-0009/optimizer.pt', weights_only=True)
                self.assertEqual(optimizer['param_groups'][0]['lr'], 0.)

    def test_sft_accumulation_matches_batch_gradient_with_variable_lengths_and_short_batch(self):
        class Student(nn.Module):
            def __init__(self, check_graphs=False):
                super().__init__()
                self.weight = nn.Parameter(torch.arange(16).reshape(4, 4).float() / 17)
                self.device = torch.device('cpu')
                self.pending = False
                self.check_graphs = check_graphs

            def forward(self, input_ids, use_cache=False, attention_mask=None):
                if self.check_graphs and torch.is_grad_enabled():
                    if self.pending:
                        raise AssertionError('Previous row must backward before the next forward')
                    self.pending = True
                # A causal context makes padding/answer-position errors observable.
                embedded = self.weight[input_ids % 4]
                if attention_mask is not None:
                    embedded = embedded * attention_mask.unsqueeze(-1)
                logits = embedded.cumsum(dim=1)
                if self.check_graphs and torch.is_grad_enabled():
                    self.batch_sizes.append(input_ids.shape[0])
                if self.check_graphs and logits.requires_grad:
                    logits.register_hook(self.finished)
                return SimpleNamespace(logits=logits)

            def finished(self, gradient):
                self.pending = False
                return gradient

        train = [dict(row, response_token_ids=[3] * (1 + i % 3))
                 for i, row in enumerate(self.train)]
        val = [dict(row, response_token_ids=[3, 2]) for row in self.val]
        reference = Student()
        optimizer = torch.optim.AdamW(reference.parameters(), lr=.01, betas=(.9, .999), weight_decay=0.)
        expected_metrics = []
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, 1, 9)
        for epoch in range(3):
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(42+epoch)).tolist()
            for offset in range(0, len(train), 2):
                chosen = [train[i] for i in order[offset:offset+2]]
                optimizer.zero_grad(set_to_none=True)
                total = sum(len(row['response_token_ids']) for row in chosen)
                losses = []
                for row in chosen:
                    prompt, answer = row['prompt_token_ids'], row['response_token_ids']
                    logits = reference(torch.tensor([prompt+answer[:-1]])).logits
                    losses.append(torch.nn.functional.cross_entropy(
                        logits[0, len(prompt)-1:], torch.tensor(answer), reduction='sum'))
                loss = sum(losses) / total
                expected_metrics.append(float(loss.detach()))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
                optimizer.step()
                scheduler.step()
        with TemporaryDirectory() as folder:
            root = Path(folder) / 'sft'
            session = self.fresh()
            session.actor.student = Student(check_graphs=True)
            session.actor.student.batch_sizes = []
            session.fit(train, val, output=root, objective='sft', batch_size=2,
                        micro_batch=2, learning_rate=.01, validation_backend='hf', validation_schedule='epoch')
            self.assertEqual(session.actor.student.batch_sizes, [2, 2, 1]*3)
            final = load_file(str(root / 'checkpoints/step-0009/model.safetensors'))['weight']
            torch.testing.assert_close(final, reference.weight, atol=1e-6, rtol=1e-6)
            records = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
            torch.testing.assert_close(torch.tensor([r['train_nll'] for r in records]),
                                       torch.tensor(expected_metrics))
            self.assertEqual([r['response_tokens'] for r in records],
                             [sum(len(train[i]['response_token_ids']) for i in r['train_indices'])
                              for r in records])
            selection = json.loads((root/'selection.json').read_text())
            self.assertEqual(selection['best_step'], 9)
            self.assertEqual(selection['policy'], 'final_sft_checkpoint_v1')
            torch.testing.assert_close(session.actor.student.weight, final)
            self.assertEqual([c['step'] for c in selection['candidates']], [3, 6, 9])
            self.assertEqual(sorted(p.name for p in (root/'evaluation').glob('*.json')),
                             [f'step-{s:04d}.json' for s in (0, 3, 6, 9)])
            self.assertEqual(json.loads((root/'protocol.json').read_text())['checkpoint_steps'], [3, 6, 9])
            self.assertFalse((root/'checkpoints/step-0008').exists())
            self.assertFalse((root/'checkpoints/step-0000').exists())
            self.assertIsNone(json.loads((root/'evaluation/step-0000.json').read_text())['checkpoint_sha256'])
            resumed = Path(folder)/'resumed'
            original = checkpoints.save_step
            def interrupt(root, step, *args, **kwargs):
                original(root, step, *args, **kwargs)
                if step == 4:
                    raise RuntimeError('interrupt between validations')
            def fit_again():
                fresh = self.fresh()
                fresh.actor.student = Student(check_graphs=True)
                fresh.actor.student.batch_sizes = []
                return fresh.fit(train, val, output=resumed, objective='sft', batch_size=2,
                    micro_batch=2, learning_rate=.01, validation_backend='hf', validation_schedule='epoch', checkpoint_interval=4)
            with patch.object(checkpoints, 'save_step', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'between validations'):
                    fit_again()
            receipt = fit_again()
            self.assertEqual(receipt['selected_step'], selection['best_step'])
            torch.testing.assert_close(load_file(str(resumed/'checkpoints/step-0009/model.safetensors'))['weight'], final)
            # An interruption before the next checkpoint must roll back unsaved updates.
            resumed = Path(folder)/'unsaved'
            actual_step = torch.optim.AdamW.step
            calls = 0
            def interrupted_step(optimizer, *args, **kwargs):
                nonlocal calls
                actual_step(optimizer, *args, **kwargs)
                calls += 1
                if calls == 2:
                    raise RuntimeError('unsaved update interrupted')
            with patch.object(torch.optim.AdamW, 'step', new=interrupted_step):
                with self.assertRaisesRegex(RuntimeError, 'unsaved update'):
                    fit_again()
            self.assertEqual(len((resumed/'metrics.jsonl').read_text().splitlines()), 1)
            self.assertIsNone(checkpoints.latest(resumed))
            fit_again()
            self.assertTrue(list((resumed/'recovery').glob('step-0000.json-before-restart-*')))
            torch.testing.assert_close(load_file(str(resumed/'checkpoints/step-0009/model.safetensors'))['weight'], final)
            self.assertEqual(len((resumed/'metrics.jsonl').read_text().splitlines()), 9)


if __name__ == '__main__':
    unittest.main()
