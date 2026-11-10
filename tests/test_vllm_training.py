"""CPU contracts for vLLM-only rollouts, epoch budgets and validation selection."""
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

from platour_api import Qwen35Session
from training_protocol import budget, rollout_log_probs


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.student.weight)
        self.student.device = torch.device('cpu')

    def trainable(self):
        return tuple(self.parameters())

    def inference_state(self):
        return self.student.state_dict()

    def load_inference_state(self, state):
        self.student.load_state_dict(state)

    def generate(self, *a, **kw):
        raise AssertionError('HF rollout is forbidden')

    def response_log_probs(self, prompt, tokens):
        return -1 + self.student.weight.flatten() * (1 if tokens == [3] else -1)


@contextmanager
def decoder(actor, case, receiver, output, **options):
    snapshot = actor.student.weight.detach().clone()
    def generate(rows, *, count, cap, seed_base, training):
        assert training and count == 8
        return [dict(index=row['index'], sample_index=i, response_token_ids=[3 if i % 2 else 4],
                     log_probs=[-1 + float(snapshot) * (1 if i % 2 else -1)])
                for row in rows for i in range(count)]
    yield SimpleNamespace(generate=generate)


class VLLMTrainingTests(unittest.TestCase):
    def setUp(self):
        self.train = [dict(index=i, prompt_token_ids=[1, 10+i], gold='3', response_token_ids=[3])
                      for i in range(5)]
        self.val = [dict(index=0, prompt_token_ids=[1, 99], gold='3', response_token_ids=[3])]

    def fresh(self, fail=None):
        session = Qwen35Session.__new__(Qwen35Session)
        session.actor = Actor()
        session.case = SimpleNamespace(method='full', task='dapo')
        session.receiver = Path('/unused-receiver')
        session.identity = 'test-model'
        session.tokenizer = SimpleNamespace(decode=lambda ids, **kw: str(ids[0]))
        session.code_cases = session.official_checker = session.docker_image = None
        session._verify_judge = lambda rows: None
        def evaluate(rows, **kw):
            self.assertEqual(rows, self.val)
            self.assertEqual(kw['backend'], 'vllm')
            step = int(kw['output'].name.split('-')[1])
            if fail == step:
                raise RuntimeError('interrupted validation')
            correct = 2 if step in (1, 2) else 0
            return dict(complete=True, questions=1, responses=2, correct=correct,
                        accuracy=correct/2, backend='vllm')
        session.evaluate = evaluate
        original_log_probs = session.actor.response_log_probs
        def log_probs(prompt, tokens):
            if fail is not None and torch.is_inference_mode_enabled() and prompt == self.val[0]['prompt_token_ids']:
                if int(session._current.parent.name.split('-')[1]) == fail:
                    raise RuntimeError('interrupted validation')
            return original_log_probs(prompt, tokens)
        session.actor.response_log_probs = log_probs
        return session

    def fit(self, session, path, objective='grpo'):
        with patch('qwen35.vllm_decode.training_decoder', decoder), \
             patch('qwen35.cli._grade', side_effect=lambda c,r,text,tokens,*a,**kw: int(tokens == [3])):
            return session.fit(self.train, self.val, output=path, batch_size=2,
                               validation_samples=2, response_cap=2, objective=objective, control='TR', validation_schedule='step')

    def test_q35_one_epoch_best_checkpoint_and_completed_reuse(self):
        for objective in ('sft', 'grpo'):
            with self.subTest(objective=objective), TemporaryDirectory() as tmp:
                path = Path(tmp) / 'run'
                session = self.fresh()
                receipt = self.fit(session, path, objective)
                selected = 3 if objective == 'sft' else 1
                self.assertEqual((receipt['training_steps'], receipt['selected_step']), (3, selected))
                protocol = json.loads((path / 'protocol.json').read_text())
                self.assertEqual((protocol['epochs'], protocol['steps_per_epoch']), (1, 3))
                metrics = [json.loads(line) for line in (path / 'metrics.jsonl').read_text().splitlines()]
                self.assertEqual([len(row['train_indices']) for row in metrics], [2, 2, 1])
                self.assertEqual(sorted(i for r in metrics for i in r['train_indices']), list(range(5)))
                best = load_file(path / f'checkpoints/step-{selected:04d}/model.safetensors')['weight']
                torch.testing.assert_close(session.actor.student.weight, best)
                restored = self.fresh(fail=0)
                self.assertEqual(self.fit(restored, path, objective), receipt)
                torch.testing.assert_close(restored.actor.student.weight, best)
                history = path / 'evaluation/step-0003.json'
                value = json.loads(history.read_text()); value['heldout_nll' if objective == 'sft' else 'correct'] = 2
                history.write_text(json.dumps(value))
                with self.assertRaisesRegex(RuntimeError, 'history|denominator'):
                    self.fit(self.fresh(), path, objective)

    def test_q35_resume_matches_uninterrupted(self):
        for objective in ('sft', 'grpo'):
            for method in ('full', 'lora', 'bridge', 'random'):
                with self.subTest(objective=objective, method=method), TemporaryDirectory() as tmp:
                    run, direct = Path(tmp) / 'resume', Path(tmp) / 'direct'
                    def fresh(fail=None):
                        session = self.fresh(fail=fail)
                        session.case.method = method
                        return session
                    with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                        self.fit(fresh(fail=2), run, objective)
                    resumed, baseline = fresh(), fresh()
                    self.fit(resumed, run, objective)
                    self.fit(baseline, direct, objective)
                    torch.testing.assert_close(resumed.actor.student.weight, baseline.actor.student.weight, rtol=0, atol=0)
                    records = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
                    peak = (1e-5 if method == 'full' else 1e-4) if objective == 'sft' else (1e-6 if method == 'full' else 1e-5)
                    if objective == 'sft':
                        for row, lr in zip(records, [0., peak, peak/2]):
                            self.assertAlmostEqual(row['train/learning_rate'], lr)
                    protocol = json.loads((run/'protocol.json').read_text())
                    self.assertEqual(protocol['learning_rate'], peak)
                    self.assertEqual(protocol.get('scheduler'), 'linear' if objective == 'sft' else None)
                    for name in ('model.safetensors', 'optimizer.pt'):
                        if name.endswith('.safetensors'):
                            left = load_file(run / 'checkpoints/step-0003' / name)
                            right = load_file(direct / 'checkpoints/step-0003' / name)
                            torch.testing.assert_close(left['weight'], right['weight'], rtol=0, atol=0)
                        else:
                            left = torch.load(run / 'checkpoints/step-0003' / name, weights_only=True)
                            right = torch.load(direct / 'checkpoints/step-0003' / name, weights_only=True)
                            self.assertEqual(left['param_groups'], right['param_groups'])
                            for key, value in left['state'][0].items():
                                torch.testing.assert_close(value, right['state'][0][key], rtol=0, atol=0)

    def test_budget_and_missing_logprob_rejection(self):
        self.assertEqual(budget(5, 2, 3, 3), (3, 9))
        self.assertEqual(budget(5, 2, 1, 1), (3, 3))
        for args in [(5,2,2,3), (5,2,1,1,1)]:
            with self.assertRaises(ValueError): budget(*args)
        self.assertEqual(budget(5,2,1,1,2,True), (3,2))
        for values in (None, [], [float('nan')], [0., 0.]):
            with self.assertRaises(RuntimeError):
                rollout_log_probs(dict(response_token_ids=[3], log_probs=values), 2)

    def test_q3_update_requires_vllm_and_uses_returned_logprobs(self):
        from qwen3.runner import grpo_update
        actor = Actor()
        bridge = SimpleNamespace(P=actor.student, response_log_probs=actor.response_log_probs,
                                 generate=actor.generate)
        config = dict(task='gsm8k', group_size=2, response_cap=2,
                      train_sampling=dict(temperature=1., top_p=1., top_k=-1), rollout_output='/unused')
        calls = []
        @contextmanager
        def q3_decoder(bridge, path, **kw):
            def generate(rows, **kw):
                calls.append(kw)
                return [dict(index=r['index'], sample_index=i, response_token_ids=[3 if i else 4],
                             log_probs=[-2.]) for r in rows for i in range(2)]
            yield SimpleNamespace(generate=generate)
        with patch('qwen3.vllm_decode.VLLMDecoder', q3_decoder), \
             patch('qwen3.runner.score', side_effect=lambda task,text,*a,**kw: int(text == '3')):
            loss, correct = grpo_update(bridge, SimpleNamespace(decode=lambda ids:str(ids[0])),
                                       self.train, [0,1], config, 3, 2)
        self.assertEqual(correct, 2)
        self.assertTrue(calls[0]['parity'])
        self.assertEqual(calls[0]['seed_base'], 3042)
        self.assertTrue(torch.isfinite(actor.student.weight.grad).all())
        self.assertNotEqual(float(actor.student.weight.grad), 0.)

    def test_q35_worker_handoff_includes_first_token_logprob_and_eos(self):
        from qwen35.vllm_worker import response_row
        row = dict(index=7, sample_index=1, sampling_seed=100,
                   prompt_token_ids=[1,2], first_token=3, first_log_prob=-.4)
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: str(ids))
        output = SimpleNamespace(prompt_token_ids=[1,2,3], outputs=[SimpleNamespace(
            token_ids=[4,5], finish_reason='stop',
            logprobs=[{4:SimpleNamespace(logprob=-.5)}, {5:SimpleNamespace(logprob=-.6)}])])
        reply = response_row(row, output, 'bridge', 3, tokenizer, True)
        self.assertEqual(reply['response_token_ids'], [3,4,5])
        self.assertEqual(reply['log_probs'], [-.4,-.5,-.6])
        for reason, field in [('stop','handoff_is_eos'), ('length','handoff_only')]:
            reply = response_row(dict(row, **{field:True}), None, 'bridge', 1, tokenizer, True)
            self.assertEqual(reply['log_probs'], [-.4])
            self.assertEqual(reply['finish_reason'], reason)
        output.outputs[0].logprobs = None
        with self.assertRaisesRegex(RuntimeError, 'Missing'):
            response_row(row, output, 'bridge', 3, tokenizer, True)

    def test_q35_rollout_snapshots_track_current_weights_and_clean_success(self):
        from qwen35.vllm_decode import training_decoder
        actor, seen = Actor(), []
        @contextmanager
        def native(case, receiver, weight, output, **kwargs):
            seen.append((Path(weight), load_file(weight)['weight'].clone(), kwargs['device']))
            yield SimpleNamespace()
        with TemporaryDirectory() as tmp, patch('qwen35.vllm_decode.VLLMDecoder', native):
            for value in (1., 2.):
                with torch.no_grad(): actor.student.weight.fill_(value)
                with training_decoder(actor, SimpleNamespace(method='full'), Path('/unused'), tmp,
                                      cap=2, prompt_length=3):
                    self.assertTrue(seen[-1][0].exists())
                self.assertFalse(seen[-1][0].exists())
            self.assertEqual([float(v) for _,v,_ in seen], [1.,2.])
            self.assertNotEqual(seen[0][0], seen[1][0])
            self.assertEqual(seen[0][2], actor.student.device)

    def test_q35_bridge_training_first_token_uses_unfiltered_distribution(self):
        from qwen35.vllm_decode import VLLMDecoder
        logits = torch.tensor([-1., 0., 1., 2.])
        with TemporaryDirectory() as tmp:
            d = VLLMDecoder.__new__(VLLMDecoder)
            d.actor = SimpleNamespace(prefix_cache=lambda prompt: (None, logits))
            d.identity, d.memory = 'snapshot', Path(tmp)
            row = dict(index=9, prompt_token_ids=[1,2])
            with patch('qwen35.vllm_decode.hybrid_tensors', return_value={'test':torch.ones(1)}), \
                 patch('qwen35.vllm_decode.sample_first', side_effect=AssertionError('eval sampling in TRAIN')):
                rows = d._export(row, 2, 8, seed_base=1042, training=True)
            self.assertEqual([r['sampling_seed'] for r in rows], list(range(1114,1122)))
            for result in rows:
                self.assertAlmostEqual(result['first_log_prob'],
                    float(logits.log_softmax(-1)[result['first_token']]))
