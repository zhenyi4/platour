"""Real Gloo collectives; tiny actors and mocked decoding are not GPU benchmarks."""
from contextlib import contextmanager, ExitStack
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import patch

import torch
from torch import nn
from qwen3.parallel import parallel_evaluate, collective_evaluation, dist


class Bridge(nn.Module):
    pad_token_id = 0
    def __init__(self):
        super().__init__()
        self.P = nn.ModuleList([nn.Linear(1, 1, bias=False)])
        nn.init.zeros_(self.P[0].weight)
        self.sender = SimpleNamespace()
        self.receiver = SimpleNamespace(device=torch.device('cpu'))

    def batch_response_loss(self, rows, pad):
        values = torch.tensor([r['prompt_token_ids'][1] for r in rows], dtype=torch.float,
                              device=self.P[0].weight.device)
        return ((self.P[0].weight.flatten() - values / 100) ** 2).mean()

    def response_log_probs(self, prompt, tokens):
        return -1 + self.P[0].weight.flatten() * (1 if tokens == [3] else -1)


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 8))
        self.device = torch.device('cpu')
        self.config = SimpleNamespace(pad_token_id=0)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.weight.expand(*input_ids.shape, 8))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = LM()
        self.experiment = SimpleNamespace(method='full', task='dapo', lora_rank=2, lora_alpha=2)

    def trainable(self): return tuple(self.parameters())
    def inference_state(self): return self.student.state_dict()
    def load_inference_state(self, state): self.student.load_state_dict(state)
    def response_log_probs(self, prompt, tokens):
        return self.student.weight.log_softmax(-1)[0, tokens]


@parallel_evaluate
def evaluate(self, rows, *, samples=2, **kwargs):
    correct = sum(row['index'] % 2 == 0 for row in rows) * samples
    return dict(complete=True, questions=len(rows), responses=len(rows) * samples,
                correct=correct, accuracy=correct / (len(rows) * samples), backend='vllm')


@contextmanager
def rollout(model, *args, **kwargs):
    def generate(rows, *, count, **options):
        generated = []
        for row in rows:
            for i in range(count):
                tokens = [3 if i % 2 else 4]
                with torch.no_grad():
                    logs = model.response_log_probs(row['prompt_token_ids'], tokens).tolist()
                generated.append(dict(index=row['index'], sample_index=i,
                    response_token_ids=tokens, response=str(tokens[0]), tokens=1,
                    log_probs=logs, finish_reason='stop'))
        return generated
    yield SimpleNamespace(generate=generate)


def session(kind):
    from platour_api import PLatourSession, ReceiverSession, Qwen35Session
    tokenizer = SimpleNamespace(decode=lambda tokens, **kw: str(tokens[0]))
    if kind == 'sc':
        result = PLatourSession(Bridge(), tokenizer, model_identity='parallel-test-sc')
    elif kind == 'tr':
        result = ReceiverSession.__new__(ReceiverSession)
        result.actor = Actor()
        result.tokenizer = tokenizer
        result.method, result.model_identity, result.model_path = 'full', 'parallel-test-tr', None
        result._decoder = lambda *a, **kw: rollout(result.actor)
    else:
        result = Qwen35Session.__new__(Qwen35Session)
        result.actor = Actor()
        result.case = result.actor.experiment
        result.identity, result.receiver = 'parallel-test-q35', Path('/unused')
        result.tokenizer = tokenizer
        result.code_cases = result.official_checker = result.docker_image = None
        result._verify_judge = lambda rows: None
    result.evaluate = MethodType(evaluate, result)
    if os.environ.get('PLATOUR_TEST_DEVICE'):
        device = torch.device(os.environ['PLATOUR_TEST_DEVICE'])
        if kind == 'sc':
            result.bridge.to(device)
            result.bridge.receiver.device = device
        else:
            result.actor.to(device)
            result.actor.student.device = device
    return result


def fit(kind, objective, output, *, fail_step=None):
    result = session(kind)
    train = [dict(index=i, prompt_token_ids=[1, 10 + i], gold='3', response_token_ids=[3])
             for i in range(int(os.environ.get('PLATOUR_TEST_ROWS', '5')))]
    validation = [dict(index=20, prompt_token_ids=[1, 99], gold='3', response_token_ids=[3])]
    options = dict(output=output, objective=objective, max_steps=3, smoke=True,
        batch_size=int(os.environ.get('PLATOUR_TEST_BATCH', '2')),
        validation_samples=2, response_cap=2, validation_schedule='step',
        control='SC' if kind == 'sc' else 'TR')
    if kind != 'q35':
        options.update(task='gsm8k', micro_batch=1)
    original_evaluate = result.evaluate
    def failing(rows, **kw):
        if fail_step is not None and kw['output'].name == f'step-{fail_step:04d}-vllm':
            raise RuntimeError('injected validation interruption')
        return original_evaluate(rows, **kw)
    if fail_step is not None:
        result.evaluate = failing
    with ExitStack() as stack:
        # This test isolates collectives and resume; worker lifecycle has separate tests.
        stack.enter_context(patch('qwen3.vllm_decode.decoder_session', rollout))
        stack.enter_context(patch('qwen35.vllm_decode.training_decoder', rollout))
        stack.enter_context(patch('qwen3.runner.score', side_effect=lambda task, text, gold, **kw: int(text == gold)))
        stack.enter_context(patch('qwen3.scoring.score', side_effect=lambda task, text, gold, **kw: int(text == gold)))
        stack.enter_context(patch('qwen35.cli._grade', side_effect=lambda c,r,text,tokens,*a,**kw: int(text == r['gold'])))
        receipt = result.fit(train, validation, **options)
    weight = (result.bridge.P[0].weight if kind == 'sc' else result.actor.student.weight).detach().clone()
    return receipt, weight


def worker(rank, world, port, root):
    import torch.distributed as td
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    td.init_process_group('gloo', rank=rank, world_size=world, timeout=timedelta(seconds=90))
    try:
        for kind in ('sc', 'tr', 'q35'):
            for objective in ('sft', 'grpo'):
                name = kind + '-' + objective
                path = Path(root) / ('distributed-' + name)
                # Deliberately skew startup: a fast rank must not create step 0
                # while a slower peer is still checking whether this is a resume.
                from qwen3.parallel import sync_parameters
                def delayed_sync(parameters):
                    sync_parameters(parameters)
                    if rank == 1:
                        import time
                        time.sleep(.05)
                with patch('platour_api.sync_parameters', delayed_sync):
                    receipt, weight = fit(kind, objective, path)
                # Completed runs restore selected state without duplicating metrics.
                again, restored = fit(kind, objective, path)
                assert receipt == again and torch.equal(weight, restored)
                torch.save(weight, Path(root) / f'{name}-rank-{rank}.pt')
                if objective == 'grpo':
                    interrupted = Path(root) / ('resume-' + name)
                    try:
                        fit(kind, objective, interrupted, fail_step=1)
                    except RuntimeError as exc:
                        assert 'injected validation' in str(exc)
                    else:
                        raise AssertionError('Expected interruption')
                    resumed, restored = fit(kind, objective, interrupted)
                    torch.testing.assert_close(weight, restored, rtol=0, atol=0)
                    assert resumed['selected_step'] == receipt['selected_step']
        # Fewer questions than ranks, and uneven shards, retain exact coverage.
        for length in (1, 5):
            rows = [dict(index=i) for i in range(length)]
            result = collective_evaluation(rows, 2, Path(root) / 'evaluation',
                lambda local, path: dict(correct=len(local), responses=2*len(local), accuracy=.5))
            assert result['accuracy'] == .5 and result['question_indices'] == list(range(length))
        # PACT DT calibration and both inference interventions use all ranks.
        from pact_check import PACTCheck, build_dummy_statistics, select_irrelevant_donors
        train = [dict(index=i, prompt_token_ids=[1, 10+i, 2, 3]) for i in range(5)]
        validation = [dict(index=20, prompt_token_ids=[1, 80, 2, 3])]
        test = [dict(index=30+i, prompt_token_ids=[1, 90+i, 2, 3]) for i in range(3)]
        def features(prompt):
            return (torch.tensor(prompt[:-1], dtype=torch.float).reshape(1, -1, 1),)
        dummy = build_dummy_statistics(None, train, prefix_ids=[1], suffix_ids=[2],
                                       sample_count=1, features=features)
        assert dummy.prefix_means[0].item() == 1 and dummy.suffix_means[0].item() == 2
        donors = select_irrelevant_donors(train, test, unrelated=lambda a,b: True)
        def inference(system, rows, *, feature_provider):
            def local(part, path):
                for row in part:
                    assert feature_provider(row['prompt_token_ids'])[0].shape == (1, 3, 1)
                return dict(complete=True, correct=len(part), responses=len(part), accuracy=1., backend='vllm')
            return collective_evaluation(rows, 1, Path(root) / 'pact-inference', local)
        report = PACTCheck(None, train, validation, test, evaluator=inference,
                          sender_features=features).run(dummy=dummy, donors=donors,
                            train_sc=None, train_tr=None, samples_per_question=1, controls=('DT','IT'))
        assert report['accuracies'] == {'DT': 1., 'IT': 1.}
        # Qwen3.5's training-final-TEST dispatcher merges each rank's suite files.
        from qwen35.cli import _parallel_test
        case = SimpleNamespace(task='dapo', method='bridge')
        args = SimpleNamespace(output=Path(root) / 'q35-final-test', data_root=Path(root), eval_suites=['aime'])
        all_rows = [dict(index=i) for i in range(5)]
        def local_test(case, args, **kw):
            assert kw['_local']
            rows = dist().local_slice(all_rows)
            args.output.mkdir(parents=True)
            score = dict(complete=True, question_indices=[r['index'] for r in rows],
                correct=len(rows), responses=len(rows)*2, accuracy=.5, samples_per_question=2,
                passed_questions=len(rows), clipped=0, clean80_correct=len(rows), clean80_responses=len(rows)*2)
            (args.output / 'summary.json').write_text(json.dumps(dict(selection={'step':3},results={'aime':score})))
        with patch('qwen35.cli.run_eval', local_test), patch('qwen35.cli.signed_rows', return_value=all_rows):
            summary = _parallel_test(case, args)
        assert summary['results']['aime']['responses'] == 10
        assert summary['results']['aime']['accuracy'] == .5
        assert summary['three_suite_macro'] is None
        try:
            collective_evaluation([dict(index=i) for i in range(2)], 1, root,
                lambda rows, path: (_ for _ in ()).throw(ValueError('worker failed')) if rank else
                    dict(correct=1, responses=1, accuracy=1.))
        except RuntimeError as exc:
            assert 'worker failed' in str(exc)
        else:
            raise AssertionError('Replica failure must propagate')
    finally:
        td.destroy_process_group()


class ParallelTrainingTests(unittest.TestCase):
    def test_launcher_groups_devices_and_preserves_native_contract(self):
        from qwen3.parallel import maybe_launch, configure_group
        process = SimpleNamespace(wait=lambda **kw: 0, poll=lambda: 0)
        with patch.dict(os.environ, {}, clear=True), patch('torch.cuda.device_count', return_value=8), \
             patch('qwen3.parallel.subprocess.Popen', return_value=process) as start, patch('builtins.print'):
            for action, experiment, count, replicas in [('train', 18, 1, 1), ('train', 18, 8, 8),
                    ('train', 15, 8, 4), ('train', 20, 8, 2), ('pact-check', 18, 8, 8)]:
                self.assertTrue(maybe_launch([action, '--experiment', str(experiment), '--n-gpu', str(count)]))
                command = start.call_args.args[0]
                self.assertIn(f'--nproc-per-node={replicas}', command)
                self.assertNotIn('--n-gpu', command)
                groups = json.loads(start.call_args.kwargs['env']['PLATOUR_LAUNCH_GROUPS'])
                self.assertEqual(len(groups), replicas)
                self.assertEqual(sum(map(len, groups)), count)
            with self.assertRaisesRegex(ValueError, 'eight-GPU'):
                maybe_launch(['train', '--experiment', '18', '--n-gpu', '1', '--train-backend', 'historical-grpo'])
            maybe_launch(['train', '--experiment', '11', '--n-gpu', '8', '--train-backend', 'historical-sft'])
            self.assertIn('--nproc-per-node=4', start.call_args.args[0])
            configure_group([['GPU-a', 'GPU-b'], ['GPU-c', 'GPU-d']], 1)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-c,GPU-d')
            self.assertEqual(os.environ['LOCAL_RANK'], '0')
            self.assertEqual(os.environ['PLATOUR_GPU_GROUP'], '2')

    def test_all_api_training_paths_match_single_process_and_resume(self):
        import torch.multiprocessing as mp
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            reference = {}
            for kind in ('sc', 'tr', 'q35'):
                for objective in ('sft', 'grpo'):
                    name = kind + '-' + objective
                    _, reference[name] = fit(kind, objective, Path(tmp) / ('single-' + name))
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            mp.spawn(worker, args=(2, port, tmp), nprocs=2, join=True)
            for name, expected in reference.items():
                weights = [torch.load(Path(tmp) / f'{name}-rank-{rank}.pt', weights_only=True) for rank in range(2)]
                self.assertTrue(torch.equal(*weights), name)
                torch.testing.assert_close(weights[0], expected, rtol=1e-5, atol=1e-7)
                root = Path(tmp) / ('distributed-' + name)
                metrics = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
                self.assertEqual([m['step'] for m in metrics], [1, 2, 3])
                self.assertEqual(json.loads((root / 'protocol.json').read_text())['parallel']['world_size'], 2)
                if name.endswith('grpo'):
                    from safetensors.torch import load_file
                    def terminal(folder):
                        checkpoint = folder / 'checkpoints/step-0003'
                        if name.startswith('sc-'):
                            return torch.load(checkpoint / 'bridge.pt', weights_only=True)['P']['0.weight']
                        return load_file(checkpoint / 'model.safetensors')['weight']
                    final = terminal(root)
                    self.assertGreater(float(final.abs().sum()), 0, name)
                    torch.testing.assert_close(final, terminal(Path(tmp) / ('single-' + name)), rtol=1e-5, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
