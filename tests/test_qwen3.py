"""Unit checks for the published configuration and checkpoint-before-validation rule."""

import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

import platour_exps
from qwen3.bridge import Bridge, layer_map
from qwen3.checkpoints import load_step, save_step
from qwen3.cli import ROOT, audit_inputs, audit_published_weight, select
from qwen3.scoring import evaluation_score, last_boxed, score
from qwen3.vllm_decode import prefix_tensors
from qwen3 import runner


def mock_receiver_decoder(session):
    """CPU stand-in for the external worker; production must call the decoder."""
    from contextlib import nullcontext
    def generate(rows, *, count, cap, sampling, logprobs=False, seed_override=None):
        answers = []
        for row in rows:
            for sample in range(count):
                seed = seed_override if seed_override is not None else 1234 + count * row['index'] + sample
                ids = session.actor.generate(row['prompt_token_ids'], seed=seed, cap=cap, **sampling)
                with torch.no_grad():
                    old = session.actor.response_log_probs(row['prompt_token_ids'], ids).tolist()
                answers.append(dict(index=row['index'], sample_index=sample,
                    response_token_ids=ids, response=session.tokenizer.decode(ids), log_probs=old))
        return answers
    return nullcontext(SimpleNamespace(generate=generate))


class ToyBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.P = nn.ModuleList([nn.Linear(1, 1, bias=False)])


class SFTToyBridge(nn.Module):
    """Tiny SFT bridge: loss depends on a learned projection and the supervised rows."""

    pad_token_id = 0

    def __init__(self):
        super().__init__()
        self.P = nn.ModuleList([nn.Linear(2, 1, bias=False)])

    def batch_response_loss(self, rows, _pad):
        values = torch.stack([self.P[0].weight[0, 0] * row['prompt_token_ids'][1] -
                              self.P[0].weight[0, 1] * len(row['response_token_ids'])
                              for row in rows])
        return (values ** 2).mean() * .01

    def response_log_probs(self, prompt, answer):
        return -(self.P[0].weight[0, 0] * prompt[1] - self.P[0].weight[0, 1] * len(answer)
                 ).abs().reshape(1).repeat(len(answer)) * .01


def _sft_config():
    return {'smoke': True, 'id': 1, 'method': 'sft', 'task': 'gsm8k', 'total_steps': 4, 'steps_per_epoch': 2,
            'batch_size': 4, 'micro_batch': 1, 'learning_rate': .05, 'warmup_steps': 1,
            'eval_samples': 1, 'test_eval_samples': 1, 'response_cap': 2,
            'eval_sampling': {'temperature': 0., 'top_p': 1., 'top_k': -1}}


def _sft_rows(count, offset):
    return [{'index': offset + i, 'prompt_token_ids': [1, offset + i + 2],
             'response_token_ids': [5] * (1 + (offset + i) % 3), 'gold': '1'}
            for i in range(count)]


class SimulatedCrash(Exception):
    pass


def _portable_sft_run(output, *, resume=None, crash_at=None):
    torch.manual_seed(5)
    bridge = SFTToyBridge()
    tested = []
    original = runner.sample_indices

    def indices(count, step, *rest):
        if step == crash_at:
            raise SimulatedCrash(f'process died before update {step}')
        return original(count, step, *rest)
    with patch('qwen3.runner.evaluate_vllm', side_effect=lambda *_a, **kw: tested.append(kw['count']) or
               {'correct': 1, 'responses': 2, 'accuracy': .5}), \
         patch('qwen3.runner.sample_indices', side_effect=indices):
        runner.train(_sft_config(), bridge, None, _sft_rows(8, 0), _sft_rows(3, 50), Path(output),
                     resume=resume, final_test=_sft_rows(2, 90))
    return bridge.P[0].weight.detach().clone(), tested


def _sft_data_parallel_worker(rank, world, port, folder):
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    dist.init_process_group('gloo', rank=rank, world_size=world)
    try:
        weight, tested = _portable_sft_run(Path(folder) / 'run')
        torch.save({'weight': weight, 'tested': tested}, Path(folder) / f'rank-{rank}.pt')
    finally:
        dist.destroy_process_group()


def _published_grpo_validation(config):
    if not config.get('published_grpo_validation'):
        return {}
    return {'validation_rows': runner.read_rows(ROOT / 'data/qwen3' / config['tag'] / 'validation.jsonl')}


class PublicReleaseTests(unittest.TestCase):
    def test_portable_sft_memory_defaults_preserve_math_and_historical_recipes(self):
        from qwen3.cli import portable_training_config
        from training_protocol import qwen3_micro_batch
        for identifier, micro in ((4, 2), (5, 2), (6, 1), (7, 8), (9, 8), (10, 8), (11, 4), (12, 1)):
            original = select(identifier)
            actual = portable_training_config(original)
            self.assertEqual(actual['micro_batch'], micro)
            self.assertEqual({k: v for k, v in actual.items() if k != 'micro_batch'},
                             {k: v for k, v in original.items() if k != 'micro_batch'})
        self.assertEqual(select(11)['micro_batch'], 8)
        self.assertEqual(select(10)['micro_batch'], 8)
        self.assertEqual(qwen3_micro_batch('sft', 'math500'), 1)

    def test_portable_sft_explicit_micro_batch_preserves_recipe_budget(self):
        from qwen3.cli import portable_training_config, main
        original = select(10)
        actual = portable_training_config(original, micro_batch=2)
        self.assertEqual(actual, dict(original, micro_batch=2))
        self.assertEqual(original['micro_batch'], 8)
        for invalid in (0, -1, True, 1.5, 65):
            with self.assertRaisesRegex(ValueError, 'micro-batch'):
                portable_training_config(original, micro_batch=invalid)
        with self.assertRaisesRegex(ValueError, 'portable SFT'):
            portable_training_config(select(16), micro_batch=2)
        for identifier, backend in ((10, 'historical-sft'), (16, 'portable')):
            with patch('sys.argv', ['qwen3', 'train', '--experiment', str(identifier),
                    '--sender', '/sender', '--receiver', '/receiver', '--output', '/unused',
                    '--train-backend', backend, '--micro-batch', '2']):
                with self.assertRaisesRegex(ValueError, 'only available for portable SFT'):
                    main()

    def test_sft_test_clears_gradients_and_restores_signed_state_even_on_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), TemporaryDirectory() as folder:
                root = Path(folder)
                bridge = ToyBridge()
                optimizer = torch.optim.AdamW(bridge.P.parameters(), lr=.01)
                bridge.P[0].weight.sum().backward()
                optimizer.step()
                saved_weight = bridge.P[0].weight.detach().clone()
                saved_moment = optimizer.state[bridge.P[0].weight]['exp_avg'].clone()
                save_step(root, 1, bridge, optimizer)
                config = dict(method='sft', total_steps=1, eval_samples=1,
                              response_cap=2, eval_backend='vllm')
                selection = dict(selected_step=1, policy='final_sft_checkpoint_v1')

                def evaluate(*args, **kwargs):
                    self.assertIsNone(bridge.P[0].weight.grad)
                    self.assertFalse(bridge.training)
                    self.assertTrue(torch.equal(bridge.P[0].weight, saved_weight))
                    self.assertTrue(all(value.device.type == 'cpu'
                        for state in optimizer.state.values() for value in state.values()
                        if isinstance(value, torch.Tensor)))
                    # A failed worker must not lose the committed optimizer/model.
                    optimizer.state.clear()
                    with torch.no_grad():
                        bridge.P[0].weight.add_(1)
                    if fail:
                        raise RuntimeError('worker failed')
                    return dict(complete=True, questions=1, responses=1, accuracy=1.)

                with patch.object(runner, 'evaluate_vllm', side_effect=evaluate):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                            runner.evaluate_selected_once(config, bridge, None, [], [{}],
                                                          root, optimizer, selection=selection)
                    else:
                        runner.evaluate_selected_once(config, bridge, None, [], [{}],
                                                      root, optimizer, selection=selection)
                self.assertTrue(bridge.training)
                self.assertTrue(torch.equal(bridge.P[0].weight, saved_weight))
                self.assertTrue(torch.equal(optimizer.state[bridge.P[0].weight]['exp_avg'],
                                            saved_moment))

    def test_sft_resume_rejects_changed_micro_batch(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / 'run'
            _portable_sft_run(root)
            config = dict(_sft_config(), micro_batch=2)
            with self.assertRaisesRegex(RuntimeError, 'protocol changed'):
                runner.train(config, SFTToyBridge(), None, _sft_rows(8, 0),
                             _sft_rows(2, 10), root,
                             resume=root / 'checkpoints/step-0004')

    def test_vllm_startup_releases_cache_and_records_memory_before_spawning(self):
        from qwen3.vllm_decode import VLLMDecoder
        events = []
        bridge = SimpleNamespace(P=nn.Linear(2, 2), receiver=SimpleNamespace(
            device=torch.device('cuda:0'), config=SimpleNamespace(_name_or_path='/receiver')))
        with TemporaryDirectory() as folder:
            def spawn(command, **kwargs):
                directory = Path(command[-1])
                snapshot = json.loads((directory/'memory_before_worker.json').read_text())
                self.assertEqual(events, ['sync', 'empty_cache'])
                self.assertEqual(snapshot, dict(device='cuda:0', free_bytes=120, total_bytes=140,
                                               allocated_bytes=10, reserved_bytes=11))
                raise RuntimeError('stop after memory check')
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.synchronize', side_effect=lambda device: events.append('sync')), \
                 patch('torch.cuda.empty_cache', side_effect=lambda: events.append('empty_cache')), \
                 patch('torch.cuda.mem_get_info', return_value=(120, 140)), \
                 patch('torch.cuda.memory_allocated', return_value=10), \
                 patch('torch.cuda.memory_reserved', return_value=11), \
                 patch('qwen3.vllm_decode.subprocess.Popen', side_effect=spawn):
                with self.assertRaisesRegex(RuntimeError, 'after memory check'):
                    VLLMDecoder(bridge, Path(folder), max_model_len=3072, worker_python=sys.executable)

    def test_resume_after_crash_between_checkpoints_matches_uninterrupted_run(self):
        with TemporaryDirectory() as folder:
            reference, _ = _portable_sft_run(Path(folder) / 'reference')
            run = Path(folder) / 'crashed'
            with self.assertRaises(SimulatedCrash):
                _portable_sft_run(run, crash_at=4)
            logged = [json.loads(line)['step'] for line in (run / 'metrics.jsonl').open()]
            self.assertEqual(logged, [1, 2, 3])
            self.assertFalse((run / 'checkpoints/step-0003').exists())
            with (run / 'metrics.jsonl').open('a') as stream:
                stream.write('{"step": 4, "train/lo')
            resumed, tested = _portable_sft_run(run, resume=run / 'checkpoints/step-0002')
            self.assertTrue(torch.equal(resumed, reference))
            self.assertEqual([json.loads(line)['step'] for line in (run / 'metrics.jsonl').open()],
                             [1, 2, 3, 4])
            archived = list((run / 'recovery').glob('metrics-before-resume-step-0002-*.jsonl'))
            self.assertEqual(len(archived), 1)
            self.assertTrue(archived[0].read_text().endswith('"train/lo'))
            self.assertEqual(tested, [1])

    def test_vllm_evaluation_dispatches_collectively_under_torchrun(self):
        rows = [{'index': 0, 'prompt_token_ids': [1, 2]}]
        for rank in (0, 1):
            with patch('qwen3.runner.world_size', return_value=2), \
                 patch('qwen3.runner.rank', return_value=rank), \
                 patch('qwen3.parallel.collective_evaluation', return_value={'accuracy': .5}) as collective:
                result = runner.evaluate_vllm(None, None, rows, {'task': 'gsm8k'},
                    count=1, cap=1, output=Path('/unused-test-output'))
                self.assertEqual(result, {'accuracy': .5})
                self.assertEqual(collective.call_args.args[:3], (rows, 1, Path('/unused-test-output')))

    def test_resume_with_a_different_rank_count_names_both_counts(self):
        with TemporaryDirectory() as folder:
            model = ToyBridge()
            optimizer = torch.optim.AdamW(model.P.parameters(), lr=.1)
            signed = save_step(Path(folder), 1, model, optimizer, cuda_devices=())
            with patch('qwen3.checkpoints.world_size', return_value=2):
                with self.assertRaisesRegex(RuntimeError, 'saved with 1, now 2'):
                    load_step(signed, model, optimizer)

    def test_resume_rejects_a_metric_history_with_gaps(self):
        with TemporaryDirectory() as folder:
            run = Path(folder) / 'crashed'
            with self.assertRaises(SimulatedCrash):
                _portable_sft_run(run, crash_at=4)
            lines = (run / 'metrics.jsonl').read_text().splitlines()
            (run / 'metrics.jsonl').write_text('\n'.join([lines[0], lines[2]]) + '\n')
            with self.assertRaisesRegex(RuntimeError, 'contiguous'):
                _portable_sft_run(run, resume=run / 'checkpoints/step-0002')

    def test_sft_final_test_uses_the_selected_eval_backend(self):
        from qwen3.cli import main
        config = select(7)
        rows = [{'index': i, 'prompt_token_ids': [1, 2], 'response_token_ids': [3], 'gold': 'A'}
                for i in range(config['train_examples'])]
        bridge = SimpleNamespace(sender=SimpleNamespace(config=SimpleNamespace(
                                     num_hidden_layers=36, hidden_size=4096)),
                                 receiver=SimpleNamespace(config=SimpleNamespace(
                                     num_hidden_layers=28, hidden_size=2048)))
        for backend, python in (('vllm', '/opt/vllm/bin/python'),):
            argv = ['platour-exps', 'train', '--experiment', '7', '--sender', 'sender',
                    '--receiver', 'receiver', '--train-data', 'train', '--validation-data', 'val',
                    '--test-data', 'test', '--eval-backend', backend, '--output', 'unused']
            if python:
                argv += ['--vllm-python', python]
            with self.subTest(backend=backend), patch('sys.argv', argv), \
                 patch.dict('os.environ', {'WORLD_SIZE': '1'}), \
                 patch('qwen3.cli.audit_inputs') as audit, \
                 patch('qwen3.cli.read_rows', side_effect=(rows, rows[:3], rows[:2])), \
                 patch('platour_exps.require_vllm_python') as require, \
                 patch('qwen3.cli.load_bridge', return_value=(bridge, None)), \
                 patch('qwen3.cli.train') as train:
                main()
                self.assertEqual(set(audit.call_args.args[1]), {'train', 'validation', 'test'})
                submitted = train.call_args.args[0]
                self.assertEqual((submitted['eval_backend'], submitted['vllm_python']),
                                 (backend, python))
                self.assertEqual(len(train.call_args.kwargs['final_test']), 2)
                self.assertEqual(require.called, backend == 'vllm')

    def test_fp8_sender_kernel_is_pinned_and_trusted_without_a_network_lookup(self):
        from transformers.integrations import hub_kernels
        fp8 = SimpleNamespace(quantization_config={'quant_method': 'fp8'})
        trusted = []
        with patch('transformers.AutoConfig.from_pretrained', return_value=fp8), \
             patch('transformers.integrations.finegrained_fp8.load_finegrained_fp8_kernel',
                   side_effect=lambda: trusted.append(hub_kernels.ALLOW_ALL_KERNELS)), \
             patch('qwen3.runner.AutoModelForCausalLM.from_pretrained',
                   side_effect=RuntimeError('stop before loading weights')), \
             patch.dict(hub_kernels._HUB_KERNEL_MAPPING, {}, clear=False):
            with self.assertRaisesRegex(RuntimeError, 'stop before'):
                runner.load_bridge('fp8-sender', 'receiver', 'last_aligned')
            self.assertEqual(trusted, [True])
            self.assertFalse(hub_kernels.ALLOW_ALL_KERNELS)
            self.assertEqual(hub_kernels._HUB_KERNEL_MAPPING['finegrained-fp8'], {
                'repo_id': 'kernels-community/finegrained-fp8',
                'revision': '8c178950e8710e26a2210e4e909cd02dc16c8715'})

    def test_portable_sft_tests_once_at_final_checkpoint(self):
        from qwen3.cli import selected_run_state
        with TemporaryDirectory() as folder:
            weight, tested = _portable_sft_run(Path(folder) / 'run')
            run = Path(folder) / 'run'
            selection = json.loads((run / 'selection.json').read_text())
            candidates = selection['candidates']
            self.assertEqual([row['step'] for row in candidates], [2, 4])
            best = max(candidates, key=lambda row: row['step'])
            self.assertEqual(selection['selected_step'], best['step'])
            self.assertFalse(selection['test_used_for_selection'])
            summary = json.loads((run / 'test/summary.json').read_text())
            self.assertEqual((summary['selected_step'], summary['selection_policy']),
                             (4, 'final_sft_checkpoint_v1'))
            self.assertEqual(tested, [1])
            state, provenance = selected_run_state(run)
            self.assertEqual(provenance['selected_step'], best['step'])
            self.assertEqual(provenance['selected_checkpoint_sha256'], hashlib.sha256(
                (run / f"checkpoints/step-{best['step']:04d}/bridge.pt").read_bytes()).hexdigest())
            terminal = torch.load(run / f"checkpoints/step-{best['step']:04d}/bridge.pt",
                                  weights_only=True)['P']
            self.assertTrue(torch.equal(state['0.weight'], terminal['0.weight']))
            final = json.loads((run / 'checkpoints/step-0004/complete.json').read_text())
            self.assertTrue(final['complete'])
            with torch.no_grad():
                (run / f"checkpoints/step-{best['step']:04d}/bridge.pt").write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'SHA differs'):
                selected_run_state(run)

    def test_sft_final_policy_ignores_lower_earlier_loss_and_native_fallback(self):
        from qwen3.cli import selected_run_state
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for step, nll in ((2, .1), (4, .9)):
                checkpoint = root / 'checkpoints' / f'step-{step:04d}'
                checkpoint.mkdir(parents=True)
                torch.save({'P': {'weight': torch.tensor([float(step)])}}, checkpoint/'bridge.pt')
                checksum = hashlib.sha256((checkpoint/'bridge.pt').read_bytes()).hexdigest()
                marker = dict(complete=True, step=step, metric={'heldout_nll': nll},
                              sha256={'bridge.pt': checksum})
                (checkpoint/'complete.json').write_text(json.dumps(marker))
                (root/'evaluation').mkdir(exist_ok=True)
                (root/'evaluation'/f'step-{step:04d}.json').write_text(
                    json.dumps(dict(step=step, heldout_nll=nll)))
            (root/'complete.json').write_text(json.dumps(dict(complete=True, steps=4)))
            selected = runner.best_sft_checkpoint(root, {'total_steps': 4}, [2, 4])
            self.assertEqual(selected['selected_step'], 4)
            self.assertEqual(selected['policy'], 'final_sft_checkpoint_v1')
            state, provenance = selected_run_state(root)
            self.assertEqual(provenance['selected_step'], 4)
            self.assertEqual(float(state['weight']), 4.)
            (root/'checkpoints/step-0004/complete.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Final SFT checkpoint'):
                selected_run_state(root)

    def test_portable_sft_two_process_data_parallel_matches_one_process(self):
        import socket
        import torch.multiprocessing as mp
        with TemporaryDirectory() as folder:
            reference, _ = _portable_sft_run(Path(folder) / 'reference')
        try:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
        except PermissionError:
            self.skipTest('Sandbox denies sockets required for multiprocess training test')
        with TemporaryDirectory() as folder:
            mp.spawn(_sft_data_parallel_worker, args=(2, port, folder), nprocs=2, join=True)
            ranks = [torch.load(Path(folder) / f'rank-{rank}.pt', weights_only=False)
                     for rank in range(2)]
            summary = json.loads((Path(folder) / 'run/test/summary.json').read_text())
            marker = json.loads((Path(folder) / 'run/checkpoints/step-0004/complete.json').read_text())
        self.assertTrue(torch.equal(ranks[0]['weight'], ranks[1]['weight']))
        torch.testing.assert_close(ranks[0]['weight'], reference, atol=1e-6, rtol=1e-5)
        self.assertEqual(marker['world_size'], 2)
        self.assertEqual(summary['responses'], 2)

    @unittest.skipUnless(os.name == 'posix', 'Native training needs POSIX process groups')
    def test_native_rank_failure_stops_only_its_new_group(self):
        foreign = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(15)'],
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        previous = signal.getsignal(signal.SIGTERM)
        try:
            with TemporaryDirectory() as folder:
                output = Path(folder)
                failure = output / 'smoke'
                failure.mkdir()
                program = ('import json,subprocess,sys,time;from pathlib import Path;'
                    'r=Path(sys.argv[1]);p=subprocess.Popen([sys.executable,"-B","-c",'
                    '"import time;time.sleep(15)"]);'
                    '(r/"child.pid").write_text(str(p.pid));time.sleep(.1);'
                    '(r/"smoke/failure.rank-00.json").write_text(json.dumps({"error":"synthetic"}));'
                    'time.sleep(15)')
                with self.assertRaisesRegex(RuntimeError, 'failure.rank-00'):
                    platour_exps.run_native_training([sys.executable, '-B', '-c', program,
                        str(output)], cwd=ROOT, environment=dict(os.environ), output=output,
                        failure_directory=failure, smoke=True, max_seconds=5,
                        grace_seconds=.2, poll_seconds=.05)
                self.assertIsNone(foreign.poll())
                child = Path('/proc') / (output / 'child.pid').read_text()
                if child.exists():
                    self.assertEqual((child / 'stat').read_text().rsplit(') ', 1)[-1].split()[0], 'Z')
                identity = json.loads((output / 'native_process.json').read_text())
                stopped = json.loads((output / 'native_termination.json').read_text())
                self.assertTrue(identity['new_session'])
                self.assertEqual(stopped['pgid'], identity['pid'])
                self.assertIn('Native rank failure', stopped['reason'])
        finally:
            foreign.terminate()
            foreign.wait(timeout=5)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    @unittest.skipUnless(os.name == 'posix', 'Native training needs POSIX process groups')
    def test_native_supervisor_timeout_and_fresh_output_gate(self):
        with TemporaryDirectory() as folder:
            output = Path(folder)
            with self.assertRaises(TimeoutError):
                platour_exps.run_native_training([sys.executable, '-B', '-c',
                    'import time;time.sleep(15)'], cwd=ROOT, environment=dict(os.environ),
                    output=output, failure_directory=output / 'smoke', smoke=True,
                    max_seconds=.2, grace_seconds=.2, poll_seconds=.05)
            with self.assertRaises(FileExistsError):
                platour_exps.run_native_training([sys.executable, '-B', '-c', 'pass'],
                    cwd=ROOT, environment=dict(os.environ), output=output,
                    failure_directory=output / 'smoke', smoke=True)
        with TemporaryDirectory() as folder:
            output = Path(folder)
            result = platour_exps.run_native_training([sys.executable, '-B', '-c', 'pass'],
                cwd=ROOT, environment=dict(os.environ), output=output,
                failure_directory=output / 'smoke', smoke=True, max_seconds=5,
                grace_seconds=.2, poll_seconds=.05)
            self.assertEqual(result.returncode, 0)
            self.assertFalse((output / 'native_termination.json').exists())

    @unittest.skipUnless(os.name == 'posix', 'Native training needs POSIX process groups')
    def test_native_resume_attempt_preserves_old_logs_and_monitors_new_rank(self):
        with TemporaryDirectory() as folder:
            output = Path(folder)
            failure = output / 'train'
            failure.mkdir()
            (failure / 'failure.rank-00.json').write_text('{"error":"old"}')
            (output / 'train.log').write_text('original-scientific-failure\n')
            (output / 'native_process.json').write_text('{"pid":2147483630,"smoke":false}')
            clean = ('from pathlib import Path;import sys;'
                     'r=Path(sys.argv[1]);(r/"archive").mkdir();'
                     '(r/"train/failure.rank-00.json").rename('
                     'r/"archive/failure.rank-00.json")')
            result = platour_exps.run_native_training(
                [sys.executable, '-B', '-c', clean, str(output)],
                cwd=ROOT, environment=dict(os.environ), output=output,
                failure_directory=failure, smoke=False, resume_step=33,
                max_seconds=5, grace_seconds=.2, poll_seconds=.05)
            self.assertEqual(result.returncode, 0)
            self.assertEqual((output / 'train.log').read_text(),
                             'original-scientific-failure\n')
            self.assertEqual(json.loads((output / 'archive/failure.rank-00.json').read_text()),
                             {'error': 'old'})
            first = output / 'native_attempts/attempt-0002'
            self.assertEqual(json.loads((first / 'native_process.json').read_text())['resume_step'], 33)
            self.assertTrue((first / 'train.log').is_file())
            broken = ('from pathlib import Path;import sys,time;'
                      'r=Path(sys.argv[1]);'
                      '(r/"train/failure.rank-00.json").write_text("new failure");'
                      'time.sleep(10)')
            with self.assertRaisesRegex(RuntimeError, 'failure.rank-00'):
                platour_exps.run_native_training(
                    [sys.executable, '-B', '-c', broken, str(output)],
                    cwd=ROOT, environment=dict(os.environ), output=output,
                    failure_directory=failure, smoke=False, resume_step=33,
                    max_seconds=5, grace_seconds=.2, poll_seconds=.05)
            second = output / 'native_attempts/attempt-0003'
            self.assertTrue((second / 'native_termination.json').is_file())
            self.assertEqual((output / 'train.log').read_text(),
                             'original-scientific-failure\n')

    @unittest.skipUnless(os.name == 'posix', 'Native training needs POSIX process groups')
    def test_native_supervisor_allows_only_explicit_same_mode_smoke_resume(self):
        with TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'smoke').mkdir()
            (output / 'train.log').write_text('original interrupted smoke\n')
            (output / 'native_process.json').write_text('{"pid":2147483630,"smoke":true}')
            kwargs = dict(cwd=ROOT, environment=dict(os.environ), output=output,
                          failure_directory=output / 'smoke', smoke=True,
                          resume_step=1, max_seconds=5, grace_seconds=.2, poll_seconds=.05)
            with self.assertRaisesRegex(ValueError, 'explicitly audited'):
                platour_exps.run_native_training([sys.executable, '-B', '-c', 'pass'],
                                                 **kwargs)
            result = platour_exps.run_native_training(
                [sys.executable, '-B', '-c', 'pass'], allow_smoke_resume=True, **kwargs)
            self.assertEqual(result.returncode, 0)
            self.assertEqual((output / 'train.log').read_text(), 'original interrupted smoke\n')
            self.assertEqual(json.loads((output / 'native_attempts/attempt-0002/'
                                         'native_process.json').read_text())['resume_step'], 1)
            self.assertFalse((output / 'native_attempts/attempt-0002/'
                              'native_termination.json').exists())

    def test_fixed_padded_sft_matches_per_response_loss_and_gradient(self):
        class CausalLayer(nn.Module):
            def forward(self, hidden_states):
                return hidden_states.cumsum(dim=1)

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(16, 4)
                self.layers = nn.ModuleList([CausalLayer()])

            def forward(self, input_ids, **_):
                hidden = self.embed(input_ids)
                states = (hidden,)
                for layer in self.layers:
                    hidden = layer(hidden)
                return SimpleNamespace(hidden_states=states, last_hidden_state=hidden)

        class CausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(num_hidden_layers=1, hidden_size=4,
                                              pad_token_id=0, eos_token_id=0)
                self.model = Encoder()
                self.lm_head = nn.Linear(4, 16, bias=False)

            @property
            def device(self):
                return next(self.parameters()).device

        torch.manual_seed(42)
        bridge = Bridge(CausalLM(), CausalLM(), 'last_aligned')
        nn.init.uniform_(bridge.P[0].weight, -.1, .1)
        rows = [{'prompt_token_ids': [1, 2], 'response_token_ids': [3, 4]},
                {'prompt_token_ids': [5, 6, 7, 8], 'response_token_ids': [9]}]
        tokens = sum(len(row['response_token_ids']) for row in rows)
        reference = sum(-bridge.response_log_probs(row['prompt_token_ids'],
                                                    row['response_token_ids']).sum()
                        for row in rows) / tokens
        reference.backward()
        gradient = bridge.P[0].weight.grad.detach().clone()
        for micro_batch in (1, 2):
            with self.subTest(micro_batch=micro_batch):
                bridge.P[0].weight.grad = None
                padded = reference.detach().new_zeros(())
                for start in range(0, len(rows), micro_batch):
                    chunk = rows[start:start + micro_batch]
                    amount = sum(len(row['response_token_ids']) for row in chunk)
                    value = bridge.batch_response_loss(chunk, pad_token_id=0,
                                                       lm_chunk_tokens=2)
                    (value * (amount / tokens)).backward()
                    padded += value.detach() * (amount / tokens)
                torch.testing.assert_close(padded, reference, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(bridge.P[0].weight.grad, gradient,
                                           atol=1e-6, rtol=1e-6)
        self.assertFalse(any(parameter.grad is not None for model in
                             (bridge.sender, bridge.receiver) for parameter in model.parameters()))

    def test_grpo_backpropagates_each_answer_with_the_batch_objective_gradient(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.P = nn.ModuleList([nn.Linear(2, 1, bias=False)])
                nn.init.constant_(self.P[0].weight, .05)
                self.live = 0
                self.peak = 0

            def generate(self, prompt, *, seed, **_kwargs):
                return [3 + (seed % 3)] * (1 + seed % 2)

            def response_log_probs(self, prompt, answer):
                x = torch.tensor([[float(prompt[-1]), float(len(answer))]] * len(answer))
                if torch.is_grad_enabled():
                    self.live += 1
                    self.peak = max(self.peak, self.live)
                return torch.nn.functional.logsigmoid(self.P[0](x)).squeeze(1) - .1 * torch.arange(len(answer))

        rows = [{'prompt_token_ids': [1, i + 2], 'gold': str(i)} for i in range(3)]
        config = {'task': 'gsm8k', 'group_size': 4, 'response_cap': 8,
                  'train_sampling': {'temperature': 1., 'top_p': 1., 'top_k': -1}}
        rewards = iter([1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0])
        model = Toy()
        original = model.response_log_probs

        def tracked(prompt, answer):
            value = original(prompt, answer)
            if value.requires_grad:
                value.register_hook(lambda grad: setattr(model, 'live', model.live - 1))
            return value
        model.response_log_probs = tracked
        from contextlib import contextmanager
        @contextmanager
        def decoder(bridge, *args, **kwargs):
            def generate(rows, *, count, seed_base, **kwargs):
                result = []
                for row in rows:
                    for sample in range(count):
                        tokens = bridge.generate(row['prompt_token_ids'], seed=seed_base + count*row['index'] + sample)
                        with torch.no_grad():
                            logs = bridge.response_log_probs(row['prompt_token_ids'], tokens).tolist()
                        result.append(dict(index=row['index'], sample_index=sample,
                                           response_token_ids=tokens, log_probs=logs))
                return result
            yield SimpleNamespace(generate=generate)
        config['rollout_output'] = '/unused'
        with patch('qwen3.vllm_decode.VLLMDecoder', decoder), patch.object(runner, 'score', side_effect=lambda *_a, **_k: next(rewards)):
            loss, correct = runner.grpo_update(model, SimpleNamespace(decode=str), rows,
                                               [0, 1, 2], config, 1, 3)
        self.assertEqual(model.peak, 1)
        gradient = model.P[0].weight.grad.clone()
        reference = Toy()
        rewards = iter([1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0])
        total = 0
        for index in (0, 1, 2):
            prompt = rows[index]['prompt_token_ids']
            answers = [reference.generate(prompt, seed=42 + 1000 + 4 * index + s) for s in range(4)]
            values = torch.tensor([float(next(rewards)) for _ in answers])
            advantage = (values - values.mean()) / (values.std(correction=1) + 1e-6)
            for answer, a in zip(answers, advantage):
                if float(a):
                    with torch.no_grad():
                        old = reference.response_log_probs(prompt, answer)
                    total = total + runner.policy_loss(reference.response_log_probs(prompt, answer), old, float(a))
        (total / 12).backward()
        self.assertTrue(torch.allclose(gradient, reference.P[0].weight.grad, atol=1e-7, rtol=1e-6))
        self.assertAlmostEqual(loss, float(total / 12), places=6)
        self.assertEqual(correct, 6)

    def test_portable_sft_uses_the_reference_layout_for_any_process_count(self):
        from qwen3.aligned_sft import local_chunks
        rows = [{'prompt_token_ids': [1] * (1 + i % 5), 'response_token_ids': [2] * (1 + i % 3)}
                for i in range(64)]
        batch = list(range(63, -1, -1))
        canonical = [chunk for share in range(4) for chunk in local_chunks(batch, rows, share, 4, 8)]
        for world in (1, 2, 3, 4, 8, 16):
            assigned = []
            for member in range(world):
                with patch.object(runner, 'world_size', return_value=world), \
                     patch.object(runner, 'rank', return_value=member):
                    mine = runner.reference_sft_chunks(batch, rows, 8)
                if world == 4:
                    self.assertEqual(mine, local_chunks(batch, rows, member, 4, 8))
                assigned.extend(mine)
            self.assertEqual(sorted(map(tuple, assigned)), sorted(map(tuple, canonical)))

        class Toy(nn.Module):
            pad_token_id = 0

            def __init__(self):
                super().__init__()
                self.P = nn.ModuleList([nn.Linear(2, 1, bias=False), nn.Linear(1, 1, bias=False)])
                nn.init.constant_(self.P[0].weight, .3)
                nn.init.constant_(self.P[1].weight, -.2)

            def batch_response_loss(self, batch, pad):
                values = torch.tensor([[len(r['prompt_token_ids']), len(r['response_token_ids'])]
                                       for r in batch], dtype=torch.float32)
                return (self.P[0](values).squeeze(1) * self.P[1].weight[0, 0]).square().mean()

        model = Toy()
        expected_sum = expected_tokens = 0.
        for chunk in canonical:
            micro = [rows[i] for i in chunk]
            n = sum(len(r['response_token_ids']) for r in micro)
            expected_sum += float(model.batch_response_loss(micro, 0).double()) * n
            expected_tokens += n
        value, tokens = runner.heldout_nll(model, [rows[i] for i in batch], 8)
        self.assertEqual(tokens, expected_tokens)
        self.assertAlmostEqual(value, expected_sum / expected_tokens, places=12)
        optimizer = torch.optim.AdamW(model.P.parameters(), lr=.1)
        total, count = runner.sft_update(model, optimizer, rows, batch, 8, int(expected_tokens))
        self.assertEqual(count, expected_tokens)
        self.assertAlmostEqual(total / count, expected_sum / expected_tokens, places=12)
        flat = torch.cat([p.grad.reshape(-1) for p in model.P.parameters()]).clone()
        norm = runner.clip_flat_gradients(model.P.parameters())
        self.assertTrue(torch.equal(norm, torch.linalg.vector_norm(flat)))
        clipped = torch.cat([p.grad.reshape(-1) for p in model.P.parameters()])
        self.assertTrue(torch.equal(clipped, flat * (1. / (norm + 1e-6)).clamp(max=1.)))

    def test_padded_sft_three_cpu_updates_match_signed_resume(self):
        class PaddedToy(ToyBridge):
            pad_token_id = 0

            def __init__(self):
                super().__init__()
                nn.init.constant_(self.P[0].weight, .1)

            def batch_response_loss(self, batch, pad_token_id):
                self.assert_pad(pad_token_id)
                targets = self.P[0].weight.new_tensor(
                    [row['prompt_token_ids'][0] / 10 for row in batch])
                return (self.P[0].weight.view(1) - targets).square().mean()

            @staticmethod
            def assert_pad(pad_token_id):
                if pad_token_id != 0:
                    raise ValueError('Unexpected signed pad token')

        rows = [{'prompt_token_ids': [1], 'response_token_ids': [3, 4]},
                {'prompt_token_ids': [5], 'response_token_ids': [6]},
                {'prompt_token_ids': [7], 'response_token_ids': [8]}]
        schedule = ((0, 1), (2,), (1, 2))

        def run(resume, directory=None):
            model = PaddedToy()
            optimizer = torch.optim.AdamW(model.P.parameters(), lr=.01)
            outcomes = []
            for step, indices in enumerate(schedule, 1):
                tokens = sum(len(rows[index]['response_token_ids']) for index in indices)
                loss, actual = runner.sft_update(model, optimizer, rows, list(indices), 2,
                                                 tokens)
                self.assertEqual(actual, tokens)
                self.assertTrue(torch.isfinite(torch.tensor(loss)))
                optimizer.step()
                outcomes.append(float(model.P[0].weight.detach()))
                if resume and step == 2:
                    signed = save_step(directory, step, model, optimizer)
                    model = PaddedToy()
                    optimizer = torch.optim.AdamW(model.P.parameters(), lr=.01)
                    self.assertEqual(load_step(signed, model, optimizer), step)
            return outcomes

        with TemporaryDirectory() as folder:
            self.assertEqual(run(False), run(True, Path(folder) / 'signed'))

    def test_historical_sft_private_resume_recovers_history_and_optimizer(self):
        from qwen3.aligned_sft import reconcile_sft_history

        def make_state():
            model = ToyBridge()
            nn.init.constant_(model.P[0].weight, .1)
            optimizer = torch.optim.AdamW(model.P.parameters(), lr=.01,
                                           weight_decay=0.)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: min((step + 1) / 2, 1.))
            return model, optimizer, scheduler

        def update(model, optimizer, scheduler, number):
            optimizer.zero_grad(set_to_none=True)
            loss = (model.P[0].weight.view(()) - number / 10).square()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return {'step': number, 'train_nll': float(loss.detach())}

        control, original_optimizer, original_scheduler = make_state()
        for step in (1, 2, 3):
            update(control, original_optimizer, original_scheduler, step)
        with TemporaryDirectory() as folder:
            output = Path(folder)
            model, optimizer, scheduler = make_state()
            save_step(output, 0, model, optimizer, scheduler, cuda_devices=())
            metric = update(model, optimizer, scheduler, 1)
            (output / 'metrics.jsonl').write_text(json.dumps(metric) + '\n')
            last = update(model, optimizer, scheduler, 2)
            checkpoint = save_step(output, 2, model, optimizer, scheduler,
                                   cuda_devices=(), record=last)
            with (output / 'metrics.jsonl').open('ab') as stream:
                stream.write(b'{"step":3')
            restored, optim, schedule = make_state()
            self.assertEqual(load_step(checkpoint, restored, optim, schedule), 2)
            recovered = reconcile_sft_history(output, checkpoint)
            self.assertEqual([row['step'] for row in recovered], [1, 2])
            self.assertEqual(recovered[1], last)
            self.assertEqual(len(list((output / 'recovery').glob('metrics-before-resume-*'))), 1)
            update(restored, optim, schedule, 3)
            torch.testing.assert_close(restored.P[0].weight, control.P[0].weight,
                                       atol=0, rtol=0)
            for key in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(optim.state_dict()['state'][0][key],
                                           original_optimizer.state_dict()['state'][0][key],
                                           atol=0, rtol=0)
            self.assertEqual(schedule.state_dict()['last_epoch'],
                             original_scheduler.state_dict()['last_epoch'])
            with (checkpoint / 'training.pt').open('ab') as stream:
                stream.write(b'corrupted-state')
            with self.assertRaisesRegex(ValueError, 'SHA differs'):
                load_step(checkpoint, restored, optim, schedule)

    def test_historical_sft_dispatch_preserves_micro_accumulation(self):
        from qwen3.cli import main
        import qwen3.aligned_sft

        for identifier, micro, accumulation in ((7, 8, 2), (4, 2, 8), (12, 1, 16)):
            config = select(identifier)
            rows = [{'prompt_token_ids': [1, 2], 'response_token_ids': [3]}] * config['train_examples']
            with self.subTest(identifier=identifier), TemporaryDirectory() as folder, \
                 patch.dict('os.environ', {'WORLD_SIZE': '4', 'RANK': '0',
                                           'LOCAL_RANK': '0'}), \
                 patch('sys.argv', ['platour-exps', 'train', '--experiment',
                     str(identifier), '--train-backend', 'historical-sft', '--smoke',
                     '--sender', 'sender', '--receiver', 'receiver',
                     '--train-data', 'signed-train', '--validation-data', 'signed-val',
                     '--output', str(Path(folder) / 'fresh')]), \
                 patch('qwen3.cli.audit_inputs'), \
                 patch('qwen3.cli.read_rows', side_effect=(rows, rows)), \
                 patch('importlib.util.find_spec', return_value=True), \
                 patch('qwen3.aligned_sft.train_aligned') as historical:
                main()
                historical.assert_called_once()
                submitted = historical.call_args.args[0]
                self.assertEqual((submitted['micro_batch'],
                                  submitted['gradient_accumulation_steps'],
                                  historical.call_args.kwargs['micro_batch']),
                                 (micro, accumulation, micro))
                self.assertTrue(submitted['smoke'])
                self.assertEqual(4 * micro * accumulation, submitted['batch_size'])
        with patch.dict('os.environ', {'WORLD_SIZE': '1'}), \
             patch('sys.argv', ['platour-exps', 'train', '--experiment', '7',
                 '--train-backend', 'historical-sft', '--sender', 'sender',
                 '--receiver', 'receiver', '--train-data', 'missing',
                 '--validation-data', 'missing', '--output', 'missing']):
            with self.assertRaisesRegex(RuntimeError, 'four ranks'):
                main()

    def test_fp8_sft_reduced_micro_is_explicit_smoke_only_and_global_batch_unchanged(self):
        from qwen3.cli import main
        import qwen3.aligned_sft

        config = select(2)
        rows = [{'prompt_token_ids': [1, 2], 'response_token_ids': [3]}] * config['train_examples']
        with TemporaryDirectory() as folder, patch.dict('os.environ', {
                'WORLD_SIZE': '4', 'RANK': '0', 'LOCAL_RANK': '0',
                'PLATOUR_SFT_FP8_SMOKE_MICRO_BATCH': '2'}), \
             patch('qwen3.cli.audit_inputs'), \
             patch('qwen3.cli.read_rows', side_effect=(rows, rows)), \
             patch('importlib.util.find_spec', return_value=True), \
             patch('qwen3.aligned_sft.train_aligned') as train:
            argv = ['platour-exps', 'train', '--experiment', '2',
                    '--train-backend', 'historical-sft', '--smoke',
                    '--sender', 'teacher', '--receiver', 'receiver',
                    '--train-data', 'signed-train', '--validation-data', 'signed-val',
                    '--output', str(Path(folder) / 'new-run')]
            with patch('sys.argv', argv):
                main()
            train.assert_called_once()
            chosen = train.call_args.args[0]
            self.assertEqual((chosen['micro_batch'], chosen['gradient_accumulation_steps'],
                              chosen['batch_size']), (2, 8, 64))
            self.assertEqual(chosen['smoke_fp8_memory_override']['original_micro_batch'], 8)
            self.assertTrue(chosen['smoke_fp8_memory_override']['global_batch_size_unchanged'])
            with patch('sys.argv', [value for value in argv if value != '--smoke']):
                with self.assertRaisesRegex(ValueError, 'smoke-only'):
                    main()
            self.assertFalse((Path(folder) / 'new-run').exists())

    def test_historical_sft_cli_only_resumes_its_signed_output(self):
        from qwen3.cli import main
        import qwen3.aligned_sft

        config = select(7)
        rows = [{'prompt_token_ids': [1, 2], 'response_token_ids': [3]}] * config['train_examples']
        with TemporaryDirectory() as folder:
            output = Path(folder) / 'native'
            checkpoint = output / 'checkpoints/step-0001'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'complete.json').write_text('{"complete":true}')
            argv = ['platour-exps', 'train', '--experiment', '7',
                    '--train-backend', 'historical-sft', '--smoke',
                    '--sender', 'teacher', '--receiver', 'receiver',
                    '--train-data', 'signed-train', '--validation-data', 'signed-val',
                    '--output', str(output), '--resume', str(checkpoint)]
            with patch.dict('os.environ', {'WORLD_SIZE': '4', 'RANK': '0',
                                           'LOCAL_RANK': '0'}), \
                 patch('sys.argv', argv), patch('qwen3.cli.audit_inputs'), \
                 patch('qwen3.cli.read_rows', side_effect=(rows, rows)), \
                 patch('qwen3.aligned_sft.train_aligned') as historical:
                main()
                self.assertEqual(historical.call_args.kwargs['resume'], checkpoint)
            foreign = Path(folder) / 'other-run/checkpoints/step-0001'
            foreign.mkdir(parents=True)
            (foreign / 'complete.json').write_text('{"complete":true}')
            with patch.dict('os.environ', {'WORLD_SIZE': '4', 'RANK': '0',
                                           'LOCAL_RANK': '0'}), \
                 patch('sys.argv', argv[:-1] + [str(foreign)]), \
                 patch('qwen3.cli.audit_inputs'):
                with self.assertRaisesRegex(ValueError, 'this output root'):
                    main()

    def test_native_grpo_bundle_and_holdout_match_portable_policy(self):
        from qwen3.native_grpo import prepare, verify_bundle
        from qwen3.native_grpo_validation import verify_inputs

        self.assertEqual(len(verify_bundle()), 222)
        # (train, validation, test, val samples, test samples); GSM8K uses
        # avg@4 validation and avg@2 final TEST.
        for identifier, expected in ((13, (1680, 210, 210, 8, 8)),
                                     (14, (6154, 1319, 1319, 4, 2)),
                                     (15, (7000, 500, 500, 4, 4)),
                                     (16, (1680, 210, 210, 8, 8)),
                                     (17, (6154, 1319, 1319, 4, 2)),
                                     (18, (7000, 500, 500, 4, 4))):
            config = select(identifier)
            base = ROOT / 'data/qwen3' / config['tag']
            training = runner.read_rows(base / 'train.jsonl')
            test = runner.read_rows(base / 'test.jsonl')
            options = {'validation_rows': runner.read_rows(base / 'validation.jsonl')} if config.get('published_grpo_validation') else {}
            if options and config['task'] == 'gsm8k':
                expected = (6973, 500, 1319, 4, 2)
            portable_train, portable_val, portable_config, portable_split = (
                runner.prepare_grpo_holdout(training, test, config, **options))
            with self.subTest(identifier=identifier), TemporaryDirectory() as folder, \
                 patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}):
                root = Path(folder)
                models = root / 'models'
                output = root / 'native'
                prepare(config, models / config['sender'], models / 'Qwen3-1.7B',
                        training, test, output, smoke=True, **options)
                policy = json.loads((output / 'protocol.json').read_text())
                split = json.loads((output / 'validation_split.json').read_text())
                verify_inputs(output, policy)
                patch_record = json.loads((output / 'native_resume_source_patch.json').read_text())
                original_script = (Path(__file__).resolve().parents[1] /
                                   'qwen3/native_reference/overlays' / f'{identifier:02d}' /
                                   patch_record['source'])
                active_script = output / 'reference_source' / patch_record['source']
                self.assertEqual(patch_record['original_scientific_sha256'],
                                 hashlib.sha256(original_script.read_bytes()).hexdigest())
                self.assertEqual(patch_record['active_sha256'],
                                 hashlib.sha256(active_script.read_bytes()).hexdigest())
                if identifier == 13:
                    core_name = 'recipe/llm_communicate/qwen35_bridge_opd_core.py'
                    original_core = ROOT / 'qwen3/native_reference/overlays/13' / core_name
                    active_core = output / 'reference_source' / core_name
                    marker = json.loads((output / 'zero_bridge_initialization_patch.json').read_text())
                    self.assertEqual(marker['original_core_sha256'],
                                     hashlib.sha256(original_core.read_bytes()).hexdigest())
                    self.assertEqual(marker['active_core_sha256'],
                                     hashlib.sha256(active_core.read_bytes()).hexdigest())
                    self.assertIn('p.zero_()', active_core.read_text())
                    self.assertNotIn('p.uniform_(-bound, bound, generator=rng)', active_core.read_text())
                    self.assertIn("initialization='exact-zero linear bridge'", active_script.read_text())
                    self.assertEqual(policy['initialization'], 'exact-zero linear bridge')
                    self.assertEqual(json.loads((output / 'original_protocol.json').read_text())
                                     ['initialization'], policy['historical_initialization'])
                    self.assertIn('uniform +/-1e-4', policy['historical_initialization'])
                    (output / 'native_process.json').write_text(json.dumps(
                        {'pid': 2147483630, 'smoke': True}))
                    seal = output / 'zero_bridge_initialization_patch.json'
                    original_seal = seal.read_text()
                    seal.write_text(json.dumps({**marker, 'active_core_sha256': '0' * 64}))
                    from qwen3.native_grpo import audit_resume
                    with self.assertRaisesRegex(ValueError, 'zero bridge source'):
                        audit_resume(output, 1)
                    seal.write_text(original_seal)
                frozen_text, active_text = original_script.read_text(), active_script.read_text()
                first = 'advantages,group_stats=group_advantages(scored,group_size)'
                last = 'optimizer.step()'
                self.assertEqual(frozen_text[frozen_text.index(first):
                                             frozen_text.index(last, frozen_text.index(first)) + len(last)],
                                 active_text[active_text.index(first):
                                             active_text.index(last, active_text.index(first)) + len(last)])
                self.assertTrue(policy['smoke_resume_checkpoints'])
                self.assertEqual((split['train_count'], split['validation_count'],
                                  split['test_count'], policy['eval_samples'],
                                  portable_config['test_eval_samples']), expected)
                self.assertEqual(policy['eval_samples'], portable_config['eval_samples'])
                self.assertEqual(policy['eval_steps'], portable_config['eval_steps'])
                self.assertEqual(split['train_source_indices'],
                                 portable_split['train_source_indices'])
                self.assertEqual(split['validation_source_indices'],
                                 portable_split['validation_source_indices'])
                self.assertEqual(policy['total_steps'], portable_config['total_steps'])
                self.assertEqual(json.loads((output / 'test_prompts.json').read_text()), test)
                self.assertFalse((output / 'train/metrics.jsonl').exists())
                with self.assertRaises(FileExistsError):
                    prepare(config, models / config['sender'], models / 'Qwen3-1.7B',
                            training, test, output, smoke=True, **_published_grpo_validation(config))

    def test_signed_native_grpo_resume_rejects_corrupt_optimizer_and_restores_bridge(self):
        from qwen3.native_grpo import (archive_resume_tail, audit_resume, prepare,
                                       restore_bridge_and_optimizer, sign_checkpoint)

        config = select(16)
        data = ROOT / 'data/qwen3' / config['tag']
        train = runner.read_rows(data / 'train.jsonl')
        test = runner.read_rows(data / 'test.jsonl')
        with TemporaryDirectory() as temporary, patch.dict('os.environ', {
                'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}):
            base = Path(temporary)
            root = base / 'run'
            prepare(config, base / 'models/Qwen3-8B', base / 'models/Qwen3-1.7B',
                    train, test, root, smoke=True, **_published_grpo_validation(config))
            (root / 'native_process.json').write_text(json.dumps({'pid': 2147483630, 'smoke': True}))
            layer = nn.Linear(1, 1, bias=False)
            nn.init.constant_(layer.weight, .1)
            bridge = SimpleNamespace(maps=nn.ModuleList([layer]))
            weight_digest = lambda value: hashlib.sha256(
                value.maps[0].weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            baseline = weight_digest(bridge)
            run = root / 'smoke'
            run.mkdir()
            (run / 'initialization.json').write_text(json.dumps({'bridge_state_sha256': baseline}))
            opt = torch.optim.AdamW(bridge.maps.parameters(), lr=.001)
            opt.zero_grad()
            (layer.weight - .4).square().sum().backward()
            opt.step()
            after = weight_digest(bridge)
            state = run / 'checkpoints/step-0001'
            state.mkdir(parents=True)
            torch.save({'P': bridge.maps.state_dict()}, state / 'bridge.pt')
            torch.save({'step': 1, 'initial_sha256': baseline,
                        'optimizer': opt.state_dict()}, state / 'optimizer.pt')
            (state / 'checkpoint.json').write_text(json.dumps({
                'step': 1, 'bridge_sha256': hashlib.sha256((state / 'bridge.pt').read_bytes()).hexdigest(),
                'bridge_state_sha256': after, 'initial_bridge_state_sha256': baseline}))
            for rank in range(8):
                torch.save({'cpu_rng': torch.get_rng_state(),
                            'cuda_rng': torch.get_rng_state()}, state / f'rng.rank-{rank:02d}.pt')
            sign_checkpoint(state, 1, 8)
            (run / 'metrics.jsonl').write_text(json.dumps({'step': 1, 'grad_norm': .3}) + '\n')
            self.assertEqual(audit_resume(root, 1), state)
            restored = SimpleNamespace(maps=nn.ModuleList([nn.Linear(1, 1, bias=False)]))
            nn.init.constant_(restored.maps[0].weight, .1)
            new_optimizer = torch.optim.AdamW(restored.maps.parameters(), lr=.001)
            self.assertEqual(restore_bridge_and_optimizer(
                state, 1, restored, new_optimizer, baseline, weight_digest), after)
            torch.testing.assert_close(restored.maps[0].weight, bridge.maps[0].weight,
                                       atol=0, rtol=0)
            (root / 'phase.json').write_text('{"phase":"failed"}')
            (run / 'workers/rank-00/vllm').mkdir(parents=True)
            (run / 'workers/rank-00/vllm/ready.json').write_text('{"stale":true}')
            (run / 'rollouts/step-0002').mkdir(parents=True)
            with (run / 'metrics.jsonl').open('ab') as stream:
                stream.write(b'{"step":2')
            archived = archive_resume_tail(root, 1)
            self.assertTrue((archived / 'workers/rank-00/vllm/ready.json').is_file())
            self.assertTrue((archived / 'rollouts/step-0002').is_dir())
            self.assertIn(b'{"step":2', (archived / 'metrics-before-resume.jsonl').read_bytes())
            self.assertEqual([json.loads(x)['step'] for x in
                              (run / 'metrics.jsonl').read_text().splitlines()], [1])
            self.assertEqual(audit_resume(root, 1), state)
            scientific = root / 'protocol.json'
            original_protocol = scientific.read_text()
            changed = json.loads(original_protocol)
            changed['max_response_length'] -= 1
            scientific.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'reward, model or sampling recipe'):
                audit_resume(root, 1)
            scientific.write_text(original_protocol)
            with self.assertRaisesRegex(ValueError, 'already signed'):
                sign_checkpoint(state, 1, 8)
            with (state / 'optimizer.pt').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'optimizer or rank RNG file SHA'):
                audit_resume(root, 1)

    def test_native_grpo_cli_only_resumes_signed_same_root_and_mode(self):
        from qwen3.cli import main
        from qwen3.native_grpo import prepare

        config = select(16)
        base = ROOT / 'data/qwen3' / config['tag']
        with TemporaryDirectory() as folder, patch.dict('os.environ', {
                'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}):
            root = Path(folder) / 'native'
            models = Path(folder) / 'models'
            sender, receiver = models / config['sender'], models / 'Qwen3-1.7B'
            prepare(config, sender, receiver, runner.read_rows(base / 'train.jsonl'),
                    runner.read_rows(base / 'test.jsonl'), root, smoke=True, **_published_grpo_validation(config))
            valid = root / 'smoke/checkpoints/step-0001'
            arguments = ['platour-exps', 'train', '--experiment', '16',
                         '--sender', str(sender), '--receiver', str(receiver),
                         '--train-data', str(base / 'train.jsonl'),
                         '--test-data', str(base / 'test.jsonl'),
                         '--output', str(root), '--train-backend', 'historical-grpo',
                         '--smoke', '--resume', str(valid)]
            with patch('sys.argv', arguments), \
                 patch('platour_exps.require_vllm_python'), \
                 patch('qwen3.native_grpo.audit_resume', return_value=valid) as audited, \
                 patch('qwen3.native_grpo.execute') as executed:
                main()
                audited.assert_called_once_with(root, 1)
                executed.assert_called_once_with(root, resume_step=1)
            with patch('sys.argv', [*arguments[:-1], str(Path(folder) / 'external-step-0001')]), \
                 patch('platour_exps.require_vllm_python'):
                with self.assertRaisesRegex(ValueError, 'signed step in this output'):
                    main()
            with patch('sys.argv', [a for a in arguments if a != '--smoke']), \
                 patch('platour_exps.require_vllm_python'):
                with self.assertRaisesRegex(ValueError, 'signed step in this output'):
                    main()

    def test_native_grpo_formal_boundary_requires_complete_heldout_history(self):
        from qwen3.native_grpo import audit_resume, prepare, sign_checkpoint

        config = select(16)
        data = ROOT / 'data/qwen3' / config['tag']
        with TemporaryDirectory() as temporary, patch.dict('os.environ', {
                'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}):
            base = Path(temporary)
            root = base / 'formal'
            prepare(config, base / 'models/Qwen3-8B', base / 'models/Qwen3-1.7B',
                    runner.read_rows(data / 'train.jsonl'), runner.read_rows(data / 'test.jsonl'),
                    root, smoke=False, **_published_grpo_validation(config))
            protocol = json.loads((root / 'protocol.json').read_text())
            boundary = protocol['eval_steps'][0]
            self.assertEqual(boundary, protocol['checkpoint_steps'][0])
            self.assertGreater(boundary, 1)
            (root / 'native_process.json').write_text('{"pid":2147483630,"smoke":false}')
            run = root / 'train'
            run.mkdir()
            (run / 'initialization.json').write_text('{"bridge_state_sha256":"synthetic-zero"}')
            state = run / 'checkpoints' / f'step-{boundary:04d}'
            state.mkdir(parents=True)
            torch.save({'P': {'weight': torch.ones(1)}}, state / 'bridge.pt')
            torch.save({'step': boundary, 'initial_sha256': 'synthetic-zero',
                        'optimizer': {'state': {}, 'param_groups': []}}, state / 'optimizer.pt')
            digest = hashlib.sha256((state / 'bridge.pt').read_bytes()).hexdigest()
            (state / 'checkpoint.json').write_text(json.dumps({
                'step': boundary, 'bridge_sha256': digest,
                'initial_bridge_state_sha256': 'synthetic-zero',
                'bridge_state_sha256': 'synthetic-after'}))
            for rank in range(8):
                torch.save({'cpu_rng': torch.get_rng_state(),
                            'cuda_rng': torch.get_rng_state()}, state / f'rng.rank-{rank:02d}.pt')
            sign_checkpoint(state, boundary, 8)
            (run / 'metrics.jsonl').write_text(''.join(json.dumps({'step': step,
                'grad_norm': 1. if step == 1 else 0.}) + '\n'
                for step in range(1, boundary + 1)))
            samples = protocol['eval_questions'] * protocol['eval_samples']
            for checkpoint in (0, boundary):
                destination = run / 'evaluation' / f'step-{checkpoint:04d}'
                destination.mkdir(parents=True)
                (destination / 'summary.json').write_text(json.dumps({
                    'complete': True, 'step': checkpoint, 'evaluation_role': 'validation',
                    'dataset_sha256': protocol['eval_prompts_sha256'],
                    'questions': protocol['eval_questions'], 'responses': samples,
                    'scored_responses': samples, 'gold_missing': 0, 'scoring_timeouts': 0,
                    'bridge_state_sha256': ('synthetic-after' if checkpoint else 'synthetic-zero')}))
            self.assertEqual(audit_resume(root, boundary), state)
            with self.assertRaisesRegex(ValueError, 'signed checkpoint boundary'):
                audit_resume(root, boundary - 1)
            first = run / 'evaluation/step-0000/summary.json'
            summary = json.loads(first.read_text())
            summary['evaluation_role'] = 'test'
            first.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'held-out validation is incomplete'):
                audit_resume(root, boundary)

    def test_fp8_native_grpo_memory_workaround_is_smoke_only_and_audited(self):
        from qwen3.native_grpo import prepare

        config = select(13)
        base = ROOT / 'data/qwen3' / config['tag']
        training = runner.read_rows(base / 'train.jsonl')
        test = runner.read_rows(base / 'test.jsonl')
        with TemporaryDirectory() as folder, patch.dict('os.environ', {
                'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
                'PLATOUR_NATIVE_VLLM_FRACTION': '0.075'}):
            output = Path(folder) / 'valid'
            models = Path(folder) / 'models'
            prepare(config, models / config['sender'], models / 'Qwen3-1.7B',
                    training, test, output, smoke=True, **_published_grpo_validation(config))
            current = json.loads((output / 'protocol.json').read_text())
            original = json.loads((output / 'original_protocol.json').read_text())
            self.assertEqual(current['vllm_memory_fraction'], .075)
            self.assertEqual(original['vllm_memory_fraction'], .10)
            self.assertEqual(current['smoke_decoder_memory_override']['original_fraction'], .10)
            with self.assertRaisesRegex(ValueError, 'FP8 smoke-only'):
                prepare(config, models / config['sender'], models / 'Qwen3-1.7B',
                        training, test, Path(folder) / 'formal-forbidden', smoke=False, **_published_grpo_validation(config))
            self.assertFalse((Path(folder) / 'formal-forbidden').exists())
            with self.assertRaisesRegex(ValueError, 'FP8 smoke-only'):
                prepare(select(16), models / 'Qwen3-8B', models / 'Qwen3-1.7B',
                        training, test, Path(folder) / 'dense-forbidden', smoke=True, **_published_grpo_validation(select(16)))
            self.assertFalse((Path(folder) / 'dense-forbidden').exists())

    def test_fp8_math_secondary_decoder_placement_is_smoke_only(self):
        from qwen3.native_grpo import prepare, ASSETS, sha256, resume_source_text

        config = select(15)
        base = ROOT / 'data/qwen3' / config['tag']
        train = runner.read_rows(base / 'train.jsonl')
        test = runner.read_rows(base / 'test.jsonl')
        with TemporaryDirectory() as folder, patch.dict('os.environ', {
                'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
                'PLATOUR_NATIVE_VLLM_FRACTION': '0.075',
                'PLATOUR_NATIVE_VLLM_GPU_PLACEMENT': 'secondary'}):
            root = Path(folder)
            models = root / 'models'
            target = root / 'qualified'
            prepare(config, models / config['sender'], models / 'Qwen3-1.7B',
                    train, test, target, smoke=True, **_published_grpo_validation(config))
            protocol = json.loads((target / 'protocol.json').read_text())
            old = json.loads((target / 'original_protocol.json').read_text())
            self.assertEqual(protocol['decoder_gpu_ids'], [1, 3, 5, 7])
            self.assertEqual(old['decoder_gpu_ids'], [0, 2, 4, 6])
            self.assertEqual(protocol['vllm_memory_fraction'], .075)
            manifest = json.loads((target / 'smoke_source_patch.json').read_text())
            self.assertEqual(manifest['substitutions'], 2)
            rel = Path(manifest['source'])
            expected = (ASSETS / 'overlays/15' / rel).read_text()
            patched = (target / 'reference_source' / rel).read_text()
            placed = expected.replace(
                "assert physical[device_index]==protocol['decoder_gpu_ids'][local]",
                "assert physical[device_index+1]==protocol['decoder_gpu_ids'][local]").replace(
                "gpu=physical[device_index],memory_fraction=protocol['vllm_memory_fraction']",
                "gpu=protocol['decoder_gpu_ids'][local],memory_fraction=protocol['vllm_memory_fraction']")
            self.assertEqual(sha256(ASSETS / 'overlays/15' / rel),
                             manifest['original_sha256'])
            self.assertEqual(hashlib.sha256(placed.encode()).hexdigest(),
                             manifest['before_resume_sha256'])
            self.assertEqual(patched, resume_source_text(placed, 15))
            self.assertEqual(sha256(target / 'reference_source' / rel),
                             manifest['active_sha256'])
            with self.assertRaisesRegex(ValueError, 'only for ID15'):
                prepare(config, models / config['sender'], models / 'Qwen3-1.7B',
                        train, test, root / 'formal-forbidden', smoke=False, **_published_grpo_validation(config))
            self.assertFalse((root / 'formal-forbidden').exists())
            gsm = select(14)
            signed = ROOT / 'data/qwen3' / gsm['tag']
            with self.assertRaisesRegex(ValueError, 'only for ID15'):
                prepare(gsm, models / config['sender'], models / 'Qwen3-1.7B',
                        runner.read_rows(signed / 'train.jsonl'),
                        runner.read_rows(signed / 'test.jsonl'),
                        root / 'gsm-forbidden', smoke=True, **_published_grpo_validation(gsm))
            self.assertFalse((root / 'gsm-forbidden').exists())

    def test_one_entry_point_routes_both_model_families(self):
        for experiment, action, family in ((1, 'train', 'qwen3'),
                                            (18, 'evaluate', 'qwen3'),
                                            (19, 'evaluate', 'qwen35'),
                                            (22, 'init-random', 'qwen35'),
                                            (30, 'train', 'qwen35')):
            with self.subTest(experiment=experiment), \
                 patch('sys.argv', ['platour-exps', action,
                                    '--experiment', str(experiment)]), \
                 patch('qwen3.cli.main') as qwen3_main, \
                 patch('qwen35.cli.main') as qwen35_main:
                platour_exps.main()
                self.assertEqual(qwen3_main.call_count, int(family == 'qwen3'))
                self.assertEqual(qwen35_main.call_count, int(family == 'qwen35'))

    def test_cli_defaults_to_vllm_for_qwen3_evaluation_and_grpo_validation(self):
        from qwen3.cli import main
        for experiment, action in ((7, 'evaluate'), (14, 'train')):
            with self.subTest(experiment=experiment), \
                 patch('sys.argv', ['platour-exps', action,
                     '--experiment', str(experiment), '--sender', 'sender',
                     '--receiver', 'receiver', '--output', 'fresh',
                     '--test-data', 'original',
                     *(['--train-data', 'original'] if action == 'train' else [])]), \
                 patch('platour_exps.require_vllm_python',
                       side_effect=RuntimeError('vllm preflight')) as gate:
                with self.assertRaisesRegex(RuntimeError, 'vllm preflight'):
                    main()
                gate.assert_called_once_with(None)

    def test_vllm_prefix_cache_and_full_test_coverage(self):
        key = torch.arange(1 * 2 * 4 * 3).reshape(1, 2, 4, 3)
        values = key + 100
        packed = prefix_tensors(((key, values),), 3)['layer.0.kv']
        self.assertEqual(tuple(packed.shape), (2, 3, 2, 3))
        self.assertTrue(torch.equal(packed[0, :, 1], key[0, 1, :3]))
        calls = []

        class FakeDecoder:
            runtime = {'backend': 'vllm', 'version': '0.17.1'}

            def __init__(self, *args, **kwargs):
                calls.append(kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def generate(self, rows, *, count, **kwargs):
                return [dict(index=row['index'], sample_index=sample, tokens=4,
                             response=r'\boxed{B}')
                        for row in rows for sample in range(count)]

        rows = [{'index': i, 'gold': 'B', 'prompt_token_ids': [1, 2]}
                for i in range(10)]
        with TemporaryDirectory() as folder, \
             patch('qwen3.vllm_decode.VLLMDecoder', FakeDecoder), \
             patch('qwen3.vllm_decode.check_parity',
                   return_value={'mean_abs_logprob_error': 1.,
                                 'max_abs_logprob_error': 2.}) as probe:
            config = {'task': 'chemistry_no_cot', 'eval_sampling': {'temperature': 0}}
            output = Path(folder) / 'eval'
            result = runner.evaluate_vllm(None, None, rows, config,
                                          count=1, cap=64, output=output, batch_size=3)
            probe.assert_not_called()
            self.assertIsNone(result['parity'])
            self.assertFalse((output / 'parity.json').exists())
            self.assertEqual((result['correct'], result['responses'], result['backend']),
                             (10, 10, 'vllm'))
            self.assertEqual(len((output / 'scored.jsonl').read_text().splitlines()), 10)
            self.assertEqual(calls[0]['max_num_seqs'], 32)
            self.assertEqual(calls[0]['max_num_batched_tokens'], 4096)
            report = Path(folder) / 'report'
            reported = runner.evaluate_vllm(None, None, rows,
                {**config, 'parity_report': True}, count=1, cap=64, output=report)
            probe.assert_called_once()
            self.assertEqual(reported['responses'], 10)
            self.assertEqual(json.loads((report / 'parity.json').read_text()),
                             reported['parity'])
            self.assertNotIn('passed', reported['parity'])
            with self.assertRaises(FileExistsError):
                runner.evaluate_vllm(None, None, rows, config,
                                     count=1, cap=64, output=output)

    def test_vllm_uses_per_question_valid_choice_labels(self):
        class Decoder:
            runtime = {'backend': 'vllm', 'version': '0.17.1'}

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def generate(self, rows, *, count, **kwargs):
                return [dict(index=row['index'], sample_index=sample, tokens=2,
                             response=r'\boxed{E}')
                        for row in rows for sample in range(count)]

        rows = [{'index': 0, 'gold': 'E', 'labels': ['A', 'B', 'C', 'D'],
                 'prompt_token_ids': [1, 2]},
                {'index': 1, 'gold': 'E', 'labels': ['A', 'B', 'C', 'D', 'E'],
                 'prompt_token_ids': [1, 3]}]
        with TemporaryDirectory() as folder, patch('qwen3.vllm_decode.VLLMDecoder', Decoder):
            result = runner.evaluate_vllm(None, None, rows,
                {'task': 'arc_c', 'eval_sampling': {'temperature': 0}},
                count=1, cap=16, output=Path(folder) / 'result')
            recorded = [json.loads(line) for line in
                        (Path(folder) / 'result/scored.jsonl').read_text().splitlines()]
        self.assertEqual((result['correct'], result['responses']), (1, 2))
        self.assertEqual([row['correct'] for row in recorded], [0, 1])

    def test_vllm_math_and_gsm8k_details_do_not_duplicate_correct(self):
        class Decoder:
            runtime = {'backend': 'vllm', 'version': '0.17.1'}

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def generate(self, rows, *, count, **kwargs):
                return [dict(index=row['index'], sample_index=sample, tokens=4,
                             response=r'\boxed{1}')
                        for row in rows for sample in range(count)]

        rows = [{'index': 0, 'gold': '1', 'prompt_token_ids': [1, 2]}]
        with TemporaryDirectory() as folder, \
             patch('qwen3.vllm_decode.VLLMDecoder', Decoder):
            for task in ('gsm8k', 'math500'):
                with self.subTest(task=task):
                    output = Path(folder) / task
                    result = runner.evaluate_vllm(None, None, rows,
                        {'task': task, 'eval_sampling': {'temperature': .7}},
                        count=1, cap=16, output=output)
                    recorded = json.loads((output / 'scored.jsonl').read_text().strip())
                    self.assertEqual((result['correct'], result['responses']), (1, 1))
                    self.assertEqual(recorded['correct'], 1)

    def test_vllm_runtime_rejects_wrong_version_before_model_load(self):
        with patch('platour_exps.subprocess.run') as invoke:
            invoke.return_value.returncode = 0
            invoke.return_value.stdout = '0.17.0\n'
            with self.assertRaisesRegex(RuntimeError, 'vLLM 0.17.1'):
                platour_exps.require_vllm_python(Path('/decoder/python'))

    def test_download_verifies_manifest_and_uses_local_cache(self):
        data = b'example checked weight'
        digest = hashlib.sha256(data).hexdigest()
        with TemporaryDirectory() as folder:
            model_dir = Path(folder)
            manifest = json.loads((ROOT / 'models/manifest.json').read_text())
            manifest['files']['01'] = [select(1)['tag'] + '.safetensors', len(data), digest]
            manifest.pop('repositories', None)
            manifest.pop('configs', None)
            manifest['repo_id'] = 'zen-E/Platour'
            (model_dir / 'manifest.json').write_text(json.dumps(manifest))

            def fake_download(*, repo_id, filename, local_dir):
                self.assertEqual(repo_id, 'zen-E/Platour')
                self.assertEqual(local_dir, model_dir)
                destination = local_dir / filename
                destination.write_bytes(data)
                return str(destination)

            with patch.object(platour_exps, 'MODEL_DIR', model_dir), \
                 patch('huggingface_hub.hf_hub_download', side_effect=fake_download) as fetch:
                result = platour_exps.ensure_weight(1)
                self.assertEqual(result.read_bytes(), data)
                self.assertEqual(platour_exps.ensure_weight(1), result)
                fetch.assert_called_once()
                result.write_bytes(b'wrong')
                with self.assertRaisesRegex(ValueError, 'Checkpoint size differs'):
                    platour_exps.ensure_weight(1)
                result.write_bytes(b'wrong' + b' ' * (len(data) - 5))
                with self.assertRaisesRegex(ValueError, 'Checkpoint SHA-256 differs'):
                    platour_exps.ensure_weight(1)

    def test_every_new_qwen3_bridge_projection_starts_exactly_zero(self):
        class Backbone(nn.Module):
            def __init__(self, depth, width):
                super().__init__()
                self.config = SimpleNamespace(num_hidden_layers=depth, hidden_size=width,
                                              pad_token_id=0)
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
                self.lm_head = nn.Linear(width, 16)

            @property
            def device(self):
                return self.model.layers[0].weight.device

        for identifier in range(1, 19):
            with self.subTest(experiment=identifier):
                config = select(identifier)
                bridge = Bridge(Backbone(3, 4), Backbone(2, 3),
                                config['layer_mapping'], config.get('initialization', 'zero'))
                self.assertTrue(all(torch.count_nonzero(layer.weight) == 0
                                    for layer in bridge.P))
        with self.assertRaisesRegex(ValueError, 'zero initialization'):
            Bridge(Backbone(3, 4), Backbone(2, 3), 'last_aligned', 'small_uniform')

    def test_all_experiments_match_released_weights(self):
        configs = [select(i) for i in range(1, 19)]
        self.assertEqual(len({entry["tag"] for entry in configs}), 18)
        self.assertTrue(all(config.get("initialization", "zero") == "zero"
                            for config in configs))
        for config in configs:
            self.assertEqual(config["total_steps"], config["steps_per_epoch"] * 3)
            self.assertEqual(config["steps_per_epoch"],
                             (config["train_examples"] + config["batch_size"] - 1) // config["batch_size"])
            _, filename, size, digest = platour_exps.weight_record(config['id'])
            self.assertEqual(filename, config['tag'] + '.safetensors')
            self.assertGreater(size, 0)
            self.assertEqual(len(digest), 64)
        source = (ROOT / "qwen3/experiments.json").read_text()
        self.assertNotIn("/mnt/", source)
        self.assertNotIn("hostname", source.lower())

    def test_fixed_historical_sft_schedule_and_choice_sampling(self):
        micros = (8, 8, 8, 2, 2, 1, 8, 8, 8, 8, 8, 1)
        warmups = (6, 46, 9, 9, 35, 35, 6, 46, 9, 9, 35, 35)
        for identifier, (micro, warmup) in enumerate(zip(micros, warmups), 1):
            config = select(identifier)
            self.assertEqual((config['micro_batch'], config['warmup_steps']),
                             (micro, warmup))
            if config['task'] in ('arc_c', 'mmlu_redux', 'chemistry_no_cot'):
                self.assertEqual(config['eval_sampling']['temperature'], 0)
            else:
                self.assertEqual(config['eval_sampling']['top_k'], 20)

    def test_sft_learning_rate_warms_from_zero(self):
        config = {"smoke": True, "method": "sft", "task": "arc_c", "train_examples": 2,
                  "total_steps": 2, "steps_per_epoch": 2, "batch_size": 1,
                  "micro_batch": 1, "learning_rate": .01,
                  "warmup_steps": 2, "response_cap": 4, "eval_samples": 1}
        data = [{"prompt_token_ids": [1, 2], "response_token_ids": [3]}] * 2
        rates = []

        def update(bridge, optimizer, rows, indices, micro, total):
            rates.append(optimizer.param_groups[0]['lr'])
            for parameter in bridge.P.parameters():
                parameter.grad = torch.ones_like(parameter)
            return .5, 1

        with TemporaryDirectory() as folder, \
             patch.object(runner, 'sft_update', side_effect=update), \
             patch.object(runner, 'validate', return_value={'heldout_nll': .1,
                                                              'response_tokens': 2}):
            runner.train(config, ToyBridge(), None, data, data, Path(folder) / 'run')
        self.assertEqual(rates, [0, .005])

    def test_bundled_signed_training_validation_test(self):
        folders = {p.name for p in (ROOT / "data").iterdir() if p.is_dir()}
        # Optional HF IT cache lives beside the two published data families.
        self.assertEqual(folders - {"it"}, {"qwen3", "qwen35"})
        self.assertEqual({p.name for p in (ROOT / "data/qwen3").iterdir() if p.is_dir()},
                         {select(i)["tag"] for i in range(1, 19)})
        for identifier in range(1, 19):
            config = select(identifier)
            folder = ROOT / "data/qwen3" / config["tag"]
            manifest_text = (folder / "manifest.json").read_text()
            self.assertNotIn("/mnt/", manifest_text)
            self.assertNotIn("/home/", manifest_text)
            manifest = json.loads(manifest_text)
            self.assertEqual(manifest["experiment_id"], identifier)
            self.assertEqual(manifest["method"], config["method"])
            indices = {}
            for name in ("train", "validation", "test"):
                file = folder / f"{name}.jsonl"
                digest = hashlib.sha256()
                values = []
                with file.open("rb") as stream:
                    for line in stream:
                        digest.update(line)
                        record = json.loads(line)
                        values.append(record["index"])
                        self.assertIsInstance(record["gold"], str)
                        self.assertTrue(record["prompt_token_ids"])
                        if config["method"] == "sft" and name != "test":
                            self.assertTrue(record["response_token_ids"])
                            self.assertIn("response", record)
                        else:
                            self.assertNotIn("response_token_ids", record)
                self.assertEqual(digest.hexdigest(), manifest["sha256"][name])
                self.assertEqual(len(values), manifest["counts"][name])
                self.assertEqual(len(values), len(set(values)))
                indices[name] = values
            self.assertEqual(len(indices["train"]), config["train_examples"])
            if config["method"] == "sft":
                split = json.loads((folder / "split.json").read_text())
                self.assertEqual(indices["train"], split["train_indices"])
                self.assertEqual(indices["validation"], split["validation_indices"])
                self.assertFalse(set(indices["train"]) & set(indices["validation"]))
                self.assertEqual(set(indices["train"]) | set(indices["validation"]),
                                 set(range(len(indices["train"]) + len(indices["validation"]))))
            else:
                self.assertEqual(indices["train"], list(range(config["train_examples"])))
                if config.get('published_grpo_validation'):
                    split = json.loads((folder / 'split.json').read_text())
                    self.assertEqual(len(indices['validation']), 210 if config['task']=='chemistry_cot' else 500)
                    self.assertFalse(set(split['train_source_indices']) & set(split['validation_source_indices']))
                    self.assertNotEqual(manifest['sha256']['validation'], manifest['sha256']['test'])
                else:
                    self.assertEqual(indices["validation"], indices["test"])
                    self.assertEqual(manifest["sha256"]["validation"],
                                     manifest["sha256"]["test"])
            self.assertEqual(len(indices["test"]) * config["eval_samples"],
                             {"arc_c": 1172, "mmlu_redux": 5632,
                              "chemistry_no_cot": 210, "chemistry_cot": 1680,
                              "gsm8k": 2638, "math500": 2000}[config["task"]])

    def test_fixed_holdout_protocol_is_separate_from_reported_weights(self):
        expected = {'chemistry_cot': (1680, 210, 159, 8, 8),
                    'gsm8k': (6154, 1319, 291, 4, 2),
                    'math500': (7000, 500, 330, 4, 4)}
        for identifier in range(13, 19):
            config = select(identifier)
            directory = ROOT / 'data/qwen3' / config['tag']
            original = runner.read_rows(directory / 'train.jsonl')
            test = runner.read_rows(directory / 'test.jsonl')
            options = {'validation_rows': runner.read_rows(directory / 'validation.jsonl')} if config.get('published_grpo_validation') else {}
            training, validation, fresh, split = runner.prepare_grpo_holdout(
                original, test, config, **options)
            count, heldout, steps, val_samples, test_samples = expected[config['task']]
            if options and config["task"] == "gsm8k":
                count, heldout, steps = 6973, 500, 327
            self.assertEqual((len(training), len(validation), fresh['total_steps'],
                              fresh['eval_samples'], fresh['test_eval_samples']),
                             (count, heldout, steps, val_samples, test_samples))
            self.assertEqual(len(fresh['eval_steps']), 6)
            self.assertEqual(fresh['eval_steps'][-1], steps)
            self.assertEqual(split['eval_steps'], [0, *fresh['eval_steps']])
            self.assertNotIn('selected_step', fresh)
            self.assertEqual([row['index'] for row in training], list(range(count)))
            self.assertFalse({tuple(row['prompt_token_ids']) for row in validation} &
                             {tuple(row['prompt_token_ids']) for row in test})
            self.assertEqual(len(original), config['train_examples'])
            self.assertNotIn('selected_step', config)

    def test_original_layer_mapping(self):
        self.assertEqual(layer_map(94, 28, "last_aligned"), list(range(66, 94)))
        self.assertEqual(layer_map(36, 28, "last_aligned"), list(range(8, 36)))
        self.assertEqual(layer_map(94, 28, "uniform_endpoints")[0], 0)
        self.assertEqual(layer_map(94, 28, "uniform_endpoints")[-1], 93)

    def test_public_answer_scoring(self):
        self.assertEqual(last_boxed(r"draft \boxed{a} final \boxed{\frac{1}{2}}"),
                         r"\frac{1}{2}")
        self.assertEqual(score("chemistry_no_cot", r"\boxed{B}", "B"), 1)
        self.assertEqual(score("chemistry_cot", "<answer>C</answer>", "C"), 1)
        self.assertEqual(score("gsm8k", r"\boxed{1,234}", "1234"), 1)
        self.assertEqual(score("gsm8k", "no final answer", "1234"), 0)
        self.assertEqual(score("gsm8k", r"\boxed{1234}", "1234", truncated=True), 0)
        self.assertEqual(evaluation_score("arc_c", r"draft B final \boxed{E}", "E"), 1)
        self.assertEqual(evaluation_score("arc_c", "draft E without a final box", "E"), 0)
        self.assertEqual(evaluation_score("chemistry_no_cot", r"\boxed{B}", "B"), 1)
        self.assertEqual(evaluation_score("gsm8k", r"\boxed{1,234}", "1234"), 1)
        self.assertEqual(evaluation_score("gsm8k", r"\boxed{1234}", "1234"), 1)
        self.assertEqual(evaluation_score("chemistry_cot", "<reasoning>...<answer>B", "B"), 0)
        self.assertEqual(evaluation_score("math500", r"draft \boxed{3} final \boxed{2}", "2"), 1)
        self.assertEqual(evaluation_score("math500", "no final box", "2"), 0)
        self.assertEqual(evaluation_score("arc_c", r"\boxed{B} final \boxed{", "B"), 0)
        self.assertEqual(evaluation_score("gsm8k", r"\boxed{1234} final \boxed{", "1234"), 0)
        self.assertEqual(evaluation_score("math500", r"\boxed{2^{10000000}}", "2^10000001"), 0)
        self.assertEqual(evaluation_score("math500", r"\boxed{2^{10000000}}", "2^10000000"), 1)

    def test_reported_checkpoint_provenance_and_input_gate(self):
        from safetensors import safe_open

        for identifier in range(1, 19):
            config = select(identifier)
            repo_id, filename, size, digest = platour_exps.weight_record(identifier)
            self.assertEqual((repo_id, filename),
                             (json.loads((ROOT / 'models/manifest.json').read_text())['repositories'][f'{identifier:02d}'],
                              config['tag'] + '.safetensors'))
            self.assertGreater(size, 0)
            self.assertEqual(len(digest), 64)
            file = ROOT / 'models' / filename
            if file.is_file():
                audit_published_weight(config, file)
                with safe_open(file, framework='pt', device='cpu') as source:
                    metadata = source.metadata() or {}
                    self.assertNotIn('selected_step', metadata)
                    self.assertNotIn('selected_checkpoint_step', metadata)
                    self.assertNotIn('source_checkpoint_sha256', metadata)
                    if identifier <= 17:
                        self.assertEqual(metadata['experiment'], str(identifier))
        for identifier in (1, 13, 18):
            config = select(identifier)
            folder = ROOT / 'data/qwen3' / config['tag']
            audit_inputs(config, {'train': folder / 'train.jsonl',
                                  'validation': folder / 'validation.jsonl',
                                  'test': folder / 'test.jsonl'})

    def test_checkpoint_survives_scoring_timeout(self):
        config = {"smoke": True, "method": "sft", "task": "arc_c", "total_steps": 2,
                  "train_examples": 2, "steps_per_epoch": 2, "batch_size": 1,
                  "micro_batch": 1, "learning_rate": 0.01, "warmup_steps": 0,
                  "response_cap": 4, "eval_samples": 1}
        rows = [{"prompt_token_ids": [1, 2], "response_token_ids": [3]}] * 2
        model = ToyBridge()

        def fake_sft(bridge, _optim, _data, _indices, _micro, _global_tokens):
            for parameter in bridge.P.parameters():
                parameter.grad = torch.ones_like(parameter)
            return 0.25, 1

        with TemporaryDirectory() as folder:
            root = Path(folder)
            initial = {"heldout_nll": .3, "response_tokens": 2}
            with patch.object(runner, "sft_update", side_effect=fake_sft), patch.object(
                runner, "validate", side_effect=[initial, TimeoutError("scoring stalled")]
            ):
                with self.assertRaisesRegex(TimeoutError, "scoring stalled"):
                    runner.train(config, model, None, rows, rows, root)
            self.assertEqual(json.loads((root / "initial_validation.json").read_text())
                             ["heldout_nll"], .3)
            checkpoint = root / "checkpoints/step-0002"
            self.assertTrue((checkpoint / "complete.json").is_file())
            self.assertTrue((checkpoint / "bridge.pt").is_file())
            self.assertTrue((checkpoint / "rng.rank-00.pt").is_file())
            self.assertFalse((root / "evaluation/step-0002.json").exists())
            trained_weight = model.P[0].weight.detach().clone()
            fresh = ToyBridge()
            optimizer = torch.optim.AdamW(fresh.P.parameters(), lr=config["learning_rate"],
                                          betas=(.9, .999), weight_decay=0.)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            self.assertEqual(load_step(checkpoint, fresh, optimizer, scheduler), 2)
            self.assertTrue(torch.equal(fresh.P[0].weight, trained_weight))
            with patch.object(runner, "validate", return_value={"heldout_nll": .2,
                                                                   "response_tokens": 2}):
                runner.train(config, ToyBridge(), None, rows, rows, root, resume=checkpoint)
            saved = json.loads((root / "evaluation/step-0002.json").read_text())
            self.assertEqual(saved["step"], 2)
            self.assertEqual(saved["heldout_nll"], .2)
            self.assertEqual(len((root / "metrics.jsonl").read_text().splitlines()), 2)

    def test_grpo_checkpoint_is_complete_before_validation(self):
        config = {"smoke": True, "method": "grpo", "task": "gsm8k", "total_steps": 2,
                  "train_examples": 2, "steps_per_epoch": 2, "batch_size": 1,
                  "group_size": 8, "learning_rate": .01, "eval_steps": [2],
                  "eval_samples": 2, "response_cap": 4}
        data = [{"prompt_token_ids": [1, 2], "gold": "3", "index": i}
                for i in range(2)]
        model = ToyBridge()

        def fake_update(bridge, _tokenizer, _data, _indices, _config, _step, _global_groups):
            for parameter in bridge.P.parameters():
                parameter.grad = torch.ones_like(parameter)
            return .5, 4

        with TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(runner, "grpo_update", side_effect=fake_update), patch.object(
                runner, "evaluate_vllm", side_effect=TimeoutError("validation timeout")
            ):
                with self.assertRaisesRegex(TimeoutError, "validation timeout"):
                    runner.train(config, model, None, data, data, root)
            checkpoint = root / "checkpoints/step-0002"
            self.assertEqual(json.loads((checkpoint / "complete.json").read_text())["step"], 2)
            self.assertEqual(torch.load(checkpoint / "training.pt", weights_only=True)["step"], 2)
            self.assertFalse((root / "evaluation/step-0002.json").exists())
            with patch.object(runner, "evaluate_vllm", return_value={"correct": 3,
                "responses": 4, "accuracy": .75}):
                runner.train(config, ToyBridge(), None, data, data, root, resume=checkpoint)
            self.assertEqual(json.loads((root / "evaluation/step-0002.json").read_text())["correct"], 3)
            self.assertEqual(len((root / "metrics.jsonl").read_text().splitlines()), 2)

    def test_holdout_selects_validation_best_then_tests_once_after_resume(self):
        config = {'smoke': True, 'method': 'grpo', 'task': 'gsm8k', 'total_steps': 2,
                  'steps_per_epoch': 2, 'batch_size': 1, 'group_size': 8,
                  'learning_rate': .01, 'eval_steps': [1, 2], 'eval_samples': 2,
                  'test_eval_samples': 1, 'response_cap': 4,
                  'validation_policy': 'train_holdout_best_validation_v1'}
        data = [{'index': i, 'prompt_token_ids': [1, i + 2], 'gold': 'train'}
                for i in range(2)]
        validation = [{'index': 0, 'prompt_token_ids': [7], 'gold': 'holdout'}]
        test = [{'index': i, 'prompt_token_ids': [i + 8], 'gold': 'test'}
                for i in range(2)]
        split = {'train_source_indices': [0, 1], 'validation_source_indices': [2]}
        updates, validations, tests, failed = [], [], [], [True]

        def update(bridge, _tokenizer, _data, _indices, _config, step, _groups):
            updates.append(step)
            for parameter in bridge.P.parameters():
                parameter.grad = torch.ones_like(parameter)
            return .5, 4

        with TemporaryDirectory() as folder:
            output = Path(folder) / 'run'

            def validate(bridge, tokenizer, rows, config, **kwargs):
                self.assertEqual(rows, validation)
                saved = list((output / 'checkpoints').glob('step-*/complete.json'))
                step = max(int(path.parent.name[5:]) for path in saved)
                validations.append(step)
                if step == 1 and failed[0]:
                    failed[0] = False
                    raise TimeoutError('holdout scorer stalled')
                correct = {0: 0, 1: 2, 2: 1}[step]
                return {'correct': correct, 'responses': 2, 'accuracy': correct / 2}

            def evaluate(bridge, tokenizer, rows, config, *, count, cap, **kwargs):
                self.assertEqual(rows, test)
                self.assertEqual(count, 1)
                self.assertEqual(json.loads((output / 'selection.json').read_text())
                                 ['selected_step'], 1)
                expected = torch.load(output / 'checkpoints/step-0001/bridge.pt',
                                      weights_only=True)['P']['0.weight']
                self.assertTrue(torch.equal(bridge.P[0].weight, expected))
                tests.append(count)
                return {'correct': 1, 'responses': 2, 'accuracy': .5}

            with patch.object(runner, 'grpo_update', side_effect=update), \
                 patch.object(runner, 'evaluate_vllm', side_effect=lambda *a, **kw: (evaluate if a[2] == test else validate)(*a, **kw)):
                with self.assertRaisesRegex(TimeoutError, 'holdout scorer stalled'):
                    runner.train(config, ToyBridge(), None, data, validation, output,
                                 final_test=test, split=split)
                signed = output / 'checkpoints/step-0001'
                self.assertTrue((signed / 'complete.json').is_file())
                self.assertEqual(updates, [1])
                self.assertFalse((output / 'test/summary.json').exists())
                restored = ToyBridge()
                runner.train(config, restored, None, data, validation, output,
                             resume=signed, final_test=test, split=split)
                terminal = torch.load(output / 'checkpoints/step-0002/bridge.pt',
                                      weights_only=True)['P']['0.weight']
                self.assertTrue(torch.equal(restored.P[0].weight, terminal))
                runner.train(config, ToyBridge(), None, data, validation, output,
                             resume=output / 'checkpoints/step-0002',
                             final_test=test, split=split)
            self.assertEqual(updates, [1, 2])
            self.assertEqual(validations, [0, 1, 1, 2])
            self.assertEqual(tests, [1])
            self.assertEqual(json.loads((output / 'test/summary.json').read_text())
                             ['selected_step'], 1)
            self.assertEqual([json.loads(line)['step'] for line in
                              (output / 'metrics.jsonl').read_text().splitlines()], [1, 2])

    def test_grpo_crash_between_checkpoints_resumes_bitwise_with_restored_rng(self):
        config = {'smoke': True, 'method': 'grpo', 'task': 'gsm8k', 'total_steps': 4, 'steps_per_epoch': 4,
                  'batch_size': 1, 'group_size': 2, 'learning_rate': .01, 'eval_steps': [2, 4],
                  'eval_samples': 1, 'test_eval_samples': 1, 'response_cap': 4,
                  'validation_policy': 'train_holdout_best_validation_v1'}
        data = [{'index': i, 'prompt_token_ids': [1, i + 2], 'gold': 'train'} for i in range(4)]
        validation = [{'index': 0, 'prompt_token_ids': [7], 'gold': 'holdout'}]
        test = [{'index': 0, 'prompt_token_ids': [8], 'gold': 'test'}]
        split = {'train_source_indices': [0, 1, 2, 3], 'validation_source_indices': [4]}

        def run(output, *, resume=None, crash_at=None):
            def update(bridge, _tokenizer, _data, indices, _config, step, _groups):
                if step == crash_at:
                    raise SimulatedCrash(f'process died before update {step}')
                # Sampled rollouts draw from the global torch and Python RNGs.
                noise = torch.randn(1, 1) * (1 + random.random()) * (indices[0] + 1)
                for parameter in bridge.P.parameters():
                    parameter.grad = noise.expand_as(parameter).clone()
                return float(noise), 1

            def validate(*_args, **_kwargs):
                torch.rand(3)  # sampling during validation must not shift the training stream
                random.random()
                return {'correct': 1, 'responses': 1, 'accuracy': 1.}
            torch.manual_seed(0)
            bridge = ToyBridge()
            with patch.object(runner, 'grpo_update', side_effect=update), \
                 patch.object(runner, 'validate', side_effect=validate), \
                 patch.object(runner, 'evaluate_vllm', return_value={'correct': 1, 'responses': 1,
                                                                'accuracy': 1.}):
                runner.train(config, bridge, None, data, validation, output, resume=resume,
                             final_test=test, split=split)
            return bridge.P[0].weight.detach().clone()

        with TemporaryDirectory() as folder:
            reference = run(Path(folder) / 'reference')
            crashed = Path(folder) / 'crashed'
            with self.assertRaises(SimulatedCrash):
                run(crashed, crash_at=3)
            self.assertFalse((crashed / 'checkpoints/step-0003').exists())
            resumed = run(crashed, resume=crashed / 'checkpoints/step-0002')
            self.assertTrue(torch.equal(resumed, reference))
            steps = lambda root: [json.loads(line) for line in (root / 'metrics.jsonl').open()]
            self.assertEqual(steps(crashed), steps(Path(folder) / 'reference'))
            for step in (2, 4):
                name = f'checkpoints/step-{step:04d}/bridge.pt'
                self.assertTrue(torch.equal(
                    torch.load(crashed / name, weights_only=True)['P']['0.weight'],
                    torch.load(Path(folder) / 'reference' / name, weights_only=True)['P']['0.weight']))

    def test_8b_release_evaluation_does_not_force_single_gpu(self):
        from qwen3.cli import main
        with TemporaryDirectory() as folder:
            output = Path(folder) / 'evaluation'
            sender, receiver = Path(folder) / 'Qwen3-8B', Path(folder) / 'Qwen3-1.7B'
            sender.mkdir()
            receiver.mkdir()
            argv = ['platour-exps', 'evaluate', '--experiment', '7',
                    '--sender', str(sender), '--receiver', str(receiver),
                    '--test-data', 'signed-test', '--weights', 'pinned.safetensors',
                    '--output', str(output), '--backend', 'vllm']
            config = lambda depth, width: SimpleNamespace(num_hidden_layers=depth,
                                                           hidden_size=width)
            bridge = SimpleNamespace(sender=SimpleNamespace(config=config(36, 4096)),
                                     receiver=SimpleNamespace(config=config(28, 2048)),
                                     load_public=lambda *_: None)

            def fake_vllm(*_args, **kwargs):
                kwargs['output'].mkdir()
                return {'complete': True, 'correct': 1, 'responses': 1}

            with patch('sys.argv', argv), \
                 patch('platour_exps.require_vllm_python'), \
                 patch('qwen3.cli.audit_inputs'), patch('qwen3.cli.audit_published_weight'), \
                 patch('qwen3.cli.load_bridge', return_value=(bridge, None)) as loader, \
                 patch('qwen3.cli.read_rows', return_value=[
                     {'index': 0, 'prompt_token_ids': [1, 2], 'gold': 'A'}]), \
                 patch('qwen3.cli.evaluate_vllm', side_effect=fake_vllm) as score:
                main()
                self.assertNotIn('device_map', loader.call_args.kwargs)
                self.assertEqual(score.call_args.kwargs['memory_fraction'], .25)
                self.assertEqual(json.loads((output / 'summary.json').read_text())['responses'], 1)

    def test_8b_portable_training_does_not_require_single_gpu_flag(self):
        from qwen3.cli import main
        config = lambda depth, width: SimpleNamespace(num_hidden_layers=depth,
                                                       hidden_size=width)
        bridge = SimpleNamespace(sender=SimpleNamespace(config=config(36, 4096)),
                                 receiver=SimpleNamespace(config=config(28, 2048)))
        training = [{'index': i, 'prompt_token_ids': [1, 2],
                     'response_token_ids': [3]} for i in range(1107)]
        validation = [{'index': 0, 'prompt_token_ids': [1, 4],
                       'response_token_ids': [3]}]
        for available in (1, 8):
            with self.subTest(available_gpus=available), TemporaryDirectory() as folder, \
                 patch('torch.cuda.device_count', return_value=available), \
                 patch('sys.argv', ['platour-exps', 'train', '--experiment', '7',
                                    '--sender', 'Qwen3-8B', '--receiver', 'Qwen3-1.7B',
                                    '--train-data', 'signed-train',
                                    '--validation-data', 'signed-validation',
                                    '--output', str(Path(folder) / 'new-run')]), \
                 patch('qwen3.cli.audit_inputs'), \
                 patch('qwen3.cli.read_rows', side_effect=(training, validation)), \
                 patch('qwen3.cli.load_bridge', return_value=(bridge, None)) as loader, \
                 patch('qwen3.cli.train') as trainer:
                main()
                self.assertNotIn('device_map', loader.call_args.kwargs)
                trainer.assert_called_once()

    def test_8b_loader_uses_one_card_only_when_one_is_visible(self):
        from qwen3.runner import load_bridge
        config = SimpleNamespace(quantization_config={}, num_hidden_layers=2, hidden_size=4)

        class FakeModel:
            device = torch.device('cpu')

            def to(self, device):
                self.device = torch.device(device)
                return self

        for visible in (1, 8):
            with self.subTest(visible_gpus=visible), \
                 patch('torch.cuda.device_count', return_value=visible), \
                 patch('transformers.AutoConfig.from_pretrained', return_value=config), \
                 patch('qwen3.runner.AutoModelForCausalLM.from_pretrained',
                       side_effect=lambda *_args, **_kwargs: FakeModel()) as from_pretrained, \
                 patch('qwen3.runner.AutoTokenizer.from_pretrained', return_value=SimpleNamespace(pad_token_id=0)), \
                 patch('qwen3.runner.Bridge', return_value=SimpleNamespace()) as bridge:
                load_bridge('sender', 'receiver', 'last_aligned')
                self.assertEqual(from_pretrained.call_count, 2)
                expected = {} if visible == 1 else {'device_map': 'auto'}
                for call in from_pretrained.call_args_list:
                    self.assertEqual(call.kwargs.get('device_map'), expected.get('device_map'))
                if visible == 1:
                    self.assertEqual(bridge.call_args.args[0].device, torch.device('cuda:0'))
                    self.assertEqual(bridge.call_args.args[1].device, torch.device('cuda:0'))

    def test_independent_vllm_seed_does_not_change_published_eval_seeds(self):
        from qwen3.vllm_decode import VLLMDecoder

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            decoder = VLLMDecoder.__new__(VLLMDecoder)
            decoder.prefill_batch_size = 1
            decoder.directory = root
            decoder.memory = root / 'memory'
            decoder.memory.mkdir()
            decoder.counter = 0
            decoder.process = SimpleNamespace(stdin=SimpleNamespace(
                write=lambda *_: None, flush=lambda: None))
            rows = [{'index': 0, 'prompt_token_ids': [1, 2]}]
            result = {'rows': [{'index': 0, 'sample_index': 0}]}
            with patch.object(decoder, '_export_batch', return_value=[{'memory_key': 'sample'}]), \
                 patch.object(decoder, '_wait', return_value=result):
                decoder.generate(rows, count=1, cap=3, sampling={}, seed_override=42)
                payload = json.loads((root / 'request-00001.json').read_text())
                self.assertEqual(payload['rows'][0]['sampling_seed'], 42)
                decoder.generate(rows, count=1, cap=3, sampling={})
                payload = json.loads((root / 'request-00002.json').read_text())
                self.assertEqual(payload['rows'][0]['sampling_seed'], 1234)
                with self.assertRaisesRegex(ValueError, 'one prompt and one reply'):
                    decoder.generate(rows, count=2, cap=3, sampling={}, seed_override=42)

    def test_pact_check_template_donor_and_four_measured_controls(self):
        from pact_check import (DummyStatistics, IrrelevantFeatures, PACTCheck,
                                build_dummy_statistics, select_irrelevant_donors)

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(32, 4)
                self.layers = nn.ModuleList([nn.Linear(4, 4)])
                self.forwards = 0

            def forward(self, input_ids, **_):
                self.forwards += 1
                value = self.embed(input_ids)
                states = (value,)
                for layer in self.layers:
                    value = layer(value)
                return SimpleNamespace(hidden_states=states, last_hidden_state=value)

        class LM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Encoder()
                self.lm_head = nn.Linear(4, 32, bias=False)
                self.config = SimpleNamespace(num_hidden_layers=1, hidden_size=4,
                                              pad_token_id=0, eos_token_id=0)

            @property
            def device(self):
                return self.model.embed.weight.device

        bridge = Bridge(LM(), LM(), 'last_aligned')
        with torch.no_grad():
            bridge.P[0].weight.copy_(torch.eye(4))
        train = [{'index': 0, 'prompt_token_ids': [1, 5, 9, 2], 'group': 'a'},
                 {'index': 1, 'prompt_token_ids': [1, 6, 9, 2], 'group': 'b'}]
        val = [{'index': 2, 'prompt_token_ids': [1, 10, 9, 2], 'group': 'c'}]
        test = [{'index': 3, 'prompt_token_ids': [1, 7, 9, 2], 'group': 'd'},
                {'index': 4, 'prompt_token_ids': [1, 8, 9, 2], 'group': 'e'}]
        stats = build_dummy_statistics(bridge, train, prefix_ids=[1], suffix_ids=[9],
                                       sample_count=2)
        with TemporaryDirectory() as folder:
            saved = Path(folder) / 'TRAIN-only.safetensors'
            stats.save(saved)
            stats = DummyStatistics.load(saved)
            before = bridge.sender.model.forwards
            with bridge.use_features(stats.features):
                dt = bridge._features(test[0]['prompt_token_ids'])
                self.assertEqual(float(dt[0][0, 1].abs().sum()), 0.)
                bridge.prefix_cache_batch([row['prompt_token_ids'] for row in test],
                                          return_output=True)
            self.assertEqual(before, bridge.sender.model.forwards)
            self.assertEqual(bridge._features(test[0]['prompt_token_ids'])[0].shape, (1, 3, 4))
            self.assertGreater(bridge.sender.model.forwards, before)
        donor_ids = select_irrelevant_donors(train, test,
            unrelated=lambda donor, target: donor['group'] != target['group'])
        donors = {row['index']: donor_ids[row['index']] for row in test}
        it = IrrelevantFeatures(bridge, {tuple(row['prompt_token_ids']): donors[row['index']]
                                         for row in test})
        with bridge.use_features(it):
            self.assertEqual(bridge._features(test[0]['prompt_token_ids'])[0].shape, (1, 3, 4))
        with self.assertRaisesRegex(ValueError, 'duplicate/paraphrase'):
            select_irrelevant_donors(train, test)
        calls = []

        def evaluate(system, rows, *, feature_provider):
            calls.append(('evaluate', len(rows)))
            if system is bridge:
                correct = 2 if getattr(feature_provider, '__self__', None) is stats else (
                    1 if isinstance(feature_provider, IrrelevantFeatures) else 4)
            else:
                correct = 3 if system == 'trained-sc' else 1
            return {'complete': True, 'correct': correct, 'responses': 4,
                    'backend': 'hf'}

        def factory(kind, checkpoint=None):
            def train_control(rows, validation):
                self.assertEqual(len(validation), 1)
                calls.append(('train', kind))
                return 'trained-' + kind.lower(), {'trained': True, 'control': kind,
                    'train_digest': stats.training_digest,
                    'validation_digest': pact.validation_digest,
                    'checkpoint': checkpoint or f'{kind}/complete.json',
                    'uses_test_for_training': False}
            return train_control

        pact = PACTCheck(bridge, train, val, test, evaluator=evaluate)
        result = pact.run(
            dummy=stats, donors=donors, train_sc=factory('SC'), train_tr=factory('TR'),
            samples_per_question=2)
        # Each control is tested once, only after its own TRAIN/VAL training finished.
        self.assertEqual(calls[:4], [('train', 'SC'), ('evaluate', 2), ('train', 'TR'),
                                     ('evaluate', 2)])
        self.assertEqual(len(calls), 6)
        with self.assertRaisesRegex(RuntimeError, 'separately trained'):
            pact.run(dummy=stats, donors=donors, train_sc=factory('SC', 'same/complete.json'),
                     train_tr=factory('TR', 'same/complete.json'), samples_per_question=2)
        self.assertNotIn('pact_score_pp', result)
        self.assertEqual(result['accuracies'], dict(DT=.5, IT=.25, SC=.75, TR=.25))
        self.assertEqual(result['responses_per_setting'], 4)
        self.assertFalse(result['dt_paper_calibration_count_met'])

    def test_independent_api_sft_load_messages_and_resume_same_checkpoint(self):
        from platour_api import PLatourSession, load_dataset
        import platour_api

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(32, 4)
                self.layers = nn.ModuleList([nn.Linear(4, 4)])

            def forward(self, input_ids, **_):
                hidden = self.embed(input_ids)
                states = (hidden,)
                for layer in self.layers:
                    hidden = layer(hidden)
                return SimpleNamespace(hidden_states=states, last_hidden_state=hidden)

        class LM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Encoder()
                self.lm_head = nn.Linear(4, 32, bias=False)
                self.config = SimpleNamespace(num_hidden_layers=1, hidden_size=4,
                                              pad_token_id=0, eos_token_id=0)

            @property
            def device(self):
                return self.model.embed.weight.device

            def generate(self, input_ids, **_):
                self.model(input_ids)
                return torch.cat((input_ids, input_ids.new_tensor([[3]])), dim=1)

        class Tokenizer:
            pad_token_id = 0

            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                if not tokenize or not add_generation_prompt or messages[0]['role'] != 'user':
                    raise AssertionError('Chat template was not applied')
                return {'input_ids': [1, 5, 2], 'attention_mask': [1, 1, 1]}

            def decode(self, tokens, **_):
                return str(tokens[0])

        def session():
            torch.manual_seed(142)
            bridge = Bridge(LM(), LM(), 'last_aligned')
            bridge.pad_token_id = 0
            return PLatourSession(bridge, Tokenizer(), model_identity='toy-bridge')

        train = [{'index': 0, 'prompt_token_ids': [1, 5, 2], 'response_token_ids': [3, 4]},
                 {'index': 1, 'prompt_token_ids': [1, 6, 2], 'response_token_ids': [4]}]
        heldout = [{'index': 2, 'prompt_token_ids': [1, 7, 2], 'response_token_ids': [3]}]
        with TemporaryDirectory() as folder:
            first = session()
            with self.assertRaisesRegex(ValueError, 'requires vLLM'):
                first.generate(messages=[{'role': 'user', 'content': 'sample'}], backend='hf')
            root = Path(folder) / 'interrupted'
            original = platour_api.atomic_json
            stopped = [False]

            def interruption(value, destination):
                if str(destination).endswith('evaluation/step-0001.json') and not stopped[0]:
                    stopped[0] = True
                    raise RuntimeError('validation interrupted after a signed update')
                return original(value, destination)

            with patch('platour_api.atomic_json', side_effect=interruption):
                with self.assertRaisesRegex(RuntimeError, 'validation interrupted'):
                    first.fit(train, heldout, output=root, validation_backend='hf',
                              batch_size=1, max_steps=2, smoke=True, validation_schedule='step')
            signed = root / 'checkpoints/step-0001'
            self.assertTrue((signed / 'complete.json').is_file())
            restored = session()
            restored.fit(train, heldout, output=root, validation_backend='hf',
                         batch_size=1, max_steps=2, smoke=True, validation_schedule='step')
            intact = session()
            intact.fit(train, heldout, output=Path(folder) / 'intact',
                       validation_backend='hf', batch_size=1, max_steps=2, smoke=True, validation_schedule='step')
            self.assertTrue(torch.equal(restored.bridge.P[0].weight,
                                        intact.bridge.P[0].weight))
            self.assertEqual([json.loads(row)['step'] for row in
                              (root / 'metrics.jsonl').read_text().splitlines()], [1, 2])
            self.assertTrue((root / 'complete.json').is_file())
            with patch.object(restored, 'evaluate', side_effect=AssertionError('already validated')):
                restored.fit(train, heldout, output=root, validation_backend='hf',
                             batch_size=1, max_steps=2, smoke=True, validation_schedule='step')
            self.assertEqual(len(load_dataset(train)), 2)

    def test_independent_receiver_full_and_lora_rolling_resume(self):
        from platour_api import ReceiverSession
        from qwen35.checkpoints import latest, sha256
        import qwen35.checkpoints

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(32, 4)
                self.layers = nn.ModuleList([nn.Linear(4, 4)])

            def forward(self, input_ids, **_):
                hidden = self.embed(input_ids)
                for layer in self.layers:
                    hidden = layer(hidden)
                return SimpleNamespace(last_hidden_state=hidden)

        class LM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Encoder()
                self.lm_head = nn.Linear(4, 32, bias=False)
                self.config = SimpleNamespace(eos_token_id=0)

            @property
            def device(self):
                return self.model.embed.weight.device

            def forward(self, input_ids, **_):
                return SimpleNamespace(logits=self.lm_head(self.model(input_ids).last_hidden_state))

            def generate(self, input_ids, **_):
                self.forward(input_ids)
                return torch.cat((input_ids, input_ids.new_tensor([[3]])), dim=1)

        class Tokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                return {'input_ids': [1, 5, 2]}

            def decode(self, tokens, **_):
                return str(tokens[0])

        def fresh(method):
            torch.manual_seed(140)
            return ReceiverSession(LM(), Tokenizer(), method=method, lora_rank=2,
                                   lora_alpha=2, model_identity='tiny-receiver')

        train = [{'index': 0, 'prompt_token_ids': [1, 5, 2], 'response_token_ids': [3]},
                 {'index': 1, 'prompt_token_ids': [1, 6, 2], 'response_token_ids': [4]}]
        validation = [{'index': 0, 'prompt_token_ids': [1, 7, 2], 'response_token_ids': [3]}]
        for method in ('full', 'lora'):
            with self.subTest(method=method), TemporaryDirectory() as folder:
                root = Path(folder) / 'interrupted'
                complete = Path(folder) / 'uninterrupted'
                original = qwen35.checkpoints.atomic_json
                failed = [False]

                def stop_after_signed_step(path, payload):
                    if str(path).endswith('evaluation/step-0001.json') and not failed[0]:
                        failed[0] = True
                        raise RuntimeError('TR validation interrupted')
                    return original(path, payload)

                first = fresh(method)
                with patch('qwen35.checkpoints.atomic_json', side_effect=stop_after_signed_step):
                    with self.assertRaisesRegex(RuntimeError, 'TR validation interrupted'):
                        first.fit(train, validation, output=root, max_steps=2, smoke=True, batch_size=1, validation_schedule='step')
                self.assertEqual(int(latest(root).name[5:]), 1)
                resumed = fresh(method)
                receipt = resumed.fit(train, validation, output=root, max_steps=2, smoke=True, batch_size=1, validation_schedule='step')
                direct = fresh(method)
                direct.fit(train, validation, output=complete, max_steps=2, smoke=True, batch_size=1, validation_schedule='step')
                self.assertTrue(receipt['trained'])
                self.assertEqual(receipt['control'], 'TR')
                self.assertEqual(sha256(latest(root) / 'model.safetensors'),
                                 sha256(latest(complete) / 'model.safetensors'))
                self.assertEqual([json.loads(x)['step'] for x in
                                  (root / 'metrics.jsonl').read_text().splitlines()], [1, 2])
                self.assertTrue((root / 'evaluation/step-0001.json').is_file())
                with patch.object(resumed, '_decoder', return_value=mock_receiver_decoder(resumed)):
                    self.assertEqual(resumed.generate(messages=[{'role': 'user', 'content': 'example'}],
                                                      max_new_tokens=1, temperature=0)['response'], '3')
                with self.assertRaisesRegex(ValueError, 'local model and private output'):
                    resumed.generate(prompt_ids=[1, 2], backend='vllm')

    def test_pact_check_cli_trains_four_controls_on_tiny_splits(self):
        import pact_check
        from platour_api import PLatourSession, ReceiverSession

        class TinyCore(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(32, 4)
                self.layers = nn.ModuleList([nn.Linear(4, 4)])

            def forward(self, input_ids, **_):
                hidden = self.embed(input_ids)
                states = (hidden,)
                for layer in self.layers:
                    hidden = layer(hidden)
                return SimpleNamespace(hidden_states=states, last_hidden_state=hidden)

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = TinyCore()
                self.lm_head = nn.Linear(4, 32, bias=False)
                self.config = SimpleNamespace(num_hidden_layers=1, hidden_size=4,
                                              pad_token_id=0, eos_token_id=0,
                                              _name_or_path='tiny')

            @property
            def device(self):
                return self.model.embed.weight.device

            def forward(self, input_ids, **_):
                result = self.model(input_ids)
                return SimpleNamespace(logits=self.lm_head(result.last_hidden_state))

            def generate(self, input_ids, **_):
                self.forward(input_ids)
                return torch.cat((input_ids, input_ids.new_tensor([[3]])), dim=1)

        class TinyTokenizer:
            pad_token_id = 0

            def decode(self, tokens, **_):
                return r'\boxed{3}'

        def base(*_args, **_kwargs):
            torch.manual_seed(16)
            bridge = Bridge(TinyLM(), TinyLM(), 'last_aligned')
            bridge.pad_token_id = 0
            return PLatourSession(bridge, TinyTokenizer(), model_identity='tiny')

        from contextlib import contextmanager
        @contextmanager
        def pact_decoder(*args, **kwargs):
            yield SimpleNamespace(generate=lambda rows, count, **kw: [dict(
                index=row['index'], sample_index=i, response_token_ids=[3], log_probs=[-1.], response=r'\boxed{3}')
                for row in rows for i in range(count)])

        def tr(*_args, **_kwargs):
            torch.manual_seed(17)
            session = ReceiverSession(TinyLM(), TinyTokenizer(), method='lora',
                                   lora_rank=2, lora_alpha=2, model_identity='tiny')
            session._decoder = lambda *_a, **_kw: mock_receiver_decoder(session)
            return session

        with TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'data'
            data.mkdir()
            for part, rows in (('train', [(0, 5, 'a'), (1, 6, 'b')]),
                               ('validation', [(2, 7, 'c')]),
                               ('test', [(3, 8, 'd')])):
                with (data / (part + '.jsonl')).open('w') as output:
                    for index, question, group in rows:
                        output.write(json.dumps({'index': index,
                            'prompt_token_ids': [1, question, 9, 2],
                            'response_token_ids': [3], 'gold': '3',
                            'semantic_group': group}) + '\n')
            bridge_file = root / 'source.safetensors'
            bridge_file.write_bytes(b'test-only-bridge-identity')
            output = root / 'pact'
            argv = ['platour-exps', 'pact-check', '--it-source', 'custom', '--sender', 'tiny-sender',
                    '--bridge', str(bridge_file), '--receiver', 'tiny-receiver',
                    '--dataset', str(data), '--task', 'gsm8k', '--prefix-ids', '[1]',
                    '--suffix-ids', '[9]', '--semantic-group-field', 'semantic_group',
                    '--samples', '1', '--dt-samples', '2', '--batch-size', '1',
                    '--response-cap', '1', '--tr-lora-rank', '2', '--control-objective', 'sft',
                    '--backend', 'vllm', '--output', str(output)]
            with patch('sys.argv', argv), patch('platour_exps.require_vllm_python'), patch('qwen3.vllm_decode.decoder_session', pact_decoder), patch('platour_api.PLatourSession.load', side_effect=base), \
                 patch('platour_api.ReceiverSession.load', side_effect=tr):
                platour_exps.main()
            report = json.loads((output / 'pact_check.json').read_text())
            self.assertEqual(set(report['raw']), {'DT', 'IT', 'SC', 'TR'})
            self.assertEqual(report['responses_per_setting'], 1)
            self.assertFalse(report['dt_paper_calibration_count_met'])
            self.assertNotIn('paper_equivalent', report)
            self.assertEqual(report['control_objectives'], {'SC': 'sft', 'TR': 'sft'})
            self.assertEqual({r['backend'] for r in report['raw'].values()}, {'vllm'})
            self.assertTrue((output / 'SC/complete.json').is_file())
            self.assertTrue((output / 'TR/complete.json').is_file())
            sc_checkpoint = hashlib.sha256((output / 'SC/checkpoints/step-0006/bridge.pt').read_bytes()).hexdigest()
            tr_checkpoint = hashlib.sha256((output / 'TR/checkpoints/step-0006/model.safetensors').read_bytes()).hexdigest()
            original_evaluate = PLatourSession.evaluate
            def no_sc_evaluation(system, *args, **kwargs):
                if (getattr(system, 'training_receipt', None) or {}).get('control') == 'SC':
                    raise AssertionError('Completed SC validation/TEST must be reused')
                return original_evaluate(system, *args, **kwargs)
            (output / 'pact_check.json').unlink()
            with patch('sys.argv', argv), patch('platour_exps.require_vllm_python'), patch('qwen3.vllm_decode.decoder_session', pact_decoder), \
                 patch('platour_api.PLatourSession.evaluate', new=no_sc_evaluation), \
                 patch('platour_api.PLatourSession.load', side_effect=base), \
                 patch('platour_api.ReceiverSession.load', side_effect=tr), \
                 patch('platour_api.ReceiverSession.evaluate',
                       side_effect=AssertionError('Completed TR validation/TEST must be reused')):
                platour_exps.main()
            self.assertEqual(hashlib.sha256(
                (output / 'SC/checkpoints/step-0006/bridge.pt').read_bytes()).hexdigest(), sc_checkpoint)
            self.assertEqual(hashlib.sha256(
                (output / 'TR/checkpoints/step-0006/model.safetensors').read_bytes()).hexdigest(), tr_checkpoint)
            self.assertTrue((output / 'pact_check.json').is_file())
            with patch('platour_exps.require_vllm_python'), patch('sys.argv', argv):
                with self.assertRaisesRegex(RuntimeError, 'same incomplete private run'):
                    platour_exps.main()
            grpo_output = root / 'pact-grpo'
            grpo_argv = argv.copy()
            grpo_argv[grpo_argv.index('sft')] = 'grpo'
            grpo_argv[-1] = str(grpo_output)
            with patch('sys.argv', grpo_argv), patch('platour_exps.require_vllm_python'), patch('qwen3.vllm_decode.decoder_session', pact_decoder), \
                 patch('platour_api.PLatourSession.load', side_effect=base), \
                 patch('platour_api.ReceiverSession.load', side_effect=tr):
                platour_exps.main()
            grpo = json.loads((grpo_output / 'pact_check.json').read_text())
            self.assertEqual(grpo['control_objectives'], {'SC': 'grpo', 'TR': 'grpo'})
            self.assertEqual({result['backend'] for result in grpo['raw'].values()}, {'vllm'})
            for label in ('SC', 'TR'):
                metric = json.loads((grpo_output / label / 'metrics.jsonl').read_text().splitlines()[0])
                self.assertEqual(metric['step'], 1)
                self.assertEqual(metric['responses'] if label == 'SC' else metric['train/responses'], 8)

    def test_receiver_only_grpo_mixed_group_updates_signed_optimizer(self):
        from platour_api import ReceiverSession
        from qwen35.checkpoints import latest, valid

        class TinyStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))

            @property
            def device(self):
                return self.weight.device

        class TinyActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.student = TinyStudent()
                self.experiment = SimpleNamespace(method='full', lora_rank=2, lora_alpha=2)

            def trainable(self):
                return (self.student.weight,)

            def inference_state(self):
                return {'weight': self.student.weight}

            def load_inference_state(self, state):
                with torch.no_grad():
                    self.student.weight.copy_(state['weight'])

            def generate(self, _prompt, *, seed, **_):
                return [3 if seed % 2 else 4]

            def response_log_probs(self, _prompt, response):
                return self.student.weight * (1 if response[0] == 3 else -1)

        session = ReceiverSession.__new__(ReceiverSession)
        session.actor = TinyActor()
        session.tokenizer = SimpleNamespace(decode=lambda tokens, **_: r'\boxed{' + str(tokens[0]) + '}')
        session.method = 'full'
        session.model_identity = 'tiny-receiver'
        session.model_path = None
        train = [{'index': 0, 'prompt_token_ids': [1, 2], 'gold': '3'}]
        val = [{'index': 1, 'prompt_token_ids': [1, 5], 'gold': '3'}]
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / 'tr-grpo'
            session._decoder = lambda *_a, **_kw: mock_receiver_decoder(session)
            receipt = session.fit(train, val, output=root, objective='grpo', task='gsm8k',
                                  validation_backend='vllm', response_cap=2, max_steps=1, smoke=True)
            self.assertEqual(receipt['control'], 'TR')
            self.assertTrue(valid(latest(root), expected_step=1))
            update = json.loads((root / 'metrics.jsonl').read_text().splitlines()[0])
            self.assertEqual(update['train/responses'], 8)
            self.assertEqual(update['train/correct'], 4)
            self.assertGreater(update['grad_norm'], 0)
            from safetensors.torch import load_file
            self.assertGreater(abs(load_file(str(latest(root) / 'model.safetensors'))['weight'].item()), 0)
            self.assertEqual(json.loads((root / 'evaluation/step-0001.json').read_text())['responses'], 1)

    def test_receiver_only_vllm_snapshot_merges_current_lora_without_changing_base(self):
        from qwen3.vllm_decode import ReceiverOnlyDecoder
        from qwen35.model import Actor
        from safetensors.torch import load_file

        class TinyStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Linear(2, 2, bias=False)])
                self.config = SimpleNamespace(eos_token_id=0)

            @property
            def device(self):
                return torch.device('cuda:0')

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / 'public-model'
            model.mkdir()
            (model / 'config.json').write_text('{}')
            (model / 'tokenizer.json').write_text('{}')
            student = TinyStudent()
            actor = Actor(SimpleNamespace(method='lora', lora_target_count=1,
                                          lora_rank=2, lora_alpha=2), student)
            adapter = actor.adapters['model.layers.0']
            before = adapter.base.weight.detach().clone()
            with torch.no_grad():
                adapter.lora_A.fill_(1.)
                adapter.lora_B.fill_(.5)
            processes = []

            class FakeProcess:
                def __init__(self, cmd, **_):
                    self.directory = Path(cmd[-1])
                    self.alive = True
                    self.pid = 2147483630
                    self.stdin = SimpleNamespace(write=self.write, flush=lambda: None)
                    processes.append(self)
                    (self.directory / 'ready.json').write_text('{"complete":true,"backend":"vllm"}')

                def poll(self):
                    return None if self.alive else 0

                def wait(self, **_):
                    self.alive = False
                    return 0

                def write(self, line):
                    message = json.loads(line)
                    if message.get('command') == 'stop':
                        self.alive = False
                        return
                    request = json.loads(Path(message['request']).read_text())
                    items = [{'index': r['index'], 'sample_index': r['sample_index'],
                              'sampling_seed': r['sampling_seed'], 'response': 'ok',
                              'response_token_ids': [3],
                              **({'log_probs': [-.5]} if request.get('parity') else {})}
                             for r in request['rows']]
                    Path(message['result']).write_text(json.dumps({'complete': True, 'rows': items}))

            with patch('qwen3.vllm_decode.subprocess.Popen', FakeProcess), \
                 patch('qwen3.vllm_decode.prepare_worker_memory') as prepare_memory:
                with ReceiverOnlyDecoder(actor, model, root / 'scores',
                                         max_model_len=128, worker_python=sys.executable) as decoder:
                    state = load_file(decoder.directory / 'model/model.safetensors')
                    self.assertEqual(set(state), {'model.layers.0.weight'})
                    torch.testing.assert_close(state['model.layers.0.weight'], before + 1.)
                    self.assertEqual(decoder.generate([{'index': 4, 'prompt_token_ids': [1, 2]}],
                        count=1, cap=3, sampling={'temperature': .7, 'top_p': .8, 'top_k': 20})
                        [0]['sampling_seed'], 1238)
                    self.assertEqual(decoder.generate([{'index': 4, 'prompt_token_ids': [1, 2]}],
                        count=8, cap=3, sampling={'temperature': 1., 'top_p': 1., 'top_k': -1},
                        logprobs=True)[0]['log_probs'], [-.5])
            torch.testing.assert_close(adapter.base.weight, before)
            prepare_memory.assert_called_once_with(student.device, decoder.directory)
            self.assertFalse(processes[0].alive)
            self.assertFalse((decoder.directory / 'model/model.safetensors').exists())

    def test_pact_cli_default_vllm_checks_decoder_before_loading_models(self):
        with patch('sys.argv', ['platour-exps', 'pact-check', '--it-source', 'custom', '--sender', 'missing-sender',
               '--bridge', 'missing-bridge', '--receiver', 'missing-receiver',
               '--dataset', 'missing-data', '--task', 'gsm8k', '--output', 'missing-output',
               '--prefix-ids', '[1]', '--suffix-ids', '[2]',
               '--semantic-group-field', 'group']), \
             patch('platour_exps.require_vllm_python', side_effect=RuntimeError(
                'native decoder unavailable')) as preflight:
            with self.assertRaisesRegex(RuntimeError, 'native decoder unavailable'):
                platour_exps.main()
            preflight.assert_called_once_with(None)

    def test_one_generic_loader_routes_all_released_family_method_roles(self):
        from platour_api import PLatourSession, Qwen35Session, ReceiverSession, load_session

        cases = (("sft", "last_aligned"), ("grpo", "uniform_endpoints"),
                 ("bridge", "last_aligned"))
        for method, mapping in cases:
            with self.subTest(family='qwen3', method=method), \
                 patch.object(PLatourSession, 'load', return_value=method) as loader:
                self.assertEqual(load_session(family='qwen3', sender='Qwen3-8B',
                    bridge='trained.safetensors', receiver='Qwen3-1.7B', method=method), method)
                self.assertEqual(loader.call_args.kwargs['mapping'], mapping)
        with patch.object(PLatourSession, 'load', return_value='self') as loader:
            self.assertEqual(load_session(family='qwen3', sender=None,
                receiver='Qwen3-1.7B', method='self'), 'self')
            self.assertEqual(loader.call_args.args[:2], ('Qwen3-1.7B', 'Qwen3-1.7B'))
        for task in ('dapo', 'deepcoder'):
            for method in ('bridge', 'random', 'full', 'lora', 'self'):
                with self.subTest(family='qwen35', task=task, method=method), \
                     patch.object(Qwen35Session, 'load', return_value='routed') as loader:
                    self.assertEqual(load_session(family='qwen35',
                        sender='Qwen3.5-35B-A3B' if method == 'bridge' else None,
                        bridge='trained.safetensors' if method != 'self' else None,
                        receiver='Qwen3.5-4B', method=method, task=task), 'routed')
                    self.assertEqual(loader.call_args.kwargs['method'],
                                     'bridge' if method == 'self' else method)
                    if method == 'self':
                        self.assertEqual(loader.call_args.kwargs['sender'], 'Qwen3.5-4B')
        with patch.object(ReceiverSession, 'load', return_value='tr'):
            self.assertEqual(load_session(family='qwen3', receiver='Qwen3-1.7B',
                                          method='full'), 'tr')
        with self.assertRaisesRegex(ValueError, 'receiver-as-sender'):
            load_session(family='qwen35', sender='different', receiver='Qwen3.5-4B',
                         method='self', task='dapo')


if __name__ == "__main__":
    unittest.main()
