"""CPU-only contract tests; never load the released large language models."""

import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from qwen35.catalog import EXPERIMENTS, select
from qwen35.checkpoints import (atomic_json, latest, load_step, read,
                                reconcile_history, save_step, sha256, valid)
from qwen35.cli import evaluate, evaluate_vllm, inputs
from qwen35.vllm_decode import hybrid_tensors, sample_first
from qwen35.model import Actor
from qwen35.random_init import template


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher = nn.Linear(2, 2)
        self.teacher.requires_grad_(False)
        self.projection = nn.Linear(2, 2, bias=False)

    def trainable(self):
        return self.projection.parameters()

    def inference_state(self):
        return {'P.0.weight': self.projection.weight}

    def load_inference_state(self, saved):
        if set(saved) != set(self.inference_state()):
            raise ValueError('Missing trainable projection')
        with torch.no_grad():
            self.projection.weight.copy_(saved['P.0.weight'])


class ToyReceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            nn.ModuleDict({f'linear_{index}': nn.Linear(2, 2)
                           for index in range(8)}) for _ in range(31)])
        self.lm_head = nn.Linear(2, 2)
        self.config = SimpleNamespace(eos_token_id=2)

    @property
    def device(self):
        return torch.device('cpu')


class DataParallelToy(nn.Module):
    """Deterministic toy GRPO actor: sampled tokens depend only on seed, loss on weight."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.3, -.2]))
        self.student = SimpleNamespace(device=torch.device('cpu'))

    def trainable(self):
        return (self.weight,)

    def generate(self, prompt, *, seed, **_kwargs):
        return [seed % 5 + 1]

    def response_log_probs(self, prompt, answer):
        return (self.weight[0] * answer[0] + self.weight[1] * prompt[1]).reshape(1) * .1


def portable_args(**kwargs):
    return SimpleNamespace(**dict(dict(receiver=Path('/unused'), vllm_python=None,
        vllm_memory_fraction=None, backend='vllm'), **kwargs))


def _grpo_toy_update():
    from qwen35.cli import train_step
    torch.manual_seed(7)
    actor = DataParallelToy()
    # SGD is linear in the summed gradient; Adam's first step would amplify ~eps noise.
    optimizer = torch.optim.SGD(actor.trainable(), lr=.05)
    batch = [dict(index=index, prompt_token_ids=[1, 2 + index], gold='1') for index in range(3)]
    tokenizer = SimpleNamespace(decode=lambda tokens, **_kw: str(tokens))
    from contextlib import contextmanager
    @contextmanager
    def decoder(actor, *a, **kw):
        def generate(rows, count, seed_base, **kw):
            result = []
            for row in rows:
                for i in range(count):
                    tokens = actor.generate(row['prompt_token_ids'], seed=seed_base + count*row['index'] + i)
                    with torch.no_grad():
                        logs = actor.response_log_probs(row['prompt_token_ids'], tokens).tolist()
                    result.append(dict(index=row['index'], sample_index=i, response_token_ids=tokens, log_probs=logs))
            return result
        yield SimpleNamespace(generate=generate)
    with patch('qwen35.vllm_decode.training_decoder', decoder), patch('qwen35.cli._grade', side_effect=lambda _case, _row, _text, tokens, *_a, **_kw:
               int(tokens[0] % 2 == 0)):
        metrics = train_step(actor, optimizer, tokenizer, SimpleNamespace(task='dapo'), batch, 1, receiver='/unused', output='/unused')
    return {'weight': actor.weight.detach().clone(), 'metrics': metrics}


def _grpo_data_parallel_worker(rank, world, port, folder):
    import os
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    dist.init_process_group('gloo', rank=rank, world_size=world)
    try:
        torch.save(_grpo_toy_update(), Path(folder) / f'rank-{rank}.pt')
    finally:
        dist.destroy_process_group()


class PactToySession:
    """Stand-in Qwen35Session with real sender-feature routing and signed fits."""

    created = []

    def __init__(self, method, *, sender=False, released=False):
        self.method = method
        self.released = released
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.seen = []
        provider = [None]

        def sender_features(prompt):
            return tuple(torch.full((1, len(prompt) - 1, 4), float(prompt[1]))
                         for _ in range(32))

        @__import__('contextlib').contextmanager
        def use_features(function):
            provider[0] = function
            try:
                yield
            finally:
                provider[0] = None

        self.actor = SimpleNamespace(sender_features=sender_features, use_features=use_features,
                                     provider=provider, _features=lambda p: sender_features(p))
        PactToySession.created.append(self)

    @classmethod
    def load(cls, sender=None, bridge=None, receiver=None, *, task, method, **_options):
        return cls(method, sender=sender is not None, released=bridge is not None)

    def evaluate(self, rows, *, backend, samples, cap, output, vllm_python, feature_provider):
        with (self.actor.use_features(feature_provider) if feature_provider else
              __import__('contextlib').nullcontext()):
            features = [self.actor.provider[0](row['prompt_token_ids']) if self.actor.provider[0]
                        else self.actor.sender_features(row['prompt_token_ids']) for row in rows]
        self.seen.append(float(features[0][0][0, -1, 0]))
        signal = self.released and feature_provider is None
        correct = len(rows) * samples if signal else 0
        return {'complete': True, 'correct': float(correct), 'responses': len(rows) * samples,
                'backend': backend}

    def fit(self, train, heldout, *, output, objective, control, **_options):
        from platour_api import _digest
        if (Path(output) / 'complete.json').is_file():
            return self.restore_completed(output, control=control)
        _options['max_steps'] = _options['epochs'] * ((len(train) + _options['batch_size'] - 1) // _options['batch_size'])
        Path(output).mkdir(parents=True)
        atomic_json(Path(output) / 'protocol.json', {'objective': objective,
            'validation_backend': _options['validation_backend'], 'max_steps': _options['max_steps']})
        atomic_json(Path(output) / 'complete.json', {'complete': True, 'steps': _options['max_steps']})
        if control == 'TR' and not getattr(PactToySession, 'tr_failed', False):
            PactToySession.tr_failed = True
            raise RuntimeError('TR control interrupted')
        return {'trained': True, 'control': control, 'train_digest': _digest(train),
                'validation_digest': _digest(heldout), 'uses_test_for_training': False,
                'checkpoint': str(Path(output) / 'complete.json')}

    def restore_completed(self, output, *, control):
        from platour_api import _digest
        self.restored = control
        return {'trained': True, 'control': control, 'train_digest': PactToySession.digests[0],
                'validation_digest': PactToySession.digests[1], 'uses_test_for_training': False,
                'checkpoint': str(Path(output) / 'complete.json')}


def _tiny_qwen35_text(folder, layers=4, hidden=32):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    types = ['full_attention' if (index + 1) % 4 == 0 else 'linear_attention'
             for index in range(layers)]
    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_conv_kernel_dim=4, layer_types=types, max_position_embeddings=128,
        tie_word_embeddings=False, eos_token_id=2, pad_token_id=0, bos_token_id=None)
    torch.manual_seed(0)
    Qwen3_5ForCausalLM(config).save_pretrained(folder)
    return Path(folder)


class Qwen35CPUAndShapeTests(unittest.TestCase):
    def test_load_text_uses_hf_reference_kernels_and_text_only_checkpoint_on_cpu(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as dense
        from qwen35.model import REFERENCE_KERNELS, load_text
        with tempfile.TemporaryDirectory() as folder:
            path = _tiny_qwen35_text(folder)
            with patch('torch.cuda.is_available', return_value=False):
                model = load_text(path, device='cpu')
            for name in REFERENCE_KERNELS:
                self.assertEqual(getattr(dense, name).__module__, dense.__name__, name)
            self.assertEqual(type(model).__name__, 'Qwen3_5ForCausalLM')
            self.assertEqual(next(model.parameters()).device.type, 'cpu')
            prompt = torch.tensor([[5, 6, 7, 8, 9]])
            with torch.no_grad():
                hidden = model.model(input_ids=prompt, output_hidden_states=True,
                                     use_cache=False).hidden_states
                output = model.generate(prompt, max_new_tokens=3, do_sample=False, pad_token_id=0)
            self.assertEqual(len(hidden), 5)
            self.assertEqual(tuple(output.shape), (1, 8))

    def test_virtual_sender_rejects_prompts_without_the_fixed_template(self):
        actor = Actor.__new__(Actor)
        nn.Module.__init__(actor)
        actor.experiment = SimpleNamespace(method='random', task='dapo')
        actor.student = SimpleNamespace(device=torch.device('cpu'))
        actor.register_buffer('template_token_ids', torch.tensor([1, 2, 8, 9]))
        actor.prefix_count, actor.suffix_count = 2, 3
        actor.virtual_hidden = nn.Parameter(torch.ones(4, 2, 3))
        features, mask = actor._features([1, 2, 5, 6, 8, 9, 7])
        self.assertEqual(mask.tolist(), [True, True, False, False, True, True])
        self.assertTrue(torch.equal(features[1][0, 4], torch.ones(3)))
        for prompt in ([1, 2, 7], [3, 2, 5, 8, 9, 7]):
            with self.assertRaisesRegex(ValueError, 'fixed template'):
                actor._features(prompt)

    def test_cpu_and_cuda_kernel_bindings_alternate_within_one_process(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as dense
        from qwen35.model import _ORIGINAL_KERNELS, use_cuda_kernels, use_reference_kernels
        use_cuda_kernels((dense,))
        original = _ORIGINAL_KERNELS[(dense.__name__, 'causal_conv1d_fn')]
        self.assertIs(dense.causal_conv1d_fn, original)
        self.assertNotEqual(dense.torch_chunk_gated_delta_rule.__module__, dense.__name__)
        use_reference_kernels((dense,))
        self.assertEqual(dense.torch_chunk_gated_delta_rule.__module__, dense.__name__)
        self.assertEqual(dense.causal_conv1d_fn.__module__, dense.__name__)
        use_cuda_kernels((dense,))
        self.assertIs(dense.causal_conv1d_fn, original)
        use_reference_kernels((dense,))
        self.assertEqual(dense.torch_recurrent_gated_delta_rule.__module__, dense.__name__)

    def test_fp8_qwen35_sender_requires_cuda_and_pins_a_trusted_kernel(self):
        import json as _json
        from transformers.integrations.hub_kernels import _HUB_KERNEL_MAPPING
        from qwen35.model import load_text
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'config.json').write_text(_json.dumps(
                {'model_type': 'qwen3_5_moe', 'quantization_config': {'quant_method': 'fp8'}}))
            with patch('torch.cuda.is_available', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'require CUDA'):
                    load_text(folder, device='cpu')
            from transformers.integrations import hub_kernels
            trusted = []
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('transformers.integrations.finegrained_fp8.load_finegrained_fp8_kernel',
                       side_effect=lambda: trusted.append(hub_kernels.ALLOW_ALL_KERNELS)), \
                 patch('transformers.AutoModelForImageTextToText.from_pretrained',
                       side_effect=RuntimeError('stop before loading weights')), \
                 patch.dict(_HUB_KERNEL_MAPPING, {}, clear=False):
                with self.assertRaisesRegex(RuntimeError, 'stop before'):
                    load_text(folder, device='cuda:0', device_map='auto')
                self.assertEqual(trusted, [True])
                self.assertFalse(hub_kernels.ALLOW_ALL_KERNELS)
                self.assertEqual(_HUB_KERNEL_MAPPING['finegrained-fp8']['revision'],
                                 '8c178950e8710e26a2210e4e909cd02dc16c8715')

    def test_receiver_shape_defaults_to_4b_and_accepts_only_an_explicit_other_shape(self):
        from qwen35.model import load_text
        with tempfile.TemporaryDirectory() as folder:
            student = load_text(_tiny_qwen35_text(Path(folder) / 'receiver'), device='cpu')
            teacher = load_text(_tiny_qwen35_text(Path(folder) / 'sender', hidden=48), device='cpu')
            base = dict(task='dapo', method='bridge', layer_map=(0, 1, 2, 3), teacher_dim=48)
            with self.assertRaisesRegex(ValueError, 'default 4B'):
                Actor(SimpleNamespace(**base), student, teacher=teacher)
            actor = Actor(SimpleNamespace(**base, receiver_shape=(4, 32)), student, teacher=teacher)
            self.assertEqual(len(actor.projections), 4)
            self.assertEqual(tuple(actor.projections[0].weight.shape), (32, 48))
            self.assertTrue(all(not layer.weight.any() for layer in actor.projections))
            with self.assertRaisesRegex(ValueError, 'every receiver layer'):
                Actor(SimpleNamespace(**dict(base, layer_map=(0, 1, 2)), receiver_shape=(4, 32)),
                      student, teacher=teacher)


class CheckpointTests(unittest.TestCase):
    def test_qwen35_pact_check_runs_four_controls_and_resumes_trained_controls(self):
        import pact_check
        import platour_api
        from platour_api import _digest
        rows = lambda start, count, prefix: [
            {'index': start + i, 'prompt_token_ids': [1, start + i + 10, 7, 2, 9],
             'gold': '1', 'group': f'{prefix}{i}'} for i in range(count)]
        train, validation, test = rows(0, 4, 'train'), rows(100, 2, 'val'), rows(200, 2, 'test')
        PactToySession.created, PactToySession.digests = [], (_digest(train), _digest(validation))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data'
            data.mkdir()
            for name, split in (('train', train), ('validation', validation), ('test', test)):
                (data / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in split))
            bridge = root / 'bridge.safetensors'
            bridge.write_bytes(b'released-bridge')
            argv = ['pact-check', '--family', 'qwen35', '--sender', str(root / 'sender'),
                    '--bridge', str(bridge), '--receiver', str(root / 'receiver'),
                    '--dataset', str(data), '--task', 'dapo', '--output', str(root / 'pact'),
                    '--prefix-ids', '[1]', '--suffix-ids', '[2]', '--semantic-group-field', 'group',
                    '--dt-samples', '2', '--samples', '2', '--backend', 'vllm']
            with patch('platour_exps.require_vllm_python'), patch('sys.argv', argv), patch.object(platour_api, 'Qwen35Session', PactToySession):
                PactToySession.tr_failed = False
                with self.assertRaisesRegex(RuntimeError, 'TR control interrupted'):
                    pact_check.main()
                self.assertTrue((root / 'pact/SC/complete.json').is_file())
                self.assertFalse((root / 'pact/pact_check.json').exists())
                pact_check.main()
            report = read(root / 'pact/pact_check.json')
            self.assertEqual(report['family'], 'qwen35')
            self.assertNotIn('pact_score_pp', report)
            self.assertEqual(set(report['raw']), {'DT', 'IT', 'SC', 'TR'})
            self.assertEqual(report['responses_per_setting'], 4)
            self.assertEqual(report['dummy_training_samples'], 2)
            restored = {getattr(session, 'restored', None) for session in PactToySession.created}
            self.assertIn('SC', restored)
            base = [session for session in PactToySession.created if session.released][-1]
            dt, it = base.seen
            self.assertNotEqual(it, 210.)
            self.assertNotEqual(dt, it)

    def test_qwen35_actor_feature_hook_replaces_sender_inputs(self):
        case = SimpleNamespace(method='bridge', task='dapo', layer_map=tuple(range(32)),
                               teacher_dim=4)
        actor = Actor.__new__(Actor)
        nn.Module.__init__(actor)
        actor.experiment = case
        actor.student = SimpleNamespace(device=torch.device('cpu'))
        actor.projections = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(32)])
        supplied = tuple(torch.ones(1, 3, 4) for _ in range(32))
        with actor.use_features(lambda prompt: supplied):
            features, mask = actor._features([1, 2, 3, 4])
            self.assertTrue(all(torch.equal(value, torch.ones(1, 3, 4)) for value in features))
            self.assertTrue(bool(mask.all()))
        self.assertIsNone(actor._feature_provider)
        with actor.use_features(lambda prompt: supplied[:31]):
            with self.assertRaisesRegex(ValueError, 'wrong layer'):
                actor._features([1, 2, 3, 4])

    def test_validation_best_selection_rules_for_math_and_code_without_early_stop(self):
        from qwen35.selection import update
        point = lambda accuracy, clip: {'complete': True, 'accuracy': accuracy, 'clipping_rate': clip}
        math = SimpleNamespace(task='dapo', steps=264)
        state = update(math, None, 0, point(.50, .01))
        state = update(math, state, 33, point(.60, .01))
        self.assertEqual(state['best_step'], 33)
        state = update(math, state, 66, point(.70, .06))
        self.assertEqual(state['best_step'], 66)
        self.assertFalse(state['candidates'][-1]['clip_guard'])
        state = update(math, state, 99, point(.59, .01))
        state = update(math, state, 132, point(.59, .01))
        state = update(math, state, 165, point(.65, .015))
        self.assertEqual(state['best_step'], 66)
        state = update(math, state, 198, point(.70, .031))
        self.assertEqual(state['best_step'], 66)
        state = update(math, state, 231, point(.65, .02))
        self.assertEqual(state['best_step'], 66)
        self.assertEqual([row['step'] for row in state['candidates']], [0, 33, 66, 99, 132, 165, 198, 231])
        self.assertNotIn('fails', state)
        with self.assertRaisesRegex(RuntimeError, 'reversed'):
            update(math, state, 231, point(.9, 0.))
        code = SimpleNamespace(task='deepcoder', steps=295)
        state = update(code, None, 0, point(.20, .10))
        state = update(code, state, 42, point(.30, .30))
        self.assertEqual(state['best_step'], 42)
        state = update(code, state, 84, point(.25, .10))
        self.assertEqual(state['best_step'], 42)
        state = update(code, state, 126, point(.26, .36))
        self.assertEqual(state['best_step'], 42)
        with self.assertRaisesRegex(RuntimeError, 'clipping rate'):
            update(code, state, 168, {'complete': True, 'accuracy': .3})
        with self.assertRaisesRegex(RuntimeError, 'baseline'):
            update(code, None, 42, point(.3, .1))

    def test_portable_training_runs_full_budget_and_tests_validation_best_once(self):
        from qwen35.cli import run_train
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / 'input/manifest.json'
            manifest.parent.mkdir()
            manifest.write_text('{"complete":true}')
            case = SimpleNamespace(identifier=19, tag='cpu-select', task='dapo', steps=4,
                                   checkpoint_steps=(0, 1, 2, 3, 4), validation_steps=(1, 2, 3, 4),
                                   val_samples=1, train_questions=256, learning_rate=.01,
                                   sender=None, method='bridge')
            arguments = portable_args(output=root / 'run', data_root=manifest.parent,
                                        code_cases=None, official_checker=None, docker_image=None)
            samples = {'train': [dict(index=i, prompt_token_ids=[1, 2], gold='1') for i in range(256)],
                       'validation': [dict(index=0, prompt_token_ids=[1, 3], gold='1')]}
            accuracy = {0: .5, 1: .6, 2: .55, 3: .5, 4: .58}
            trained, tested = [], []

            def update(actor, optimizer, *_args, **_kwargs):
                trained.append(_args[3])
                optimizer.zero_grad(set_to_none=True)
                actor.projection.weight.sum().backward()
                optimizer.step()
                return {'train/grad_norm': 1.0}

            def score(*_args, **_kwargs):
                step = int(latest(arguments.output).name[5:])
                return {'complete': True, 'correct': 1, 'responses': 1,
                        'accuracy': accuracy[step], 'clipping_rate': 0.}

            with patch('qwen35.cli.inputs', return_value=samples), \
                 patch('qwen35.cli._model', side_effect=lambda *_a, **_kw: (TinyActor(), None)), \
                 patch('qwen35.cli.train_step', side_effect=update), \
                 patch('qwen35.vllm_decode.decoder_session'), patch('qwen35.cli.evaluate_vllm', side_effect=score), \
                 patch('qwen35.cli._final_test', side_effect=lambda *a, **k: tested.append(a[2])):
                run_train(case, arguments)
                self.assertEqual(trained, [1, 2, 3, 4])
                chosen = read(arguments.output / 'selection.json')
                terminal = read(arguments.output / 'complete.json')
                self.assertEqual((chosen['selected_step'], terminal['steps']), (1, 4))
                self.assertEqual(terminal['selected_step'], 1)
                self.assertNotIn('stopped_early', terminal)
                self.assertEqual([row['step'] for row in chosen['candidates']], [0, 1, 2, 3, 4])
                self.assertEqual(chosen['policy'], 'validation_best_accuracy_full_budget_v2')
                self.assertFalse(chosen['test_used_for_selection'])
                self.assertTrue(valid(arguments.output / 'checkpoints/step-0001', expected_step=1))
                run_train(case, arguments)
            self.assertEqual(trained, [1, 2, 3, 4])
            self.assertEqual(tested, [arguments.output, arguments.output])

    def test_portable_crash_and_resume_matches_uninterrupted_run_bitwise(self):
        from safetensors.torch import load_file
        from qwen35.cli import run_train

        class Crash(Exception):
            pass

        def run(root, output, crash_at=None):
            case = SimpleNamespace(identifier=19, tag='cpu-resume', task='dapo', steps=4,
                                   checkpoint_steps=(0, 2, 4), validation_steps=(2, 4),
                                   val_samples=1, train_questions=256, learning_rate=.01,
                                   sender=None, method='bridge')
            arguments = portable_args(output=output, data_root=root / 'input', code_cases=None,
                                        official_checker=None, docker_image=None)
            samples = {'train': [dict(index=i, prompt_token_ids=[1, 2], gold='1') for i in range(256)],
                       'validation': [dict(index=0, prompt_token_ids=[1, 3], gold='1')]}

            def update(actor, optimizer, _tokenizer, _case, batch, step, **_kwargs):
                if step == crash_at:
                    raise Crash(f'process died before update {step}')
                # Sampled rollouts draw from the global torch and Python RNGs.
                optimizer.zero_grad(set_to_none=True)
                noise = torch.randn(2, 2) * (1 + random.random())
                (actor.projection.weight * noise).sum().backward()
                optimizer.step()
                return {'train/grad_norm': float(noise.norm()),
                        'train/batch': [row['index'] for row in batch]}

            def score(*_args, **_kwargs):
                torch.rand(3)  # validation sampling must not shift the training stream
                random.random()
                step = int(latest(output).name[5:])
                return {'complete': True, 'correct': 1, 'responses': 1,
                        'accuracy': {0: .5, 2: .7, 4: .6}[step], 'clipping_rate': 0.}

            with patch('qwen35.cli.inputs', return_value=samples), \
                 patch('qwen35.cli._model', side_effect=lambda *_a, **_kw: (TinyActor(), None)), \
                 patch('qwen35.cli.train_step', side_effect=update), \
                 patch('qwen35.vllm_decode.decoder_session'), patch('qwen35.cli.evaluate_vllm', side_effect=score), \
                 patch('qwen35.cli._final_test'):
                run_train(case, arguments)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'input').mkdir()
            (root / 'input/manifest.json').write_text('{"complete":true}')
            reference, crashed = root / 'reference', root / 'crashed'
            run(root, reference)
            with self.assertRaises(Crash):
                run(root, crashed, crash_at=3)
            self.assertEqual(latest(crashed).name, 'step-0002')
            run(root, crashed)
            for name in ('metrics.jsonl',):
                self.assertEqual((crashed / name).read_text(), (reference / name).read_text())
            self.assertEqual(read(crashed / 'selection.json'), read(reference / 'selection.json'))
            for step in (2, 3, 4):
                folder = f'checkpoints/step-{step:04d}'
                self.assertTrue(torch.equal(load_file(crashed / folder / 'model.safetensors')['P.0.weight'],
                                            load_file(reference / folder / 'model.safetensors')['P.0.weight']))
                self.assertEqual(torch.load(crashed / folder / 'optimizer.pt', weights_only=True)['state'][0]
                                 ['exp_avg'].tolist(),
                                 torch.load(reference / folder / 'optimizer.pt', weights_only=True)['state'][0]
                                 ['exp_avg'].tolist())

    def test_evaluate_run_loads_signed_validation_best_state(self):
        from qwen35.cli import _match_state, _selected_state
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = SimpleNamespace(identifier=21, tag='21_toy', method='bridge')
            actor = TinyActor()
            optimizer = torch.optim.AdamW(actor.trainable(), lr=.1)
            save_step(root, 0, actor, optimizer, None, protected={0}, identity='x:21_toy')
            with torch.no_grad():
                actor.projection.weight.fill_(.25)
            save_step(root, 1, actor, optimizer, {'step': 1}, protected={0, 1}, identity='x:21_toy')
            atomic_json(root / 'selection.json', {'policy': 'reference', 'selected_step': 1,
                        'experiment': 21, 'validation_accuracy': .4})
            state, provenance = _selected_state(case, root)
            self.assertEqual(provenance['selected_step'], 1)
            fresh = TinyActor()
            fresh.load_inference_state(_match_state(fresh.inference_state(), state))
            self.assertTrue(torch.equal(fresh.projection.weight, torch.full((2, 2), .25)))
            native = root / 'native/train'
            (native / 'checkpoints/step-0033').mkdir(parents=True)
            torch.save({'P': {'0.weight': torch.full((2, 2), .5)}},
                       native / 'checkpoints/step-0033/bridge.pt')
            atomic_json(native / 'early_stop.json', {'best_step': 33})
            state, provenance = _selected_state(case, root / 'native')
            fresh.load_inference_state(_match_state(fresh.inference_state(), state))
            self.assertTrue(torch.equal(fresh.projection.weight, torch.full((2, 2), .5)))
            with self.assertRaisesRegex(ValueError, 'do not match'):
                _match_state(fresh.inference_state(), {'0.weight': torch.zeros(3, 2)})

    def test_multi_rank_checkpoint_restores_each_rank_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actor = TinyActor()
            optimizer = torch.optim.AdamW(actor.trainable(), lr=.1)
            torch.manual_seed(1)
            first = {'cpu': torch.get_rng_state(), 'cuda': []}
            torch.manual_seed(2)
            second = {'cpu': torch.get_rng_state(), 'cuda': []}
            saved = save_step(root, 0, actor, optimizer, None, rank_rng=[first, second])
            torch.manual_seed(99)
            load_step(saved, actor, optimizer, rank=1, world=2)
            torch.manual_seed(2)
            expected = torch.rand(3)
            torch.set_rng_state(second['cpu'])
            self.assertTrue(torch.equal(torch.rand(3), expected))
            with self.assertRaisesRegex(RuntimeError, 'same torchrun world size'):
                load_step(saved, actor, optimizer, rank=0, world=4)
            with self.assertRaisesRegex(RuntimeError, 'same torchrun world size'):
                load_step(saved, actor, optimizer, rank=0, world=1)

    def test_two_process_data_parallel_grpo_matches_one_process_update(self):
        import socket
        import torch.multiprocessing as mp
        reference = _grpo_toy_update()
        self.assertGreater(reference['metrics']['train/mixed_groups'], 0)
        try:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
        except PermissionError:
            self.skipTest('Sandbox denies sockets required for multiprocess training test')
        with tempfile.TemporaryDirectory() as temporary:
            mp.spawn(_grpo_data_parallel_worker, args=(2, port, temporary), nprocs=2, join=True)
            ranks = [torch.load(Path(temporary) / f'rank-{rank}.pt', weights_only=False)
                     for rank in range(2)]
        self.assertTrue(torch.equal(ranks[0]['weight'], ranks[1]['weight']))
        torch.testing.assert_close(ranks[0]['weight'], reference['weight'], atol=1e-6, rtol=0)
        for key in ('train/accuracy', 'train/responses', 'train/examples', 'train/mixed_groups'):
            self.assertEqual(ranks[0]['metrics'][key], reference['metrics'][key])

    def test_original_lcb_v6_case_hashes_are_complete_and_do_not_bundle_cases(self):
        import re
        from qwen35.judge_cases import PINNED

        proof = json.loads(PINNED.read_text())
        cases = proof['case_sha256']
        self.assertEqual((proof['cases'], len(cases)), (175, 175))
        self.assertRegex(proof['checker_sha256'], r'^[a-f0-9]{64}$')
        self.assertRegex(proof['extractor_sha256'], r'^[a-f0-9]{64}$')
        self.assertTrue(all(re.fullmatch(r'[a-z0-9_]+\.json', name) and
                            re.fullmatch(r'[a-f0-9]{64}', digest)
                            for name, digest in cases.items()))
        self.assertFalse(any(PINNED.parent.glob('**/deepcoder__lcbv5__*.json')))
        self.assertEqual(proof['dataset_revision'], '0fe84c3912ea0c4d4a78037083943e8f0c4dd505')
        self.assertEqual(proof['livecodebench_commit'], '28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24')
        self.assertRegex(proof['test6_sha256'], r'^[a-f0-9]{64}$')

    def test_public_v6_extraction_is_restricted_to_json_string_pickles(self):
        import base64
        import hashlib
        import pickle
        import zlib
        from unittest.mock import patch as patch_object
        from qwen35 import judge_cases

        public = json.dumps([{'input': '2', 'output': '4', 'testtype': 'stdin'}])
        private = json.dumps([{'input': '3' * 260, 'output': '9', 'testtype': 'stdin'}])
        private_encoded = base64.b64encode(zlib.compress(pickle.dumps(private, protocol=4))).decode()
        row = {'platform': 'atcoder', 'question_id': 'unit_test',
               'metadata': json.dumps({'func_name': None}),
               'public_test_cases': public, 'private_test_cases': private_encoded}
        result = judge_cases._public_v6_case(row)
        self.assertEqual(json.loads(json.loads(result)['input_output'])['inputs'],
                         ['2', '3' * 260])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / 'test6.jsonl'
            variants = [{**row, 'question_id': f'unit_{i:03d}'} for i in range(175)]
            raw.write_text(''.join(json.dumps(item) + '\n' for item in variants))
            pinned = root / 'pinned.json'
            pinned.write_text(json.dumps({'test6_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
                'case_sha256': {'atcoder__unit_' + f'{i:03d}' + '.json':
                                hashlib.sha256(result).hexdigest() for i in range(175)},
                'livecodebench_commit': 'original-test-commit'}))
            with patch_object.object(judge_cases, 'PINNED', pinned):
                with self.assertRaisesRegex(RuntimeError, 'SHA differs'):
                    wrong = root / 'wrong.jsonl'
                    wrong.write_text(raw.read_text() + '\n')
                    judge_cases.materialize_public_v6(wrong, root / 'must-not-create')
                self.assertFalse((root / 'must-not-create').exists())
                target = root / 'original-public-cases'
                self.assertEqual(judge_cases.materialize_public_v6(raw, target), 175)
            self.assertEqual(json.loads((target / 'complete.json').read_text())['cases'], 175)
            self.assertEqual((target / 'atcoder__unit_000.json').read_bytes(), result)
        malicious = base64.b64encode(zlib.compress(b'\x80\x04cos\nsystem\n.')).decode()
        with self.assertRaisesRegex(ValueError, 'not a plain JSON string pickle'):
            judge_cases._original_test_rows(malicious)

    def test_public_judge_extractor_rejects_tampered_checker_before_output(self):
        from qwen35.cli import DATA
        from qwen35.judge_cases import prepare

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checker = root / 'checker'
            lcb = root / 'authorized-lcb'
            checker.mkdir()
            lcb.mkdir()
            (checker / 'testing_util.py').write_text('not the original checker')
            output = root / 'new-case-root'
            with self.assertRaisesRegex(RuntimeError, 'Official checker'):
                prepare(DATA, lcb, checker, output)
            self.assertFalse(output.exists())

    def test_shared_judge_refuses_changed_image_and_sealed_case(self):
        from qwen35.judge_service import JudgeService, ORIGINAL_IMAGE

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = root / 'cases'
            checker = root / 'checker'
            cases.mkdir()
            checker.mkdir()
            case = cases / 'sample_1.json'
            case.write_text('{}')
            official = checker / 'testing_util.py'
            official.write_text('')
            extractor = root / 'utils/extraction_utils.py'
            extractor.parent.mkdir()
            extractor.write_text('original-extractor')
            proof = root / 'complete.json'
            proof.write_text('{"complete":true}')
            service = JudgeService.__new__(JudgeService)
            service.cases, service.checker = cases, checker
            service.checker_hash, service.case_proof_sha = sha256(official), sha256(proof)
            service.extractor_hash = sha256(extractor)
            service.image, service.image_id = 'lcb-checker:local', ORIGINAL_IMAGE
            service.spec_id = 'a' * 64
            service.info = {'instance': 'b' * 32}
            service.deepcoder = {'sample_1': (sha256(case), 1)}
            service.tests = {'lcb': set(), 'humaneval': set(), 'mbpp': set()}
            service.lcb = {}
            judged = []
            service.modules = {'26': SimpleNamespace(judge=lambda spec, answer:
                (judged.append((spec['kind'], answer)) or {'correct': True}))}
            request = {'spec_id': service.spec_id, 'instance': service.info['instance'],
                       'experiment': '26', 'response': 'example',
                       'spec': {'kind': 'deepcoder', 'key': 'sample_1',
                                'sample_sha256': sha256(case), 'tests': 1}}
            with patch('qwen35.judge_service.subprocess.run', return_value=SimpleNamespace(
                    returncode=0, stdout=ORIGINAL_IMAGE + '\n')) as inspect:
                self.assertEqual(service.score(request), 1)
                self.assertEqual(judged, [('lcb', 'example')])
                inspect.assert_called_once()
            with patch('qwen35.judge_service.subprocess.run', return_value=SimpleNamespace(
                    returncode=0, stdout='sha256:' + '0' * 64)):
                with self.assertRaisesRegex(RuntimeError, 'Pinned isolated image'):
                    service.score(request)
            extractor.write_text('modified-extractor')
            with patch('qwen35.judge_service.subprocess.run', return_value=SimpleNamespace(
                    returncode=0, stdout=ORIGINAL_IMAGE + '\n')):
                with self.assertRaisesRegex(RuntimeError, 'extractor'):
                    service.score(request)
            extractor.write_text('original-extractor')
            case.write_text('{"changed":true}')
            with patch('qwen35.judge_service.subprocess.run', return_value=SimpleNamespace(
                    returncode=0, stdout=ORIGINAL_IMAGE + '\n')):
                with self.assertRaisesRegex(RuntimeError, 'case SHA differs'):
                    service.score(request)

    def test_single_host_deepcoder_requires_isolated_judge_before_model_loading(self):
        from qwen35.cli import run_train, verify_code_training_environment

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cases = base / 'cases'
            checker = base / 'official'
            cases.mkdir()
            checker.mkdir()
            (checker / 'testing_util.py').write_text('')
            args = SimpleNamespace(output=base / 'new-run', data_root=base,
                                   code_cases=cases, official_checker=checker,
                                   docker_image='lcb-checker:local')
            with patch('qwen35.cli.subprocess.run', return_value=SimpleNamespace(returncode=1)), \
                 patch('qwen35.cli._model') as model:
                with self.assertRaisesRegex(RuntimeError, 'daemon or image is unavailable'):
                    run_train(select(26), args)
                model.assert_not_called()
                self.assertFalse(args.output.exists())
            (cases / 'case_0.json').write_text('{}')
            digest = sha256(cases / 'case_0.json')
            signed = {'train': [{'index': 0, 'judge': {'kind': 'deepcoder',
                       'key': 'case_0', 'sample_sha256': digest}}], 'validation': []}
            with patch('qwen35.cli.subprocess.run', return_value=SimpleNamespace(
                    returncode=0, stdout='sha256:' + 'a' * 64 + '\n')), \
                 patch('qwen35.cli.inputs', return_value=signed):
                identity = verify_code_training_environment(select(30), args)
                self.assertEqual(len(identity), 64)
                (cases / 'case_0.json').write_text('{"modified":true}')
                with self.assertRaisesRegex(RuntimeError, 'case SHA differs'):
                    verify_code_training_environment(select(30), args)
                (cases / 'case_0.json').unlink()
                with self.assertRaisesRegex(FileNotFoundError, 'signed DeepCoder train cases'):
                    verify_code_training_environment(select(30), args)
            args.code_cases = None
            with self.assertRaisesRegex(ValueError, 'authorized cases'):
                verify_code_training_environment(select(27), args)

    def test_deepcoder_lcb_test_preflight_requires_all_original_cases(self):
        from qwen35.cli import run_eval, verify_code_evaluation_environment

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data = base / 'input/deepcoder'
            cases = base / 'cases'
            checker = base / 'checker'
            for directory in (data, cases, checker):
                directory.mkdir(parents=True)
            (checker / 'testing_util.py').write_text('')
            lines = [json.dumps({'index': index, 'prompt_token_ids': [1], 'gold': '',
                                 'judge': {'kind': 'lcb', 'key': f'case_{index}'}}) + '\n'
                     for index in range(175)]
            source = data / 'lcb.jsonl'
            source.write_text(''.join(lines))
            manifest = {'benchmark': {'lcb': {'sha256': sha256(source), 'rows': 175}}}
            (data.parent / 'manifest.json').write_text(json.dumps(manifest))
            args = SimpleNamespace(code_cases=cases, official_checker=checker,
                                   docker_image='lcb-checker:local', data_root=data.parent,
                                   weights=base / 'private-model.safetensors',
                                   receiver=base / 'receiver', output=base / 'no-evaluation')
            for index in range(175):
                (cases / f'case_{index}.json').write_text('{}')
            original = cases / 'case_174.json'
            original.unlink()
            with patch('qwen35.cli.subprocess.run', return_value=SimpleNamespace(
                    returncode=0, stdout='sha256:' + 'a' * 64 + '\n')), \
                 patch('qwen35.cli._model') as model:
                with self.assertRaisesRegex(FileNotFoundError, 'LCB v6 evaluation cases'):
                    run_eval(select(26), args)
                model.assert_not_called()
                self.assertFalse(args.output.exists())
                original.write_text('{}')
                verify_code_evaluation_environment(select(29), args)
            args.official_checker = None
            with self.assertRaisesRegex(ValueError, 'authorized cases'):
                verify_code_evaluation_environment(select(26), args)

    def test_native20_two_host_math_preflight_is_read_only(self):
        from qwen35.cli import DATA
        from qwen35.native_math20 import audit, verify_bundle

        self.assertEqual(len(verify_bundle()), 42)
        signed = inputs(select(20))
        report = audit(select(20), signed, DATA)
        self.assertEqual((report['required_hosts'], report['gpus_per_host'],
                          report['distributed_world_size'],
                          report['teacher_tensor_parallel_size'], report['total_steps'],
                          report['training_questions'], report['validation_questions']),
                         (2, 8, 4, 4, 264, 16891, 500))
        self.assertFalse(report['launch_enabled'])
        self.assertFalse(report['historical_397b_training_launcher_bundled'])
        with self.assertRaisesRegex(ValueError, 'only to mathematics ID20'):
            audit(select(25), signed, DATA)
        changed = dict(signed, validation=[dict(row) for row in signed['validation']])
        changed['validation'][0]['gold'] = 'synthetic-tampered-answer'
        with self.assertRaisesRegex(ValueError, 'token or gold differs'):
            audit(select(20), changed, DATA)

    def test_native25_two_host_deepcoder_preflight_never_launches(self):
        from qwen35.cli import DATA
        from qwen35.native_deepcoder import audit, verify_bundle

        self.assertEqual(len(verify_bundle()), 41)
        report = audit(select(25), inputs(select(25)), DATA)
        self.assertEqual((report['required_hosts'], report['gpus_per_host'],
                          report['training_questions'], report['validation_questions'],
                          report['total_steps']), (2, 8, 18873, 201, 295))
        self.assertFalse(report['launch_enabled'])
        self.assertFalse(report['judge_cases_sha256_verified'])
        self.assertIn('judge', ' '.join(report['blockers']))
        with self.assertRaisesRegex(ValueError, 'specific to DeepCoder experiment 25'):
            audit(select(27), inputs(select(25)), DATA)

    def test_new_native20_25_two_host_plan_and_staging_never_start_gpu(self):
        from qwen35.cli import DATA
        from qwen35.native_multinode import (prepare, torchrun_command,
                                             verify_prepared, _control)
        from qwen35.judge_service import ORIGINAL_IMAGE
        with self.assertRaisesRegex(ValueError, 'loopback'):
            torchrun_command('private', 0, '127.0.0.1', 29500, 'new_attempt')
        for experiment in (20, 25):
            with self.subTest(experiment=experiment), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                models = directory / 'models'
                sender = models / 'Qwen3.5-397B-A17B-FP8'
                receiver = models / 'Qwen3.5-4B'
                for model in (sender, receiver):
                    model.mkdir(parents=True)
                    (model / 'config.json').write_text('{}')
                judge = {'supported_experiments': ['25', '26', '29'],
                         'image_id': ORIGINAL_IMAGE, 'image': 'lcb-checker:local',
                         'spec_id': 'signed-private-judge', 'checker_sha256': 'a' * 64,
                         'case_proof_sha256': 'b' * 64, 'case_root': str(directory / 'cases')}
                with patch('qwen35.native_multinode.judge_service_info', return_value=judge), \
                     patch('qwen35.native_deepcoder.verify_materialized', return_value=19074) as verify_cases:
                    root = prepare(experiment, sender, receiver, directory / 'native',
                                   service=directory / 'service' if experiment == 25 else None)
                    protocol = verify_prepared(root, experiment)
                    if experiment == 25:
                        verify_cases.assert_called_once_with(DATA, judge['case_root'])
                    self.assertEqual(protocol['total_steps'], select(experiment).steps)
                    self.assertEqual(protocol['world_size'], 4)
                    self.assertEqual(protocol['initialization'], 'zero')
                    if experiment == 25:
                        import importlib.util
                        from qwen35.native_multinode_training_policy import training_seed
                        sealed_support = read(root / 'source_manifest.json')['support_source_sha256']
                        self.assertIn('native_multinode_training_policy.py', sealed_support)
                        with patch('qwen35.native_multinode._support_sources',
                                   return_value={'native_multinode_training_policy.py': '0' * 64}):
                            with self.assertRaisesRegex(RuntimeError, 'science, data or launcher SHA'):
                                verify_prepared(root, experiment)
                        seed = training_seed(0, 1, 123, 7)
                        self.assertEqual(seed, 42 + 1000 + 8 * 123 + 7)
                        shim = root / 'reference_source/recipe/llm_communicate/training_policy.py'
                        spec = importlib.util.spec_from_file_location('private_code_policy', shim)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        self.assertEqual(module.training_seed(0, 1, 123, 7), seed)
                        with self.assertRaisesRegex(ValueError, 'one of eight replies'):
                            module.training_seed(0, 1, 123, 8)
                    self.assertIs(read(root / 'prepare.json')['no_gpu_process_started'], True)
                    self.assertFalse((root / 'train').exists())
                    first = torchrun_command(root, 0, '192.0.2.42', 29500, 'first')
                    second = torchrun_command(root, 1, '192.0.2.42', 29500, 'first')
                    self.assertEqual(first[first.index('--nnodes=2') + 1], '--nproc-per-node=2')
                    self.assertIn('--node-rank=1', second)
                    with self.assertRaises(FileNotFoundError):
                        _control(root, experiment, 'train', 'first', None)
                    name = 'recipe/llm_communicate/qwen35_multinode_config.py'
                    (root / 'reference_source' / name).write_text('tampered')
                    with self.assertRaisesRegex(RuntimeError, 'science source changed'):
                        verify_prepared(root, experiment)

    def test_two_host_offline_ranks_get_the_complete_pinned_fp8_kernel_snapshot(self):
        from huggingface_hub.errors import LocalEntryNotFoundError
        from qwen35 import native_multinode
        calls = []

        def snapshot(repo, *, repo_type, revision, cache_dir, local_files_only=False):
            calls.append(local_files_only)
            self.assertEqual((repo, repo_type, revision), (*native_multinode.FP8_KERNEL[:1], 'kernel',
                                                           native_multinode.FP8_KERNEL[1]))
            if local_files_only and not (Path(cache_dir) / 'complete').exists():
                raise LocalEntryNotFoundError('partial snapshot')
            (Path(cache_dir) / 'complete').write_text('x')
            return str(cache_dir)
        with tempfile.TemporaryDirectory() as directory, \
             patch('huggingface_hub.snapshot_download', side_effect=snapshot):
            cache = Path(directory) / 'runtime_cache/fp8_kernel'
            self.assertEqual(native_multinode.fp8_kernel_cache(cache), str(cache))
            self.assertEqual(calls, [True, False])  # complete snapshot fetched once
            native_multinode.fp8_kernel_cache(cache)
            self.assertEqual(calls, [True, False, True])  # later stages stay offline
        with tempfile.TemporaryDirectory() as directory, \
             patch('huggingface_hub.snapshot_download',
                   side_effect=LocalEntryNotFoundError('offline and absent')), \
             self.assertRaisesRegex(RuntimeError, 'with Hugging Face Hub access'):
            native_multinode.fp8_kernel_cache(Path(directory) / 'cache')
        class Launched(Exception):
            pass

        for node in (0, 1):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'run'
                (root / 'reference_source').mkdir(parents=True)
                cache = root / 'runtime_cache/fp8_kernel'
                atomic_json(root / 'protocol.json', {'fp8_kernel_cache': str(cache)})
                if node == 1:
                    (root / 'invocations/a1').mkdir(parents=True)
                    atomic_json(root / 'invocations/a1/control.json', {
                        'experiment': 20, 'mode': 'probe', 'resume_step': None,
                        'root': str(root.absolute()), 'nonce': 'n'})
                seen = {}

                def popen(command, *, env, **_kwargs):
                    seen.update(command=command, env=env)
                    raise Launched()
                with patch.object(native_multinode, '_check_local_gpus'), \
                     patch.object(native_multinode, '_interface', return_value='eth0'), \
                     patch.object(native_multinode, '_control',
                                  return_value={'mode': 'probe', 'resume_step': None,
                                                'root': str(root.absolute()), 'nonce': 'n'}), \
                     patch.object(native_multinode, 'fp8_kernel_cache',
                                  side_effect=lambda path: str(path)) as kernel, \
                     patch('qwen35.native_multinode.subprocess.Popen', side_effect=popen), \
                     self.assertRaises(Launched):
                    native_multinode.run(root, 20, node, '192.0.2.1', 29500, 'probe', 'a1',
                                         bridge_python=sys.executable,
                                         generation_python=sys.executable,
                                         scoring_python=sys.executable)
                kernel.assert_called_once_with(str(cache))
                self.assertEqual(seen['env']['KERNELS_CACHE'], str(cache))
                self.assertEqual(seen['env']['HF_HUB_OFFLINE'], '1')
                self.assertTrue((root / f'invocations/a1/failure-node-{node}.json').is_file())

    def test_portable_training_refuses_a_397b_fp8_sender_under_torchrun(self):
        from qwen35.cli import run_train
        case = SimpleNamespace(identifier=20, tag='397b', task='dapo', sender='Qwen3.5-397B-A17B-FP8')
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'never-created'
            arguments = portable_args(output=destination)
            with patch('qwen35.distributed.init', return_value=(0, 16, 0)), \
                 self.assertRaisesRegex(ValueError, 'native-multinode for IDs 20/25'):
                run_train(case, arguments)
            self.assertFalse(destination.exists())

    def test_rolling_retention_sweeps_a_stale_checkpoint_left_by_a_crash(self):
        """A crash between signing n and deleting n-2 leaves n-2; the next signing removes it."""
        import ast
        import shutil
        from qwen35 import checkpoints as portable, native_formal, native_multinode
        from qwen35.native_training import ASSETS

        def generated(identifier):
            trainer = (Path(ASSETS).parent / str(identifier) /
                       'recipe/llm_communicate/train_qwen35_bridge_opd_vllm.py').read_text()
            body = (native_multinode._private_math_trainer(trainer) if identifier == 20
                    else native_multinode._private_code_trainer(trainer))
            wanted = {'checkpoint_content_sha', 'checkpoint_complete', 'retire_previous_checkpoint'}
            module = ast.Module([node for node in ast.parse(body).body
                                 if isinstance(node, ast.FunctionDef) and node.name in wanted], [])
            namespace = {'Path': Path, 'json': json, 'shutil': shutil, 'hashlib': hashlib}
            exec(compile(module, f'generated-{identifier}', 'exec'), namespace)
            return namespace['retire_previous_checkpoint']

        def native_step(root, step):
            point = root / 'checkpoints' / f'step-{step:04d}'
            point.mkdir(parents=True)
            for name in ('bridge.pt', 'optimizer.pt', *(f'rng.rank-{r:02d}.pt' for r in range(4))):
                (point / name).write_bytes(f'{name}-{step}'.encode())
            atomic_json(point / 'checkpoint.json', {'step': step, 'bridge_state_sha256': 'b' * 64,
                                                    'bridge_sha256': sha256(point / 'bridge.pt')})
            atomic_json(point / 'complete.json', {
                'complete': True, 'step': step, 'world_size': 4, 'bridge_state_sha256': 'b' * 64,
                'optimizer_sha256': sha256(point / 'optimizer.pt'),
                'rank_rng_sha256': {str(r): sha256(point / f'rng.rank-{r:02d}.pt') for r in range(4)}})

        names = lambda folder: sorted(p.name for p in folder.iterdir() if p.name.startswith('step-'))
        for identifier in (20, 25):
            with self.subTest(native=identifier), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                retire = generated(identifier)
                for step in (1, 2, 3, 4, 5):
                    native_step(root, step)
                (root / 'checkpoints/step-0006').mkdir()  # unsigned later directory: never touched
                # Steps 2 and 3 are stale (e.g. a crash during retirement); step 1 is a validation step.
                retire(root, 5, {0, 1}, 4)
                self.assertEqual(names(root / 'checkpoints'),
                                 ['step-0001', 'step-0004', 'step-0005', 'step-0006'])
                shutil.rmtree(root / 'checkpoints/step-0006')
                native_step(root, 6)
                (root / 'checkpoints/step-0004/complete.json').unlink()
                with self.assertRaisesRegex(RuntimeError, 'not signed|incomplete'):
                    retire(root, 6, {0, 1}, 4)  # n-2 unsigned: refuse instead of guessing

        class Killed(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:  # portable Qwen3.5 trainer
            output = Path(directory)
            actor = TinyActor()
            optimizer = torch.optim.AdamW(actor.trainable(), lr=.1)
            for step in (0, 1, 2):
                save_step(output, step, actor, optimizer, {'step': step}, protected={0}, identity='x:toy')
            real = shutil.rmtree
            with patch.object(portable.shutil, 'rmtree', side_effect=Killed('killed')), \
                 self.assertRaises(Killed):
                save_step(output, 3, actor, optimizer, {'step': 3}, protected={0}, identity='x:toy')
            self.assertEqual(names(output / 'checkpoints'),
                             ['step-0000', 'step-0001', 'step-0002', 'step-0003'])
            self.assertIs(portable.shutil.rmtree, real)
            save_step(output, 4, actor, optimizer, {'step': 4}, protected={0}, identity='x:toy')
            self.assertEqual(names(output / 'checkpoints'), ['step-0000', 'step-0003', 'step-0004'])

        with tempfile.TemporaryDirectory() as directory:  # single-host native 23/24 helper
            run = Path(directory)
            for step in (1, 2, 3, 4, 5):
                (run / 'checkpoints' / f'step-{step:04d}').mkdir(parents=True)
            with patch.object(native_formal, 'valid_checkpoint', return_value=True):
                native_formal.retire_predecessor(run, 'lora', 5, {0, 2})
            self.assertEqual(names(run / 'checkpoints'), ['step-0002', 'step-0004', 'step-0005'])

    def test_two_host_resume_uses_newest_complete_four_rank_step(self):
        from qwen35.native_multinode import _checkpoint_step
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'train/checkpoints'
            base.mkdir(parents=True)
            protocol = {'total_steps': 295, 'checkpoint_steps': [0, 42, 84],
                        'per_step_checkpoints': True}
            for step in (1, 2, 3):
                point = base / f'step-{step:04d}'
                point.mkdir()
                (point / 'bridge.pt').write_bytes(b'original-P-' + bytes([step]))
                (point / 'optimizer.pt').write_bytes(b'fp32-AdamW')
                for rank in range(4):
                    (point / f'rng.rank-{rank:02d}.pt').write_bytes(bytes([rank]))
                atomic_json(point / 'checkpoint.json', {
                    'step': step, 'bridge_sha256': sha256(point / 'bridge.pt'),
                    'bridge_state_sha256': 'a' * 64})
                if step < 3:
                    atomic_json(point / 'complete.json', {'complete': True, 'step': step,
                        'world_size': 4, 'bridge_state_sha256': 'a' * 64,
                        'optimizer_sha256': sha256(point / 'optimizer.pt'),
                        'rank_rng_sha256': {str(rank): sha256(point / f'rng.rank-{rank:02d}.pt')
                                            for rank in range(4)}})
            for experiment in (20, 25):
                self.assertEqual(_checkpoint_step(Path(directory), experiment, protocol), 2)
            optimizer = base / 'step-0002/optimizer.pt'
            optimizer.write_bytes(b'changed-AdamW')
            self.assertEqual(_checkpoint_step(Path(directory), 25, protocol), 1)
            optimizer.write_bytes(b'fp32-AdamW')
            rng = base / 'step-0002/rng.rank-03.pt'
            rng.write_bytes(b'changed-RNG')
            self.assertEqual(_checkpoint_step(Path(directory), 20, protocol), 1)
            rng.unlink()
            self.assertEqual(_checkpoint_step(Path(directory), 25, protocol), 1)
            (base / 'step-0001/complete.json').unlink()
            with self.assertRaisesRegex(RuntimeError, 'No newest signed'):
                _checkpoint_step(Path(directory), 25, protocol)

    def test_native_multinode_cli_stages_and_plans_without_starting_any_gpu(self):
        from qwen35.cli import main
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            command = ['platour-exps', 'train', '--experiment', '20',
                       '--sender', str(base / 'Qwen3.5-397B-A17B-FP8'),
                       '--receiver', str(base / 'Qwen3.5-4B'),
                       '--output', str(base / 'uncreated'), '--train-backend',
                       'native-multinode']
            with patch('sys.argv', [*command, '--native-stage', 'plan',
                                    '--master-addr', '192.0.2.42']), \
                 patch('qwen35.native_multinode.run') as launcher:
                from contextlib import redirect_stdout
                from io import StringIO
                output = StringIO()
                with redirect_stdout(output):
                    main()
                plan = json.loads(output.getvalue())
                self.assertFalse(plan['gpu_started'])
                self.assertEqual(len(plan['internal_torchrun_commands']), 2)
                launcher.assert_not_called()
            with patch('sys.argv', [*command, '--native-stage', 'prepare']), \
                 patch('qwen35.native_multinode.prepare', return_value=base / 'uncreated') as staged:
                main()
                staged.assert_called_once()
            with patch('sys.argv', [*command, '--native-stage', 'probe',
                                    '--master-addr', '192.0.2.42', '--node-rank', '0',
                                    '--vllm-python', 'native-python',
                                    '--scoring-python', 'scorer-python']), \
                 patch('platour_exps.require_vllm_python'), \
                 patch('qwen35.native_multinode.run') as launched:
                main()
                self.assertEqual(launched.call_args.args[5], 'probe')
                self.assertIsNone(launched.call_args.kwargs['resume_step'])

    def test_two_host_original_code_judge_adapter_scores_only_via_cpu_service(self):
        import qwen35.native_multinode_judge as native_judge
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_json(root / 'protocol.json', {'judge_service_spec_id': 'signed-service'})
            folder = root / 'train/score_worker'
            (root / 'train/rollouts/step-0001').mkdir(parents=True)
            payload = root / 'train/rollouts/step-0001/rank-00.json'
            atomic_json(payload, {'complete': True, 'rows': [
                {'index': 0, 'sample_index': 0, 'judge': {'kind': 'deepcoder'},
                 'response': 'plain-text', 'finish_reason': 'length', 'tokens': 16}]})
            reply = root / 'train/rollouts/step-0001/summary.json'
            info = {'spec_id': 'signed-service', 'case_root': str(root / 'cases'),
                    'checker_root': str(root / 'checker'), 'image': 'lcb-checker:local',
                    'image_id': native_judge.ORIGINAL_IMAGE,
                    'supported_experiments': ['25']}
            with patch.object(native_judge, 'SERVICE', root / 'service'), \
                 patch.object(native_judge, 'judge_service_info', return_value=info), \
                 patch.object(native_judge, '_has_code', return_value=True) as extract, \
                 patch.object(native_judge, 'code_score', return_value=1) as scorer:
                judge = native_judge.DeepCoderScorer(folder)
                result = judge.call({'inputs': [str(payload)], 'expected_responses': 1,
                                     'result': str(reply), 'format_penalty': True})
                scorer.assert_called_once()
                extract.assert_called_once_with('plain-text', info['checker_root'])
                self.assertEqual((result['correct'], result['accuracy_unpenalized'],
                                  result['clipped'], result['responses']), (0, 1.0, 1, 1))
                self.assertEqual(result['format_violation_rate'], 1.)
                self.assertEqual(result['no_code_rate'], 0.)
                self.assertEqual(len((reply.parent / 'scored.jsonl').read_text().splitlines()), 1)
                with self.assertRaises(ValueError):
                    judge.call({'inputs': ['/untrusted/outside.json'],
                                'expected_responses': 1, 'result': str(root / 'outside.json')})
                next_dir = root / 'train/rollouts/step-0002'
                next_dir.mkdir()
                next_payload = next_dir / 'rank-00.json'
                atomic_json(next_payload, {'complete': True, 'rows': [
                    {'index': 1, 'sample_index': 0, 'judge': {'kind': 'deepcoder'},
                     'response': 'no code', 'finish_reason': 'stop', 'tokens': 2}]})
                scorer.return_value = 0
                extract.return_value = False
                no_code = judge.call({'inputs': [str(next_payload)], 'expected_responses': 1,
                                      'result': str(next_dir / 'summary.json'),
                                      'format_penalty': True})
                self.assertEqual((no_code['no_code_rate'], no_code['format_violation_rate'],
                                  no_code['correct']), (1., 1., 0))

    def test_new_two_host_25_requires_actual_same_source_probe_and_smoke(self):
        from qwen35.native_multinode_qualification import verify_evidence
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'source_manifest.json').write_text('{"source":"signed"}')
            source_sha = sha256(root / 'source_manifest.json')
            atomic_json(root / 'probe_audit.json', {'passed': True, 'nccl_validated': True,
                'cpu_simulation': False, 'shared_storage_verified': True,
                'source_manifest_sha256': source_sha, 'hosts': ['one', 'two']})
            atomic_json(root / 'smoke_audit.json', {'passed': True,
                'source_manifest_sha256': source_sha,
                'structural': {'complete': True, 'world_size': 4,
                               'replica_exact_weights': True, 'nonzero_gradient_updates': 1},
                'worst_cases': [{'passed': True, 'vllm_resident': True,
                                 'response_tokens': 16384} for _ in range(4)]})
            proof = verify_evidence(root)
            self.assertEqual(proof['components_passed'], True)
            self.assertEqual(proof['sha256']['probe'], sha256(root / 'probe_audit.json'))
            self.assertEqual(verify_evidence(root, {'dataset': 'deepcoder',
                  'world_size': 4, 'qualification_evidence': proof}), proof)
            record = read(root / 'probe_audit.json')
            record['cpu_simulation'] = True
            atomic_json(root / 'probe_audit.json', record)
            with self.assertRaisesRegex(RuntimeError, 'NCCL'):
                verify_evidence(root)

    def test_every_bundled_native_source_is_valid_python(self):
        import ast
        from qwen35 import native_math20, native_training
        roots = [Path(native_training.ASSETS).parent,
                 Path(native_training.__file__).parents[1] / 'qwen3/native_reference']
        files = [path for root in roots for path in sorted(root.rglob('*.py'))]
        self.assertGreater(len(files), 200)
        for path in files:
            with self.subTest(path=str(path)):
                ast.parse(path.read_bytes(), filename=str(path))
        native_math20.verify_bundle()
        checker = native_math20.ASSETS / 'third_party/verl'
        script = ('import sys; sys.path.insert(0, sys.argv[1]); '
                  'import verl.utils.reward_score.math_verify as m; print(m.__file__)')
        import subprocess
        located = subprocess.run([sys.executable, '-c', script, str(checker)], capture_output=True,
                                 text=True, timeout=120)
        self.assertEqual(located.returncode, 0, located.stderr[-800:])
        imported = Path(located.stdout.strip().splitlines()[-1]).resolve()  # it logs a version line first
        self.assertEqual(imported, (checker / 'verl/utils/reward_score/math_verify.py').resolve())

    def test_native_math_scorers_find_the_original_verl_checker(self):
        import hashlib
        from qwen35 import native_full, native_lora, native_random, native_training
        root = Path(native_training.original_math_checker())
        checker = root / 'verl/utils/reward_score/math_verify.py'
        self.assertEqual(hashlib.sha256(checker.read_bytes()).hexdigest(),
                         native_training.MATH_CHECKER_SHA256)
        self.assertTrue((root / 'LICENSE').is_file())
        for identifier in (21, 22, 23, 24):
            worker = (Path(native_training.ASSETS).parent / str(identifier) /
                      'recipe/llm_communicate/qwen35_bridge_opd_score_worker.py').read_text()
            self.assertIn("third_party/verl", worker)
        for module in (native_training, native_random, native_full, native_lora):
            self.assertIn('original_math_checker()', Path(module.__file__).read_text())
        with patch.object(native_training, 'MATH_CHECKER_SHA256', '0' * 64), \
             self.assertRaisesRegex(ValueError, 'math_verify checker differs'):
            native_training.original_math_checker()

    def test_native21_source_and_signed_protocol_prepare(self):
        from qwen35.native_training import prepare, verify_bundle, verify_prepared

        self.assertEqual(len(verify_bundle()), 47)
        case = select(21)
        signed = inputs(case)
        with tempfile.TemporaryDirectory() as temporary, \
             patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}):
            base = Path(temporary)
            model = base / 'models/Qwen3.5-4B'
            model.mkdir(parents=True)
            (model / 'config.json').write_text('{}')
            output = base / 'native'
            prepare(case, model, model, signed, output, smoke=True)
            protocol = verify_prepared(output)
            self.assertEqual((protocol['world_size'], protocol['total_steps'],
                              protocol['eval_questions'], protocol['eval_samples']),
                             (4, 264, 500, 4))
            self.assertEqual(protocol['wandb_mode'], 'disabled')
            self.assertFalse((output / 'train/metrics.jsonl').exists())
            self.assertEqual(json.loads((output / 'train_prompts.json').read_text()),
                             signed['train'])
            with self.assertRaises(FileExistsError):
                prepare(case, model, model, signed, output, smoke=True)
            with self.assertRaisesRegex(ValueError, 'original sender'):
                prepare(select(19), model, model, signed, base / 'wrong-19-sender',
                        smoke=True)

    def test_native19_original_35b_source_protocol_and_cli_dispatch(self):
        from qwen35.native_training import prepare, verify_bundle, verify_prepared
        from qwen35.cli import main

        self.assertEqual(len(verify_bundle(19)), 36)
        case = select(19)
        signed = inputs(case)
        with tempfile.TemporaryDirectory() as temporary, \
             patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}):
            base = Path(temporary)
            sender = base / 'models/Qwen3.5-35B-A3B'
            receiver = base / 'models/Qwen3.5-4B'
            for model in (sender, receiver):
                model.mkdir(parents=True)
                (model / 'config.json').write_text('{}')
            output = base / 'native19'
            prepare(case, sender, receiver, signed, output, smoke=True)
            current = verify_prepared(output)
            self.assertEqual((current['native_experiment_id'], current['world_size'],
                              current['teacher_model_parallel_size'], current['total_steps'],
                              current['eval_questions'], current['eval_samples']),
                             (19, 8, 1, 264, 500, 4))
            self.assertEqual(current['layer_map'], list(case.layer_map))
            self.assertEqual(current['val_metric'], 'answer')
            self.assertEqual(current['decoder_gpu_ids'], list(range(8)))
            self.assertEqual(current['initialization'], 'exact-zero linear bridge')
            original = json.loads((output / 'source_manifest.json').read_text())['files']
            marker = json.loads((output / 'zero_bridge_initialization_patch.json').read_text())
            for name, active in marker['active_sha256'].items():
                self.assertEqual(marker['original_sha256'][name], original[name])
                self.assertEqual(sha256(output / 'reference_source' / name), active)
            self.assertIn('p.zero_()', (output / 'reference_source/recipe/llm_communicate/'
                                       'qwen35_bridge_opd_core.py').read_text())
            self.assertIn('uniform +/-1e-4', current['historical_initialization'])
            seal = output / 'zero_bridge_initialization_patch.json'
            original_seal = seal.read_text()
            seal.write_text(json.dumps({**marker, 'active_sha256': {}}))
            with self.assertRaisesRegex(ValueError, 'zero source seal'):
                verify_prepared(output)
            seal.write_text(original_seal)
            self.assertTrue((output / 'reference_source/third_party/verl/LICENSE').is_file())
            self.assertTrue((output / 'reference_source/recipe/llm_communicate/'
                             'qwen35_hybrid_memory_connector.py').is_file())
            with patch('sys.argv', ['platour-exps', 'train', '--experiment', '19',
                     '--sender', str(sender), '--receiver', str(receiver),
                     '--train-backend', 'historical-native', '--smoke',
                     '--vllm-python', 'native-python', '--scoring-python', 'math-python',
                     '--output', str(base / 'cli19')]), \
                 patch('platour_exps.require_vllm_python'), \
                 patch('qwen35.cli.inputs', return_value=signed), \
                 patch('qwen35.native_training.prepare') as submitted, \
                 patch('qwen35.native_training.execute') as executed:
                main()
                submitted.assert_called_once_with(case, sender, receiver, signed,
                                                   base / 'cli19', smoke=True)
                executed.assert_called_once()
            current['learning_rate'] *= 2
            (output / 'protocol.json').write_text(json.dumps(current))
            with self.assertRaisesRegex(ValueError, 'protocol changed'):
                verify_prepared(output)

    def test_native19_formal_resume_preflights_500x4_and_rank_rng(self):
        from qwen35.cli import main
        from qwen35.native_training import digest, prepare, verify_prepared, verify_resume

        case = select(19)
        signed = inputs(case)
        with tempfile.TemporaryDirectory() as temporary, \
             patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}):
            base = Path(temporary)
            sender = base / 'models/Qwen3.5-35B-A3B'
            receiver = base / 'models/Qwen3.5-4B'
            for model in (sender, receiver):
                model.mkdir(parents=True)
                (model / 'config.json').write_text('{}')
            root = base / 'native'
            prepare(case, sender, receiver, signed, root, smoke=False)
            protocol = verify_prepared(root)
            step = 33
            checkpoint = root / 'train/checkpoints/step-0033'
            checkpoint.mkdir(parents=True)
            initial = 'a' * 64
            bridge = 'b' * 64
            atomic_json(root / 'train/initialization.json', {'bridge_state_sha256': initial})
            torch.save({'P': {'0.weight': torch.ones(2, 2)}}, checkpoint / 'bridge.pt')
            torch.save({'step': step, 'initial_sha256': initial,
                        'optimizer': {'state': {}, 'param_groups': []}},
                       checkpoint / 'optimizer.pt')
            for rank in range(8):
                torch.save({'cpu_rng': torch.get_rng_state(),
                            'cuda_rng': torch.get_rng_state()},
                           checkpoint / f'rng.rank-{rank:02d}.pt')
            atomic_json(checkpoint / 'checkpoint.json', {
                'step': step, 'bridge_sha256': digest(checkpoint / 'bridge.pt'),
                'bridge_state_sha256': bridge,
                'initial_bridge_state_sha256': initial})
            for boundary in (0, step):
                dest = root / 'train/evaluation' / f'step-{boundary:04d}'
                dest.mkdir(parents=True)
                atomic_json(dest / 'summary.json', {'complete': True,
                    'step': boundary, 'questions': 500, 'responses': 2000,
                    'scored_responses': 2000, 'gold_missing': 0,
                    'scoring_timeouts': 0,
                    'bridge_state_sha256': bridge if boundary else initial})
            metrics = root / 'train/metrics.jsonl'
            metrics.write_text(''.join(json.dumps({'step': i}) + '\n'
                                       for i in range(1, step + 1)) + '{"step":34')
            self.assertEqual(verify_resume(root, protocol, step), checkpoint)
            self.assertEqual(len(metrics.read_text().splitlines()), step)
            self.assertEqual(len(list((root / 'train/resume_archive').glob('metrics-torn-*'))), 1)
            argv = ['platour-exps', 'train', '--experiment', '19',
                    '--sender', str(sender), '--receiver', str(receiver),
                    '--train-backend', 'historical-native', '--resume-step', str(step),
                    '--vllm-python', 'native-python', '--scoring-python', 'math-python',
                    '--output', str(root)]
            with patch('platour_exps.require_vllm_python'), patch('sys.argv', argv), patch('platour_exps.require_vllm_python'), \
                 patch('qwen35.cli.inputs', return_value=signed), \
                 patch('qwen35.native_training.prepare') as fresh, \
                 patch('qwen35.native_training.execute') as continued:
                main()
                fresh.assert_not_called()
                self.assertEqual(continued.call_args.kwargs['resume_step'], step)
            with (checkpoint / 'bridge.pt').open('ab') as stream:
                stream.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'checkpoint identity differs'):
                verify_resume(root, protocol, step)
            with self.assertRaisesRegex(ValueError, 'incomplete formal run'):
                verify_resume(root, dict(protocol, smoke_only=True), step)

    def test_native22_train_only_moments_prepare_without_private_files(self):
        from qwen35.cli import DATA
        from qwen35.native_random import prepare, verify_bundle, verify_prepared
        from qwen35.random_init import save_native_statistics, template

        self.assertEqual(len(verify_bundle()), 49)
        case = select(22)
        signed = inputs(case)
        prefix, suffix, fixed = template(signed['train'])
        chosen = sorted(sorted(signed['train'], key=lambda row: hashlib.sha256(
            f'42:{row["index"]}'.encode()).hexdigest())[:500],
            key=lambda row: row['index'])
        with tempfile.TemporaryDirectory() as temporary, \
             patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}):
            base = Path(temporary)
            model = base / 'models/Qwen3.5-4B'
            model.mkdir(parents=True)
            (model / 'config.json').write_text('{}')
            (model / 'model.safetensors.index.json').write_text('{}')
            shard_names = ('model.safetensors-00001-of-00002.safetensors',
                           'model.safetensors-00002-of-00002.safetensors')
            for name in shard_names:
                (model / name).write_bytes(name.encode())
            stats = base / 'private-train-only/dapo'
            save_native_statistics(case, model, DATA, chosen, prefix, suffix, fixed,
                                   torch.zeros(63, 32, 1), torch.ones(63, 32, 1),
                                   stats.parent)
            output = base / 'new-native'
            prepare(case, model, signed, stats.parent, output, smoke=True)
            protocol = verify_prepared(output)
            self.assertEqual((protocol['world_size'], protocol['total_steps'],
                              protocol['eval_questions'], protocol['eval_samples']),
                             (8, 264, 500, 4))
            self.assertEqual(protocol['statistics_sample_count'], 500)
            self.assertFalse((output / 'train/metrics.jsonl').exists())
            with self.assertRaises(FileExistsError):
                prepare(case, model, signed, stats.parent, output, smoke=True)
            (model / shard_names[0]).write_bytes(b'different-weights')
            with self.assertRaisesRegex(ValueError, 'exact 4B model weights'):
                prepare(case, model, signed, stats.parent, base / 'rejected',
                        smoke=True)
            self.assertFalse((base / 'rejected').exists())

    def test_native22_formal_resume_reuses_signed_train_only_statistics(self):
        from qwen35.cli import DATA, main
        from qwen35.native_random import prepare, verify_prepared
        from qwen35.random_init import save_native_statistics, template

        case = select(22)
        signed = inputs(case)
        prefix, suffix, fixed = template(signed['train'])
        chosen = sorted(sorted(signed['train'], key=lambda row: hashlib.sha256(
            f'42:{row["index"]}'.encode()).hexdigest())[:500],
            key=lambda row: row['index'])
        with tempfile.TemporaryDirectory() as temporary, \
             patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}):
            base = Path(temporary)
            model = base / 'models/Qwen3.5-4B'
            model.mkdir(parents=True)
            (model / 'config.json').write_text('{}')
            (model / 'model.safetensors.index.json').write_text('{}')
            for index in (1, 2):
                name = f'model.safetensors-{index:05d}-of-00002.safetensors'
                (model / name).write_bytes(name.encode())
            stats = base / 'original-train-statistics'
            save_native_statistics(case, model, DATA, chosen, prefix, suffix, fixed,
                                   torch.zeros(63, 32, 1), torch.ones(63, 32, 1), stats)
            output = base / 'formal-virtual-sender'
            prepare(case, model, signed, stats, output, smoke=False)
            self.assertEqual(verify_prepared(output)['world_size'], 8)
            argv = ['platour-exps', 'train', '--experiment', '22',
                    '--receiver', str(model), '--train-backend', 'historical-native',
                    '--resume-step', '33', '--vllm-python', 'native-python',
                    '--scoring-python', 'math-python', '--output', str(output)]
            with patch('platour_exps.require_vllm_python'), patch('sys.argv', argv), patch('platour_exps.require_vllm_python'), \
                 patch('qwen35.cli.inputs', return_value=signed), \
                 patch('qwen35.native_random.prepare') as fresh, \
                 patch('qwen35.native_random.execute') as restored:
                main()
                fresh.assert_not_called()
                self.assertEqual(restored.call_args.kwargs['resume_step'], 33)
            different = base / 'models/unrelated-4B'
            with patch('sys.argv', [value if value != str(model) else str(different)
                                    for value in argv]), \
                 patch('platour_exps.require_vllm_python'), \
                 patch('qwen35.cli.inputs', return_value=signed):
                with self.assertRaisesRegex(ValueError, 'model or TRAIN-only moments differ'):
                    main()

    def test_native23_offline_full_parameter_smoke_and_formal_preparation(self):
        from qwen35.native_full import prepare, verify_bundle, verify_prepared

        self.assertEqual(len(verify_bundle()), 35)
        frozen = Path(__file__).resolve().parents[1] / 'qwen35/native_reference/23'
        trainer = (frozen / 'recipe/llm_communicate/full4b_train.py').read_text()
        self.assertIn("mode='disabled'", trainer)
        self.assertNotIn("mode='online'", trainer)
        self.assertNotIn('queued_wandb.json', trainer)
        case = select(23)
        signed = inputs(case)
        with tempfile.TemporaryDirectory() as temporary, \
             patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}):
            base = Path(temporary)
            model = base / 'models/Qwen3.5-4B'
            model.mkdir(parents=True)
            (model / 'config.json').write_text('{}')
            container = base / 'full-native'
            prepare(case, model, signed, container, smoke=True)
            protocol = verify_prepared(container)
            self.assertEqual((protocol['world_size'], protocol['total_steps'],
                              protocol['eval_questions'], protocol['eval_samples'],
                              protocol['learning_rate']), (8, 264, 500, 4, 1e-6))
            self.assertFalse((container / 'run/smoke/metrics.jsonl').exists())
            formal = base / 'native-formal'
            prepare(case, model, signed, formal, smoke=False)
            current = verify_prepared(formal)
            self.assertIs(current['smoke_only'], False)
            self.assertTrue(current['per_step_checkpoints'])
            self.assertEqual(current['rolling_predecessor'], 1)
            self.assertEqual(current['eval_steps'], list(case.validation_steps))
            patch_record = read(formal / 'formal_source_patch.json')
            self.assertEqual(set(patch_record['active_sha256']),
                             {'recipe/llm_communicate/full4b_train.py'})
            name = next(iter(patch_record['active_sha256']))
            self.assertEqual(sha256(formal / 'reference_source' / name),
                             patch_record['active_sha256'][name])
            self.assertEqual(sha256(frozen / name), patch_record['original_sha256'][name])
            (formal / 'formal_source_patch.json').write_text('{"experiment":23}')
            with self.assertRaisesRegex(ValueError, 'source seal differs'):
                verify_prepared(formal)

    def test_native24_offline_lora_rank128_smoke_and_formal_preparation(self):
        from qwen35.native_lora import prepare, verify_bundle, verify_prepared

        self.assertEqual(len(verify_bundle()), 42)
        source = Path(__file__).resolve().parents[1] / 'qwen35/native_reference/24'
        for name in ('lora4b_train.py', 'full4b_train.py'):
            text = (source / 'recipe/llm_communicate' / name).read_text()
            self.assertNotIn("mode='online'", text)
            self.assertNotIn('queued_wandb.json', text)
        case = select(24)
        signed = inputs(case)
        with tempfile.TemporaryDirectory() as temporary, \
             patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}):
            base = Path(temporary)
            model = base / 'models/Qwen3.5-4B'
            model.mkdir(parents=True)
            (model / 'config.json').write_text('{}')
            prepare(case, model, signed, base / 'lora-smoke', smoke=True)
            protocol = verify_prepared(base / 'lora-smoke')
            self.assertEqual((protocol['world_size'], protocol['total_steps'],
                              protocol['lora_rank'], protocol['lora_alpha'],
                              protocol['learning_rate']), (8, 264, 128, 128, 1e-5))
            formal = base / 'native-formal'
            prepare(case, model, signed, formal, smoke=False)
            current = verify_prepared(formal)
            self.assertIs(current['smoke_only'], False)
            self.assertTrue(current['per_step_checkpoints'])
            self.assertEqual(current['rolling_predecessor'], 1)
            self.assertEqual(current['eval_steps'], list(case.validation_steps))
            original = read(formal / 'formal_source_patch.json')
            self.assertEqual(len(original['active_sha256']), 2)
            for name, value in original['active_sha256'].items():
                self.assertEqual(sha256(formal / 'reference_source' / name), value)
                self.assertEqual(sha256(source / name), original['original_sha256'][name])
            (formal / 'formal_source_patch.json').write_text('{"experiment":24}')
            with self.assertRaisesRegex(ValueError, 'source seal differs'):
                verify_prepared(formal)

    def test_native_full_and_lora_per_step_signature_rolling_and_resume_tail(self):
        from qwen35.native_formal import (archive_resume_tail, newest_checkpoint,
                                          sign_checkpoint, valid_checkpoint)
        for kind in ('full', 'lora'):
            with self.subTest(method=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run = root / 'train'
                atomic_json(root / 'protocol.json', {'per_step_checkpoints': True,
                                                      'eval_steps': [2]})
                (run / 'checkpoints').mkdir(parents=True)
                for step in range(4):
                    checkpoint = run / 'checkpoints' / f'step-{step:04d}'
                    checkpoint.mkdir()
                    files = {}
                    if kind == 'full':
                        model = run / 'weights' / checkpoint.name / 'model.safetensors'
                        model.parent.mkdir(parents=True)
                        model.write_bytes(f'weights-{step}'.encode())
                        files.update(weights=str(model), weights_sha256=sha256(model))
                        for rank in range(8):
                            (checkpoint / f'optimizer.rank-{rank:02d}.pt').write_bytes(
                                f'optimizer-{rank}-{step}'.encode())
                    else:
                        for name in ('adapter.pt', 'optimizer.pt'):
                            (checkpoint / name).write_bytes(f'{name}-{step}'.encode())
                        files.update(adapter_sha256=sha256(checkpoint / 'adapter.pt'),
                                     optimizer_sha256=sha256(checkpoint / 'optimizer.pt'))
                    for rank in range(8):
                        (checkpoint / f'rng.rank-{rank:02d}.pt').write_bytes(
                            f'rng-{rank}-{step}'.encode())
                    atomic_json(checkpoint / 'checkpoint.json', {'complete': True,
                        'step': step, 'world_size': 8, **files})
                    sign_checkpoint(checkpoint, kind, step, 8)
                    self.assertTrue(valid_checkpoint(checkpoint, kind, step))
                self.assertFalse((run / 'checkpoints/step-0001').exists())
                self.assertTrue((run / 'checkpoints/step-0000').exists())
                self.assertTrue((run / 'checkpoints/step-0002').exists())
                self.assertEqual(newest_checkpoint(run, kind, 264), 3)
                (run / 'metrics.jsonl').write_text(''.join(
                    json.dumps({'step': step}) + '\n' for step in (1, 2, 3)))
                last_rng = run / 'checkpoints/step-0003/rng.rank-00.pt'
                with last_rng.open('ab') as stream:
                    stream.write(b'corrupt')
                self.assertFalse(valid_checkpoint(last_rng.parent, kind, 3))
                self.assertEqual(newest_checkpoint(run, kind, 264), 2)
                archived = archive_resume_tail(run, 2)
                self.assertTrue((archived / 'checkpoints/step-0003').is_dir())
                self.assertEqual([json.loads(line)['step'] for line in
                    (run / 'metrics.jsonl').read_text().splitlines()], [1, 2])
                self.assertTrue((archived / 'metrics-before-resume.jsonl').is_file())

    def test_native_23_24_formal_supervisor_never_starts_unsigned_resume(self):
        import sys
        from platour_exps import run_native_training
        for identifier, package in ((23, 'native_full'), (24, 'native_lora')):
            with self.subTest(experiment=identifier), tempfile.TemporaryDirectory() as directory, \
                 patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}):
                base = Path(directory)
                model = base / 'models/Qwen3.5-4B'
                model.mkdir(parents=True)
                (model / 'config.json').write_text('{}')
                case = select(identifier)
                module = __import__('qwen35.' + package, fromlist=['prepare', 'execute'])
                root = module.prepare(case, model, inputs(case), base / 'formal', smoke=False)
                with patch('platour_exps.run_native_training',
                           side_effect=RuntimeError('CPU stopped before torchrun')) as supervisor:
                    with self.assertRaisesRegex(RuntimeError, 'CPU stopped before torchrun'):
                        module.execute(root, generation_python=Path(sys.executable),
                                       scoring_python=Path(sys.executable))
                    command = supervisor.call_args.args[0]
                    self.assertIn('--nnodes=1', command)
                    self.assertIn('--nproc-per-node=8', command)
                    self.assertIs(supervisor.call_args.kwargs['smoke'], False)
                    with self.assertRaisesRegex((FileNotFoundError, RuntimeError), 'checkpoint'):
                        module.execute(root, generation_python=Path(sys.executable),
                                       scoring_python=Path(sys.executable), resume_step=1)
                    self.assertEqual(supervisor.call_count, 1)

    def test_native21_22_23_24_cli_is_opt_in_and_other_cases_fail_closed(self):
        import qwen35.native_full
        import qwen35.native_lora
        import qwen35.native_random
        import qwen35.native_training
        from qwen35.cli import main

        options = ['platour-exps', 'train', '--experiment', '21',
                   '--sender', 'Qwen3.5-4B', '--receiver', 'Qwen3.5-4B',
                   '--output', 'new-run', '--train-backend', 'historical-native',
                   '--smoke', '--vllm-python', 'native-python',
                   '--scoring-python', 'scoring-python']
        with patch('sys.argv', options), \
             patch('platour_exps.require_vllm_python', return_value='native-python'), \
             patch('qwen35.cli.inputs', return_value={'train': [], 'validation': []}), \
             patch('qwen35.native_training.prepare') as prepare, \
             patch('qwen35.native_training.execute') as execute:
            main()
            prepare.assert_called_once()
            self.assertEqual(prepare.call_args.kwargs, {'smoke': True})
            execute.assert_called_once()
        random_options = ['platour-exps', 'train', '--experiment', '22',
                          '--receiver', 'Qwen3.5-4B', '--output', 'new-run',
                          '--train-backend', 'historical-native', '--smoke',
                          '--statistics-root', 'train-only-statistics',
                          '--vllm-python', 'native-python',
                          '--scoring-python', 'scoring-python']
        with patch('sys.argv', random_options), \
             patch('platour_exps.require_vllm_python', return_value='native-python'), \
             patch('qwen35.cli.inputs', return_value={'train': [], 'validation': []}), \
             patch('qwen35.native_random.prepare') as random_prepare, \
             patch('qwen35.native_random.execute') as random_execute:
            main()
            random_prepare.assert_called_once()
            random_execute.assert_called_once()
        full_options = ['platour-exps', 'train', '--experiment', '23',
                        '--receiver', 'Qwen3.5-4B', '--output', 'new-run',
                        '--train-backend', 'historical-native', '--smoke',
                        '--vllm-python', 'native-python',
                        '--scoring-python', 'scoring-python']
        with patch('sys.argv', full_options), \
             patch('platour_exps.require_vllm_python', return_value='native-python'), \
             patch('qwen35.cli.inputs', return_value={'train': [], 'validation': []}), \
             patch('qwen35.native_full.prepare') as full_prepare, \
             patch('qwen35.native_full.execute') as full_execute:
            main()
            full_prepare.assert_called_once()
            full_execute.assert_called_once()
        lora_options = [*full_options[:3], '24', *full_options[4:]]
        with patch('sys.argv', lora_options), \
             patch('platour_exps.require_vllm_python', return_value='native-python'), \
             patch('qwen35.cli.inputs', return_value={'train': [], 'validation': []}), \
             patch('qwen35.native_lora.prepare') as lora_prepare, \
             patch('qwen35.native_lora.execute') as lora_execute:
            main()
            lora_prepare.assert_called_once()
            lora_execute.assert_called_once()
        for experiment in (20, 27, 30):
            with self.subTest(experiment=experiment), \
                 patch('sys.argv', [*options[:3], str(experiment), *options[4:]]):
                with self.assertRaisesRegex(ValueError, 'other IDs are not bundled'):
                    main()
        for specific, backend in ((full_options, 'native_full'),
                                  (lora_options, 'native_lora')):
            with self.subTest(experiment=specific[3]), \
                 patch('sys.argv', [value for value in specific if value != '--smoke']), \
                 patch('platour_exps.require_vllm_python', return_value='native-python'), \
                 patch('qwen35.cli.inputs', return_value={'train': [], 'validation': []}), \
                 patch(f'qwen35.{backend}.prepare') as new_formal, \
                 patch(f'qwen35.{backend}.execute') as execute_formal:
                main()
                self.assertEqual(new_formal.call_args.kwargs, {'smoke': False})
                execute_formal.assert_called_once()

    def test_frozen_native_grpo_policy_matches_portable_gradient(self):
        from qwen35.native_grpo_core import group_advantages, policy_loss

        rows = [dict(index=4, sample_index=i, gold_missing=False, acc=int(i < 3),
                     scoring_timed_out=False) for i in range(8)]
        advantages, counts = group_advantages(rows)
        self.assertEqual((counts['groups'], counts['mixed_groups']), (1, 1))
        rewards = torch.tensor([1.] * 3 + [0.] * 5)
        expected = (rewards - rewards.mean()) / (rewards.std(correction=1) + 1e-6)
        torch.testing.assert_close(torch.tensor([advantages[4, i] for i in range(8)]), expected)
        for reward in (float(expected[0]), float(expected[-1])):
            native = torch.tensor([-.4, -1.2, -.7], requires_grad=True)
            old_hf = torch.tensor([-.5, -.9, -1.1])
            actual, stats = policy_loss(native, reward, old_hf)
            actual.backward()
            original_gradient = native.grad.detach().clone()
            reference = native.detach().requires_grad_(True)
            ratio = (reference - reference.detach()).clamp(-20., 20.).exp()
            raw = -reward * ratio
            clipped = -reward * ratio.clamp(.8, 1.2)
            bound = torch.maximum(raw, clipped)
            if reward < 0:
                bound = torch.minimum(bound, reference.new_tensor(-reward * 3.))
            importance = (reference.detach() - old_hf).clamp(-20., 20.).exp().clamp_max(2.)
            expected_loss = (bound * importance.detach()).mean()
            expected_loss.backward()
            torch.testing.assert_close(actual.detach(), expected_loss.detach())
            torch.testing.assert_close(original_gradient, reference.grad)
            self.assertTrue(torch.isfinite(stats['rollout_is_mean']))

    def test_signed_step_before_score_and_rolling_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actor = TinyActor()
            optimizer = torch.optim.AdamW(actor.trainable(), lr=.1)
            init = save_step(root, 0, actor, optimizer, None,
                             protected={0, 2}, identity='same-original-data')
            self.assertTrue(valid(init, expected_step=0))
            with torch.no_grad():
                actor.projection.weight.add_(1.)
            step1 = save_step(root, 1, actor, optimizer, {'step': 1, 'train/accuracy': .5},
                              protected={0, 2}, identity='same-original-data')
            self.assertTrue(valid(step1))
            half = root / 'checkpoints/step-0002'
            half.mkdir()
            (half / 'model.safetensors').write_bytes(b'not-a-checkpoint')
            self.assertEqual(latest(root), step1)
            with torch.no_grad():
                actor.projection.weight.add_(1.)
            checkpoint = save_step(root, 2, actor, optimizer,
                                   {'step': 2, 'train/accuracy': .75},
                                   protected={0, 2}, identity='same-original-data')
            self.assertTrue(step1.exists())
            self.assertTrue(init.exists())
            self.assertTrue(any((root / 'recovery').glob('unsigned-step-0002-*')))
            self.assertTrue(valid(checkpoint, expected_step=2))
            self.assertEqual(latest(root), checkpoint)
            self.assertEqual(set(__import__('safetensors.torch', fromlist=['load_file'])
                                 .load_file(checkpoint / 'model.safetensors')),
                             {'P.0.weight'})
            self.assertFalse((root / 'evaluation/step-0002.json').exists())
            with self.assertRaises(RuntimeError):
                load_step(checkpoint, actor, optimizer, expected_identity='different-data')
            with torch.no_grad():
                actor.projection.weight.zero_()
            restored, record = load_step(checkpoint, actor, optimizer,
                                         expected_identity='same-original-data')
            self.assertEqual(restored, 2)
            self.assertEqual(record['step'], 2)
            self.assertTrue(torch.equal(actor.projection.weight,
                                        __import__('safetensors.torch', fromlist=['load_file'])
                                        .load_file(checkpoint / 'model.safetensors')['P.0.weight']))
            (root / 'metrics.jsonl').write_text(json.dumps({'step': 1}) + '\n' +
                                                 json.dumps({'step': 2}) + '\n' +
                                                 json.dumps({'step': 3}) + '\n')
            history = reconcile_history(root, 2, record)
            self.assertEqual([r['step'] for r in history], [1, 2])
            self.assertEqual(len(list((root / 'recovery').glob('metrics-before-resume-*'))), 1)
            with self.assertRaises(FileExistsError):
                save_step(root, 2, actor, optimizer, record, identity='same-original-data')

    def test_two_latest_nonvalidation_steps_are_retained_until_successor_is_signed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actor = TinyActor()
            optimizer = torch.optim.AdamW(actor.trainable(), lr=.1)
            save_step(root, 0, actor, optimizer, None, protected={0}, identity='same-run')
            first = save_step(root, 1, actor, optimizer, {'step': 1},
                              protected={0}, identity='same-run')
            second = save_step(root, 2, actor, optimizer, {'step': 2},
                               protected={0}, identity='same-run')
            self.assertTrue(valid(first) and valid(second))
            self.assertEqual(latest(root), second)
            third = save_step(root, 3, actor, optimizer, {'step': 3},
                              protected={0}, identity='same-run')
            self.assertFalse(first.exists())
            self.assertTrue(valid(second) and valid(third))
            self.assertTrue(valid(root / 'checkpoints/step-0000'))

    def test_partial_last_history_line_is_archived_not_claimed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actor = TinyActor()
            optimizer = torch.optim.AdamW(actor.trainable(), lr=.1)
            save_step(root, 0, actor, optimizer, None, protected={0}, identity='source')
            signed = {'step': 1, 'train/accuracy': .7}
            save_step(root, 1, actor, optimizer, signed, protected={0}, identity='source')
            metrics = root / 'metrics.jsonl'
            metrics.write_text(json.dumps(signed) + '\n{"step":2,"train/accuracy":')
            restored = reconcile_history(root, 1, signed)
            self.assertEqual(restored, [signed])
            self.assertEqual(metrics.read_text(), json.dumps(signed, sort_keys=True) + '\n')
            self.assertEqual(len(list((root / 'recovery').glob('metrics-before-resume-*'))), 1)
            metrics.write_text('{"step":2,"broken":\n' + json.dumps(signed) + '\n')
            with self.assertRaisesRegex(RuntimeError, 'committed history'):
                reconcile_history(root, 1, signed)

    def test_corrupt_payload_not_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actor = TinyActor()
            optimizer = torch.optim.AdamW(actor.trainable(), lr=.1)
            saved = save_step(root, 0, actor, optimizer, None)
            atomic_json(saved / 'checkpoint.json', {'step': 0, 'identity': 'forged'})
            self.assertFalse(valid(saved))
            with self.assertRaises(RuntimeError):
                load_step(saved, actor, optimizer)

    def test_validation_failure_recovers_at_same_signed_optimizer_step(self):
        from qwen35.cli import run_train
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / 'input/manifest.json'
            manifest.parent.mkdir()
            manifest.write_text('{"complete":true}')
            case = SimpleNamespace(identifier=19, tag='cpu-resume', task='dapo', steps=2,
                                   checkpoint_steps=(0, 1, 2), validation_steps=(1, 2),
                                   val_samples=2, train_questions=65, learning_rate=.01, method='bridge')
            arguments = portable_args(output=root / 'formal', data_root=manifest.parent,
                                        code_cases=None, official_checker=None,
                                        docker_image=None, no_final_test=True)
            samples = {'train': [dict(index=i, prompt_token_ids=[1, 2], gold='1')
                                 for i in range(65)],
                       'validation': [dict(index=0, prompt_token_ids=[1, 2], gold='1')]}
            trained, called, error_once = [], [], [True]

            def update(actor, optimizer, tokenizer, selected, rows, step, **kwargs):
                trained.append(step)
                optimizer.zero_grad(set_to_none=True)
                actor.projection.weight.sum().backward()
                optimizer.step()
                return {'train/grad_norm': 1.0}

            def score(decoder, selected, rows, **kwargs):
                step = latest(arguments.output)
                checkpoint_step = int(step.name[5:])
                self.assertTrue(valid(step, expected_step=checkpoint_step))
                called.append(checkpoint_step)
                if checkpoint_step == 1 and error_once[0]:
                    error_once[0] = False
                    raise RuntimeError('Original judge deadline')
                return {'complete': True, 'correct': 2, 'responses': 2, 'accuracy': 1.,
                        'clipping_rate': 0.}

            with patch('qwen35.cli.inputs', return_value=samples), \
                 patch('qwen35.cli._model', side_effect=lambda *_a, **_kw: (TinyActor(), None)), \
                 patch('qwen35.cli.train_step', side_effect=update), \
                 patch('qwen35.vllm_decode.decoder_session'), patch('qwen35.cli.evaluate_vllm', side_effect=score):
                with self.assertRaisesRegex(RuntimeError, 'judge deadline'):
                    run_train(case, arguments)
                self.assertEqual(trained, [1])
                self.assertEqual(int(latest(arguments.output).name[5:]), 1)
                self.assertFalse((arguments.output / 'complete.json').exists())
                run_train(case, arguments)
            self.assertEqual(trained, [1, 2])
            self.assertEqual(called, [0, 1, 1, 2])
            with (arguments.output / 'metrics.jsonl').open() as stream:
                self.assertEqual([json.loads(line)['step'] for line in stream], [1, 2])
            self.assertTrue(read(arguments.output / 'complete.json')['complete'])

    def test_deepcoder_26_29_resume_after_vllm_validation_failure(self):
        from qwen35.cli import run_train
        for identifier, method in ((26, 'bridge'), (29, 'full')):
            with self.subTest(experiment=identifier), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data_root = root / 'input'
                data_root.mkdir()
                (data_root / 'manifest.json').write_text('{"signed":true}')
                case = SimpleNamespace(identifier=identifier, tag=f'code-{identifier}',
                    task='deepcoder', method=method, steps=2, checkpoint_steps=(0, 1, 2),
                    validation_steps=(1, 2), val_samples=16, train_questions=65,
                    learning_rate=.01)
                args = SimpleNamespace(output=root / 'interrupted', data_root=data_root,
                    code_cases=root / 'cases', official_checker=root / 'checker',
                    docker_image='lcb-checker:local', backend='vllm', receiver='frozen-4b',
                    vllm_python=None, vllm_memory_fraction=None, no_final_test=True)
                signed = {'train': [dict(index=i, prompt_token_ids=[1, 2], gold='',
                              judge={'kind': 'deepcoder', 'key': f'case_{i}'}) for i in range(65)],
                          'validation': [dict(index=0, prompt_token_ids=[1, 2], gold='',
                                         judge={'kind': 'deepcoder', 'key': 'case_0'})]}
                updates, validation_steps, fail_once = [], [], [True]

                def train_step_toy(actor, optimizer, tokenizer, selected, rows, step, **kwargs):
                    updates.append((args.output.name, step))
                    optimizer.zero_grad(set_to_none=True)
                    actor.projection.weight.sum().backward()
                    optimizer.step()
                    return {'train/grad_norm': 1.0}

                def validate_toy(*_a, **_kw):
                    step = int(latest(args.output).name[5:])
                    self.assertTrue(valid(latest(args.output), expected_step=step))
                    validation_steps.append((args.output.name, step))
                    if args.output.name == 'interrupted' and step == 1 and fail_once[0]:
                        fail_once[0] = False
                        raise RuntimeError('Original isolated judge interrupted')
                    return {'complete': True, 'questions': 1, 'responses': 16, 'correct': 8,
                            'accuracy': .5, 'clipping_rate': 0.}

                class TinyDecoder:
                    def __init__(self, *_a, **_kw):
                        pass

                    def __enter__(self):
                        return self

                    def __exit__(self, *_a):
                        pass

                with patch('qwen35.cli.verify_code_training_environment',
                           return_value='a' * 64) as preflight, \
                     patch('qwen35.cli.inputs', return_value=signed), \
                     patch('qwen35.cli._model', side_effect=lambda *_a, **_kw: (TinyActor(), None)), \
                     patch('qwen35.cli.train_step', side_effect=train_step_toy), \
                     patch('qwen35.cli.evaluate_vllm', side_effect=validate_toy), \
                     patch('qwen35.vllm_decode.decoder_session', TinyDecoder):
                    with self.assertRaisesRegex(RuntimeError, 'isolated judge interrupted'):
                        run_train(case, args)
                    self.assertEqual(int(latest(args.output).name[5:]), 1)
                    self.assertFalse((args.output / 'complete.json').exists())
                    preflight.return_value = 'b' * 64
                    with self.assertRaisesRegex(RuntimeError, 'identity differs'):
                        run_train(case, args)
                    self.assertEqual(int(latest(args.output).name[5:]), 1)
                    preflight.return_value = 'a' * 64
                    run_train(case, args)
                    restarted = args.output
                    args.output = root / 'uninterrupted'
                    run_train(case, args)
                    self.assertEqual(preflight.call_count, 4)
                self.assertEqual([x[1] for x in updates if x[0] == 'interrupted'], [1, 2])
                self.assertEqual([x[1] for x in validation_steps if x[0] == 'interrupted'],
                                 [0, 1, 1, 2])
                self.assertEqual(sha256(latest(restarted) / 'model.safetensors'),
                                 sha256(latest(args.output) / 'model.safetensors'))
                self.assertEqual([json.loads(x)['step'] for x in
                                  (restarted / 'metrics.jsonl').read_text().splitlines()], [1, 2])

    def test_code_26_29_smoke_trains_and_validates_only_signed_subsets(self):
        from qwen35.cli import run_train
        for identifier in (26, 29):
            with self.subTest(experiment=identifier), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / 'inputs'
                source.mkdir()
                (source / 'manifest.json').write_text('{"test":"signed"}')
                case = SimpleNamespace(identifier=identifier, tag=f'code-{identifier}',
                    task='deepcoder', method='bridge' if identifier == 26 else 'full',
                    steps=295, checkpoint_steps=(0, 42, 295),
                    validation_steps=(42, 295), val_samples=16,
                    train_questions=65, learning_rate=.01)
                args = SimpleNamespace(output=root / 'private-smoke', data_root=source,
                    code_cases=root / 'cases', official_checker=root / 'checker',
                    docker_image='lcb-checker:local', receiver='frozen-4b',
                    vllm_python=None, vllm_memory_fraction=None,
                    backend='vllm', smoke=True)
                signed = {'train': [{'index': i, 'prompt_token_ids': [1, 2], 'gold': ''}
                                    for i in range(65)],
                          'validation': [{'index': i, 'prompt_token_ids': [1, 3], 'gold': ''}
                                         for i in range(4)]}
                calls = []

                class TinyDecoder:
                    def __init__(self, *_args, **_kwargs):
                        pass

                    def __enter__(self):
                        return self

                    def __exit__(self, *_args):
                        pass

                def update(actor, optimizer, *_args, **_kwargs):
                    optimizer.zero_grad(set_to_none=True)
                    actor.projection.weight.sum().backward()
                    optimizer.step()
                    calls.append(('train', len(_args[2]), _kwargs['cap']))
                    return {'train/responses': len(_args[2]) * 8,
                            'train/grad_norm': 1.0}

                def validate(_decoder, _case, rows, **kwargs):
                    calls.append(('val', len(rows), kwargs['count'], kwargs['cap']))
                    return {'complete': True, 'questions': len(rows),
                            'responses': len(rows) * kwargs['count'], 'correct': 0}

                with patch('qwen35.cli.verify_code_training_environment', return_value='a' * 64), \
                     patch('qwen35.cli.inputs', return_value=signed), \
                     patch('qwen35.cli._model', side_effect=lambda *_a, **_kw: (TinyActor(), None)), \
                     patch('qwen35.cli.train_step', side_effect=update), \
                     patch('qwen35.cli.evaluate_vllm', side_effect=validate), \
                     patch('qwen35.vllm_decode.decoder_session', TinyDecoder):
                    run_train(case, args)
                report = json.loads((args.output / 'complete.json').read_text())
                self.assertTrue(report['smoke_only'])
                self.assertEqual((report['steps'], report['original_total_steps'],
                                  report['response_cap']), (2, 295, 512))
                self.assertEqual([x for x in calls if x[0] == 'val'], [('val', 2, 2, 512)] * 3)
                self.assertEqual([x for x in calls if x[0] == 'train'],
                                 [('train', 2, 512), ('train', 1, 512)])
                self.assertEqual([p.name for p in sorted((args.output / 'checkpoints').glob('step-*'))],
                                 ['step-0000', 'step-0001', 'step-0002'])

    def test_code_26_29_smoke_vllm_three_signed_test_suites(self):
        from qwen35.cli import DATA, run_eval

        class Decoder:
            runtime = {'backend': 'vllm', 'version': '0.17.1'}

            def __init__(self, *_a, **_kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                pass

        for identifier in (26, 29):
            with self.subTest(experiment=identifier), tempfile.TemporaryDirectory() as temporary:
                args = SimpleNamespace(data_root=DATA, output=Path(temporary) / 'score',
                    code_cases=Path(temporary) / 'cases',
                    official_checker=Path(temporary) / 'checker',
                    docker_image='lcb-checker:local', weights=Path('model-only'),
                    receiver=Path(temporary) / 'public-4b', vllm_python=None,
                    vllm_memory_fraction=None, backend='vllm', smoke=True)
                args.receiver.mkdir()
                calls = []

                def score(_decoder, _case, rows, **kwargs):
                    calls.append((len(rows), kwargs['count'], kwargs['cap']))
                    return {'complete': True, 'questions': len(rows),
                            'responses': len(rows) * kwargs['count'],
                            'correct': 0, 'accuracy': 0., 'backend': 'vllm'}

                with patch('qwen35.cli.verify_code_evaluation_environment'), \
                     patch('qwen35.cli._model', return_value=(
                         SimpleNamespace(load_public=lambda *_: None), None)), \
                     patch('transformers.AutoTokenizer.from_pretrained', return_value=None), \
                     patch('qwen35.vllm_decode.decoder_session', Decoder), \
                     patch('qwen35.cli.evaluate_vllm', side_effect=score):
                    run_eval(select(identifier), args)
                self.assertEqual(calls, [(2, 2, 512)] * 3)
                summary = read(args.output / 'summary.json')
                self.assertTrue(summary['smoke_only'])
                self.assertEqual(summary['response_cap'], 512)
                self.assertEqual(set(summary['results']), {'lcb', 'humaneval', 'mbpp'})
                self.assertTrue(all(row['responses'] == 4
                                    for row in summary['results'].values()))

    def test_independent_qwen35_grpo_resumes_signed_step_before_validation(self):
        from platour_api import Qwen35Session
        from qwen35.checkpoints import latest, sha256
        import qwen35.cli as cli

        def new_session():
            torch.manual_seed(731)
            actor = TinyActor()
            actor.experiment = SimpleNamespace(task='dapo', method='bridge')
            return Qwen35Session(actor, SimpleNamespace(), Path('unused-receiver'),
                                 identity='tiny-original-backbones')

        training = [{'index': 0, 'prompt_token_ids': [1, 2], 'gold': '1'},
                    {'index': 1, 'prompt_token_ids': [1, 3], 'gold': '1'}]
        validation = [{'index': 0, 'prompt_token_ids': [1, 4], 'gold': '1'}]
        updates, failed = [], [False]

        def update(actor, optimizer, *_args, **_kwargs):
            updates.append(_args[3])
            optimizer.zero_grad(set_to_none=True)
            actor.projection.weight.sum().backward()
            optimizer.step()
            return {'train/grad_norm': 1.}

        def validate(system, rows, **kwargs):
            signed = latest(Path(system._current).parents[2])
            self.assertTrue(valid(signed))
            if int(signed.name[5:]) == 2 and not failed[0] and 'interrupted' in str(signed):
                failed[0] = True
                raise RuntimeError('original held-out decoder interrupted')
            return {'complete': True, 'correct': 1, 'questions': len(rows),
                    'responses': 1, 'accuracy': 1., 'backend': kwargs['backend']}

        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(cli, 'train_step', side_effect=update), \
             patch.object(Qwen35Session, 'evaluate', autospec=True, side_effect=validate):
            root = Path(temporary)
            interrupted = root / 'interrupted'
            with self.assertRaisesRegex(RuntimeError, 'decoder interrupted'):
                new_session().fit(training, validation, output=interrupted,
                                  batch_size=1, max_steps=2, smoke=True, validation_backend='vllm', validation_samples=1)
            self.assertEqual(int(latest(interrupted).name[5:]), 2)
            self.assertFalse((interrupted / 'complete.json').exists())
            new_session().fit(training, validation, output=interrupted,
                              batch_size=1, max_steps=2, smoke=True, validation_backend='vllm', validation_samples=1)
            clean = root / 'clean'
            new_session().fit(training, validation, output=clean,
                              batch_size=1, max_steps=2, smoke=True, validation_backend='vllm', validation_samples=1)
            self.assertEqual(sha256(latest(interrupted) / 'model.safetensors'),
                             sha256(latest(clean) / 'model.safetensors'))
            self.assertEqual([json.loads(line)['step'] for line in
                              (interrupted / 'metrics.jsonl').read_text().splitlines()], [1, 2])

            self.assertEqual(updates, [1, 2, 1, 2])  # Resume retries final VAL without another update.

    def test_independent_qwen35_loads_four_methods_and_uses_single_request_seed(self):
        from platour_api import Qwen35Session
        from qwen35.catalog import MAP_35B, MAP_4B
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            file = root / 'weights.safetensors'
            file.write_bytes(b'cpu-test-weights')
            captured = []

            def load_text(path, **_kwargs):
                name = str(path)
                return SimpleNamespace(config=SimpleNamespace(
                    num_hidden_layers=40 if name.endswith('sender') else 32,
                    hidden_size=2048 if name.endswith('sender') else 2560))

            class ToyActor:
                def __init__(self, case, student, teacher=None, *, template=None):
                    captured.append((case, teacher, template))
                    self.experiment = case
                    self.loaded = None

                def load_inference_state(self, saved):
                    self.loaded = saved

                def generate(self, prompt, **_kwargs):
                    return [3]

            tokenizer = SimpleNamespace(
                apply_chat_template=lambda *_a, **_kw: {'input_ids': [1, 2, 3]},
                decode=lambda ids, **_kw: 'toy-answer')

            class Decoder:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    pass

                def generate(self, rows, *, seed_override, **_kwargs):
                    self.seed = seed_override
                    return [{'response': 'toy-native', 'response_token_ids': [3]}]

            with patch('qwen35.model.load_text', side_effect=load_text), \
                 patch('qwen35.model.Actor', ToyActor), \
                 patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer), \
                 patch('safetensors.torch.load_file', return_value={
                     'virtual_hidden': torch.zeros(67, 32, 2560),
                     'template_token_ids': torch.zeros(67, dtype=torch.long)}):
                for method in ('bridge', 'random', 'full', 'lora'):
                    with self.subTest(method=method):
                        actor = Qwen35Session.load(
                            sender=root / 'sender' if method == 'bridge' else None,
                            bridge=file if method in ('bridge', 'random') else None,
                            receiver=root / 'receiver', task='deepcoder', method=method)
                        self.assertEqual(captured[-1][0].layer_map,
                                         MAP_35B if method == 'bridge' else MAP_4B)
                        with self.assertRaisesRegex(ValueError, 'requires vLLM'):
                            actor.generate(messages=[{'role': 'user', 'content': 'hello'}], backend='hf')
                        with patch.object(actor, '_decoder', return_value=Decoder()):
                            result = actor.generate(prompt_ids=[1, 2, 3], seed=37,
                                                    output=root / 'result', backend='vllm')
                            self.assertEqual(result['response'], 'toy-native')
                        self.assertEqual(actor.actor.loaded is not None,
                                         method in ('bridge', 'random'))
                with self.assertRaisesRegex(ValueError, 'TRAIN-derived'):
                    Qwen35Session.load(receiver=root / 'receiver', method='random')

    def test_independent_qwen35_code_needs_original_sandbox_before_checkpoint(self):
        from platour_api import Qwen35Session
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actor = TinyActor()
            actor.experiment = SimpleNamespace(task='deepcoder', method='bridge')
            session = Qwen35Session(actor, SimpleNamespace(), root / 'model',
                                    identity='toy-code-original')
            training = [{'index': 0, 'prompt_token_ids': [1, 2], 'gold': '',
                         'judge': {'kind': 'deepcoder', 'key': 'case_0',
                                   'sample_sha256': 'a' * 64}}]
            validation = [{'index': 1, 'prompt_token_ids': [1, 3], 'gold': '',
                           'judge': {'kind': 'deepcoder', 'key': 'case_1',
                                     'sample_sha256': 'b' * 64}}]
            output = root / 'blocked-train'
            with self.assertRaisesRegex(RuntimeError, 'Authorized original code cases'):
                session.fit(training, validation, output=output, validation_backend='vllm')
            self.assertFalse(output.exists())

    def test_independent_code_rebuilt_image_requires_explicit_provenance(self):
        import os
        import sys
        from types import ModuleType
        from platour_api import Qwen35Session
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = root / 'cases'
            checker = root / 'lcb_runner/evaluation'
            extractor = checker.parent / 'utils/extraction_utils.py'
            for path in (cases, checker, extractor.parent):
                path.mkdir(parents=True, exist_ok=True)
            (checker / 'testing_util.py').write_text('fixed-checker')
            extractor.write_text('fixed-extractor')
            case = cases / 'case_0.json'
            case.write_text('{}')
            pinned = root / 'pinned.json'
            atomic_json(pinned, {'checker_sha256': sha256(checker / 'testing_util.py'),
                                 'extractor_sha256': sha256(extractor), 'case_sha256': {}})
            atomic_json(root / 'complete.json', {'complete': True,
                'per_question_sha256_verified': True, 'training_validation_cases': 19074,
                'lcb_v6_cases': 175, 'checker_sha256': sha256(checker / 'testing_util.py'),
                'extractor_sha256': sha256(extractor)})
            actor = TinyActor()
            actor.experiment = SimpleNamespace(task='deepcoder', method='full')
            session = Qwen35Session(actor, SimpleNamespace(), root / 'receiver',
                identity='toy-code-model', code_cases=cases, official_checker=checker,
                docker_image='rebuilt-image:local')
            rows = [{'index': 0, 'prompt_token_ids': [1, 2], 'gold': '',
                     'judge': {'kind': 'deepcoder', 'key': 'case_0',
                               'sample_sha256': sha256(case)}}]
            package, utils, official = (ModuleType('lcb_runner'),
                                        ModuleType('lcb_runner.utils'),
                                        ModuleType('lcb_runner.utils.extraction_utils'))
            package.__path__, utils.__path__ = [], []
            official.__file__ = str(extractor)
            utils.extraction_utils = official
            image = 'sha256:' + 'a' * 64
            with patch.dict(sys.modules, {'lcb_runner': package,
                                          'lcb_runner.utils': utils,
                                          'lcb_runner.utils.extraction_utils': official}), \
                 patch.dict(os.environ, {'PLATOUR_CODE_JUDGE_SERVICE': ''}), \
                 patch('qwen35.judge_cases.PINNED', pinned), \
                 patch('subprocess.run', return_value=SimpleNamespace(
                     returncode=0, stdout=image + '\n')):
                with self.assertRaisesRegex(RuntimeError, 'allow_rebuilt_image=True'):
                    session._verify_judge(rows)
                session.allow_rebuilt_image = True
                audit = session._verify_judge(rows)
                self.assertEqual(audit['image_id'], image)
                self.assertFalse(audit['original_image'])
                self.assertEqual(audit['case_proof_sha256'], sha256(root / 'complete.json'))
                with patch.object(session, '_decoder'), patch('qwen35.cli.evaluate_vllm', return_value={
                        'complete': True, 'responses': 1}) as scoring:
                    measured = session.evaluate(rows, backend='vllm', cap=2)
                    scoring.assert_called_once()
                    self.assertEqual(measured['judge_environment'], audit)
                case.write_text('{"modified":true}')
                with self.assertRaisesRegex(RuntimeError, 'case SHA differs'):
                    session._verify_judge(rows)


class PublicInputTests(unittest.TestCase):
    def test_cli_defaults_to_native_vllm_for_both_task_domains(self):
        from qwen35.cli import main
        for identifier in (19, 22, 23, 24, 25, 28, 29, 30):
            with self.subTest(experiment=identifier), \
                 patch('sys.argv', ['platour-exps', 'evaluate',
                     '--experiment', str(identifier), '--receiver', 'receiver',
                     '--output', 'fresh']), \
                 patch('platour_exps.require_vllm_python',
                       side_effect=RuntimeError('vllm preflight')) as gate:
                with self.assertRaisesRegex(RuntimeError, 'vllm preflight'):
                    main()
                gate.assert_called_once_with(None)

    def test_native_vllm_first_token_and_hybrid_cache(self):
        logits = torch.tensor([.2, 1.5, -1., .3])
        self.assertEqual(sample_first(logits, 42, 16), sample_first(logits, 42, 16))
        kv = SimpleNamespace(keys=torch.ones(1, 2, 3, 4),
                             values=torch.zeros(1, 2, 3, 4))
        recurrent = SimpleNamespace(recurrent_states=[torch.ones(1, 2, 3, 4)],
            conv_states=[torch.ones(1, 2, 6)])
        cache = SimpleNamespace(layers=[recurrent, kv] * 16)
        tensors = hybrid_tensors(cache)
        self.assertEqual(tensors['layer.1.kv'].shape, (2, 3, 2, 4))
        self.assertEqual(tensors['layer.0.conv'].shape, (3, 2))
        self.assertEqual(tensors['layer.0.recurrent'].shape, (2, 4, 3))
        self.assertEqual(len(tensors), 48)

    def test_native_vllm_full_denominator_and_clean_subset(self):
        rows = [{'index': 0, 'gold': '1', 'prompt_token_ids': [1, 2]},
                {'index': 1, 'gold': '2', 'prompt_token_ids': [1, 3]}]
        class Decoder:
            def generate(self, inputs, *, count, cap):
                return [dict(index=row['index'], sample_index=sample,
                    sampling_seed=1234 + count * row['index'] + sample,
                    response='Answer: ' + row['gold'], tokens=2)
                    for row in inputs for sample in range(count)]
        with patch('qwen35.cli.math_score', return_value=1):
            result = evaluate_vllm(Decoder(), select(19), rows, count=2, cap=4,
                                   subset={0})
        self.assertEqual((result['correct'], result['responses'],
                          result['clean80_responses'], result['backend']),
                         (4, 4, 2, 'vllm'))

    def test_default_checkpoint_uses_shared_models_directory(self):
        from qwen35.cli import run_eval
        case = select(19)
        expected = Path(__file__).resolve().parents[1] / 'models' / (case.tag + '.safetensors')
        with tempfile.TemporaryDirectory() as folder, \
             patch('platour_exps.ensure_weight', return_value=expected) as fetch, \
             patch('qwen35.cli._model', side_effect=RuntimeError('skip large model')):
            args = SimpleNamespace(weights=None, output=Path(folder) / 'new-test')
            with self.assertRaisesRegex(RuntimeError, 'skip large model'):
                run_eval(case, args)
        fetch.assert_called_once_with(19)
        self.assertEqual(args.weights, expected)

    def test_release_checkpoint_selection_is_not_published(self):
        from safetensors import safe_open
        from platour_exps import weight_record

        root = Path(__file__).resolve().parents[1] / 'models'
        for case in EXPERIMENTS:
            with self.subTest(experiment=case.identifier):
                self.assertFalse(hasattr(case, 'selected_step'))
                _, filename, size, digest = weight_record(case.identifier)
                self.assertEqual(filename, {'full': 'model.safetensors',
                    'lora': 'adapter_model.safetensors'}.get(case.method, case.tag + '.safetensors'))
                self.assertEqual(len(digest), 64)
                weight = root / filename
                if weight.is_file():
                    self.assertEqual(weight.stat().st_size, size)
                    with safe_open(weight, framework='pt', device='cpu') as source:
                        metadata = source.metadata() or {}
                        self.assertNotIn('selected_step', metadata)
                        self.assertNotIn('selected_checkpoint_step', metadata)

    def test_all_twelve_signed_data_and_training_budgets(self):
        self.assertEqual([case.identifier for case in EXPERIMENTS], list(range(19, 31)))
        roles = {role: {case.identifier for case in EXPERIMENTS if case.paper_role == role}
                 for role in ('PLatour', 'Self-Communication', 'Virtual Sender', 'Trained Receiver')}
        self.assertEqual(roles, {'PLatour': {19, 20, 25, 26},
                                 'Self-Communication': {21, 27},
                                 'Virtual Sender': {22, 28},
                                 'Trained Receiver': {23, 24, 29, 30}})
        for case in EXPERIMENTS:
            with self.subTest(experiment=case.identifier):
                rows = inputs(case)
                self.assertEqual(len(rows['train']), case.train_questions)
                self.assertEqual(len(rows['validation']), case.val_questions)
                self.assertEqual(case.steps, 264 if case.task == 'dapo' else 295)
                self.assertEqual(case.validation_steps[-1], case.steps)
                for row in rows['train'][:2]:
                    self.assertNotIn('messages', row)
                    self.assertNotIn('system', row)

    def test_train_only_template_does_not_absorb_validation(self):
        rows = [{'prompt_token_ids': [1, 2, i, 3, 4]} for i in (11, 12, 13)]
        prefix, suffix, ids = template(rows)
        self.assertEqual((prefix, suffix, ids), (2, 2, [1, 2, 3]))
        self.assertEqual(template(rows), (2, 2, [1, 2, 3]))
        # A validation row must be checked against, not mixed into, train statistics.
        self.assertEqual(template(rows + [{'prompt_token_ids': [1, 5, 30, 3, 4]}]),
                         (1, 2, [1, 3]))

    def test_clean80_same_generation(self):
        case = select(19)
        calls = []

        class ToyActor:
            def generate(self, ids, *, seed, cap, temperature, top_p, top_k,
                         presence_penalty):
                calls.append((tuple(ids), seed, presence_penalty))
                return [ids[0]]

        rows = [{'index': n, 'prompt_token_ids': [n + 1, 9], 'gold': str(n + 1)}
                for n in range(3)]
        decoder = SimpleNamespace(decode=lambda ids, skip_special_tokens: str(ids[0]))
        with patch('qwen35.cli.math_score', side_effect=lambda gold, text, **kwargs:
                   int(text == gold)):
            score = evaluate(ToyActor(), decoder, case, rows, count=2, subset={1})
        self.assertEqual((score['correct'], score['responses'], score['clean80_correct'],
                          score['clean80_responses']), (6, 6, 2, 2))
        self.assertEqual(score['backend'], 'transformers')
        self.assertEqual(len(calls), 6)
        self.assertEqual([x[1] for x in calls], [1234, 1235, 1236, 1237, 1238, 1239])
        self.assertTrue(all(x[2] == 1.5 for x in calls))


class FrozenLoRATests(unittest.TestCase):
    def test_only_496_adapter_matrices_have_gradients(self):
        base = ToyReceiver()
        case = SimpleNamespace(method='lora')
        actor = Actor(case, base)
        tensors = actor.inference_state()
        self.assertEqual(len(tensors), 496)
        self.assertEqual(len(tuple(actor.trainable())), 496)
        self.assertTrue(all(p.requires_grad for p in actor.trainable()))
        self.assertTrue(all(not p.requires_grad for n, p in base.named_parameters()
                            if '.base.' in n or 'lm_head.' in n))
        self.assertEqual(len([p for p in base.parameters() if p.requires_grad]), 496)


if __name__ == '__main__':
    unittest.main()
