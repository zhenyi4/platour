"""Default full-suite dispatch, fixed inputs, and leakage/denominator guards."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from pact_check import PACTCheck, _fingerprint, build_dummy_statistics
from qwen3 import pact_cli


def row(index, question, suffix=(9, 2)):
    return dict(index=index, prompt_token_ids=[1, question, *suffix],
                response_token_ids=[3], gold='3', text='line\u2028separator')


class FullQwen3PACTTests(unittest.TestCase):
    def test_backbone_resolution_preserves_local_and_pinned_offline_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch('huggingface_hub.snapshot_download', return_value=str(root)) as download:
                with self.assertRaisesRegex(ValueError, 'config.json'):
                    pact_cli.resolve_backbone('receiver', root)
                (root / 'config.json').write_text('{}')
                self.assertEqual(pact_cli.resolve_backbone('receiver', root), root)
                download.assert_not_called()
                self.assertEqual(pact_cli.resolve_backbone('receiver', None,
                    cache_dir=root / 'cache', local_files_only=True), root)
                self.assertEqual(download.call_args.args, ('Qwen/Qwen3-1.7B',))
                self.assertEqual(download.call_args.kwargs['revision'],
                                 '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')
                self.assertTrue(download.call_args.kwargs['local_files_only'])
                self.assertEqual(download.call_args.kwargs['cache_dir'], root / 'cache')

    def test_control_test_releases_gradients_and_cache_before_evaluation_and_on_failure(self):
        train, val, test = [row(0, 10)], [row(0, 20)], [row(0, 30)]
        for label, attribute in (('SC', 'bridge'), ('TR', 'actor')):
            for fail in (False, True):
                with self.subTest(control=label, fail=fail):
                    model = torch.nn.Linear(2, 2)
                    before = model.weight.detach().clone()
                    events = []
                    def factory(training, validation):
                        model.weight.sum().backward()
                        self.assertIsNotNone(model.weight.grad)
                        return SimpleNamespace(**{attribute: model}), dict(trained=True,
                            control=label, train_digest=_fingerprint(train), validation_digest=_fingerprint(val),
                            checkpoint='signed-final', uses_test_for_training=False)
                    def evaluate(system, rows, **kwargs):
                        self.assertEqual(events, ['release'])
                        self.assertIsNone(model.weight.grad)
                        self.assertFalse(model.training)
                        self.assertTrue(model.weight.requires_grad)
                        torch.testing.assert_close(model.weight, before, atol=0, rtol=0)
                        events.append('evaluate')
                        if fail:
                            raise RuntimeError('worker startup failed')
                        return dict(complete=True, responses=1, correct=1, backend='vllm')
                    pact = PACTCheck(None, train, val, test, evaluator=evaluate)
                    with patch('pact_check._release_accelerator_memory', side_effect=lambda: events.append('release')):
                        options = dict(controls=[label], samples_per_question=1,
                                       **{'train_' + label.lower(): factory})
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, 'worker startup'):
                                pact.run(**options)
                        else:
                            pact.run(**options)
                    self.assertEqual(events, ['release', 'evaluate', 'release'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.test = [row(0, 80), row(1, 80)]
        self.donors = {0: [1, 60, 9, 2], 1: [1, 60, 9, 2]}
        self.config = dict(task='gsm8k', mapping='uniform_endpoints', samples=2, cap=17,
                           sampling=dict(temperature=.7, top_p=.8, top_k=20))
        self.audit = dict(profile='archived-it-v1')

    def prepare(self, experiment, *, malformed=False, overlap=False):
        train = [row(i, 10+i, (8, 2)) for i in range(7)]
        if malformed:
            train[0]['prompt_token_ids'][-2] = 777
        validation = [row(0, 80 if overlap else 50, (8, 2))]
        files = {}
        for part, rows in [('train', train), ('validation', validation)]:
            name = f'qwen3/{experiment:02d}_fixture/{part}.jsonl'
            path = self.root/(part+'.jsonl')
            path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
            files[name] = path
        table = dict(files={k: dict(sha256='a'*64) for k in files},
                     repositories=dict(qwen3=dict(repo_id='fixture/data', revision='a'*40)))
        templates = {'full_pact_templates': {str(experiment): dict(prefix_ids=[1],
            suffix_ids=[9, 2], source_suffix_ids=[8, 2], protocol_sha256='b'*64)}}
        with patch.object(pact_cli, 'load_inputs', return_value=(self.test, self.donors, self.config, self.audit)), \
                patch.object(pact_cli, 'release', return_value=templates), \
                patch('platour_data.registry', return_value=table), \
                patch('platour_data.experiment_files', return_value=list(files)), \
                patch('platour_data.ensure_file', side_effect=lambda name, *a, **kw: files[name]) as fetch:
            result = pact_cli.prepare_inputs(experiment, local_files_only=True)
        return result, fetch

    def test_grpo_uses_published_train_validation_without_resplitting(self):
        result, fetch = self.prepare(16)
        train, validation, test, donors, config, audit, prefix, suffix = result
        self.assertEqual([len(train), len(validation), len(test)], [7, 1, 2])
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual([prefix, suffix], [[1], [9]])
        self.assertTrue(all(r['prompt_token_ids'][-2:] == [9, 2] for r in train+validation))
        self.assertEqual(test, self.test)
        self.assertEqual(donors, self.donors)
        self.assertEqual(audit['full_pact_data']['validation_policy'], 'published_train_validation_splits')
        self.assertTrue(all('original_train_index' not in r for r in train+validation))
        with self.assertRaisesRegex(ValueError, 'evaluation prompts'):
            self.prepare(16, overlap=True)

    def test_sft_keeps_published_validation_and_rejects_template_or_leakage(self):
        result, fetch = self.prepare(7)
        self.assertEqual([len(r) for r in result[:3]], [7, 1, 2])
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(result[5]['full_pact_data']['validation_policy'], 'published_train_validation_splits')
        with self.assertRaisesRegex(ValueError, 'source template'):
            self.prepare(7, malformed=True)
        with self.assertRaisesRegex(ValueError, 'evaluation prompts'):
            self.prepare(7, overlap=True)

    def test_duplicate_opt_in_keeps_disjointness_and_unique_index_checks(self):
        train, val = [row(0, 10)], [row(0, 20)]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            PACTCheck(None, train, val, self.test, evaluator=None)
        PACTCheck(None, train, val, self.test, evaluator=None, allow_duplicate_prompts=True)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            PACTCheck(None, train, val, [self.test[0]]*2, evaluator=None, allow_duplicate_prompts=True)
        with self.assertRaisesRegex(ValueError, 'evaluation prompts'):
            PACTCheck(None, train, train, self.test, evaluator=None, allow_duplicate_prompts=True)

    def test_dt_samples_distinct_prompts_but_signs_full_training_split(self):
        training = [row(0, 10), row(1, 10), row(2, 11)]
        seen = []
        def features(prompt):
            seen.append(tuple(prompt))
            return (torch.ones(1, len(prompt)-1, 2),)
        bridge = SimpleNamespace(_features=features)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            build_dummy_statistics(bridge, training, prefix_ids=[1], suffix_ids=[9], sample_count=2)
        dummy = build_dummy_statistics(bridge, training, prefix_ids=[1], suffix_ids=[9],
                                       sample_count=2, allow_duplicate_prompts=True)
        self.assertEqual(len(set(seen)), 2)
        self.assertEqual(dummy.training_digest, _fingerprint(training))
        with self.assertRaisesRegex(ValueError, 'distinct'):
            build_dummy_statistics(bridge, training, prefix_ids=[1], suffix_ids=[9],
                                   sample_count=3, allow_duplicate_prompts=True)

    def _run_entry(self, controls=None, smoke=False):
        import platour_exps
        train, validation = [row(0, 10), row(1, 11)], [row(0, 20)]
        prepared = (train, validation, self.test, self.donors, self.config, self.audit, [1], [9])
        calls, trained, sender_calls = {}, [], []
        def sender_model(**kwargs):
            sender_calls.append(kwargs['input_ids'].tolist()[0])
            return SimpleNamespace(hidden_states=(torch.ones(1, 3, 2),))
        bridge = SimpleNamespace(receiver=object(), indices=[0],
            sender=SimpleNamespace(model=sender_model, device='cpu'),
            _features=lambda prompt: (torch.ones(1, len(prompt)-1, 2),))
        def evaluate(rows, **kwargs):
            label = kwargs['output'].name
            calls[label] = (rows, kwargs)
            if kwargs['feature_provider'] is not None:
                for r in rows:
                    kwargs['feature_provider'](r['prompt_token_ids'])
            return dict(complete=True, questions=len(rows), responses=len(rows)*kwargs['samples'],
                        correct=0, accuracy=0., backend=kwargs['backend'])
        def fit(training, heldout, **kwargs):
            trained.append((training, heldout, kwargs))
            receipt = dict(trained=True, control=kwargs['control'], train_digest=_fingerprint(training),
                validation_digest=_fingerprint(heldout), uses_test_for_training=False,
                checkpoint=str(kwargs['output']/'complete.json'))
            if smoke:
                receipt.update(training_only=True, selection_performed=False, training_steps=10)
            return receipt
        base = SimpleNamespace(bridge=bridge, tokenizer=object(), evaluate=evaluate)
        weight = self.root/'bridge'; weight.write_bytes(b'fixture')
        selected = controls or ['DT', 'IT', 'SC', 'TR']
        output = self.root/('run-' + '-'.join(selected))
        argv = ['platour-exps', 'pact-check', '--experiment', '16',
                '--dt-samples', '2', '--bridge', str(weight), '--output', str(output)]
        if controls is not None:
            argv += ['--controls', *controls]
        if smoke:
            argv += ['--grpo-smoke-steps', '10']
        with patch('sys.argv', argv), redirect_stdout(StringIO()), \
                patch.object(pact_cli, 'prepare_inputs', return_value=prepared), \
                patch('torch.cuda.is_available', return_value=True), \
                patch('pact_check._release_accelerator_memory'), \
                patch('platour_exps.require_vllm_python'), patch('platour_exps.verify_weight'), \
                patch('qwen3.pact_cli.resolve_backbone', side_effect=lambda role, *a, **kw: self.root/role), \
                patch('platour_api.PLatourSession.load', return_value=base) as load_base, \
                patch('platour_api.PLatourSession.self_communication',
                      return_value=SimpleNamespace(evaluate=evaluate, fit=fit)), \
                patch('platour_api.PLatourSession.load_self_communication',
                      return_value=SimpleNamespace(evaluate=evaluate, fit=fit)) as load_self, \
                patch('platour_api.ReceiverSession.load',
                      return_value=SimpleNamespace(evaluate=evaluate, fit=fit)):
            platour_exps.main()
        self.assertEqual(load_base.call_count, int(bool(set(selected) & {'DT', 'IT'})))
        self.assertEqual(load_self.call_count, int('SC' in selected and not set(selected) & {'DT', 'IT'}))
        report = json.loads((output/'pact_check.json').read_text())
        self.assertEqual(set(calls), set() if smoke else set(selected))
        self.assertEqual(report['responses_per_setting'], 0 if smoke else 4)
        self.assertEqual(report['test_performed'], not smoke)
        self.assertEqual(report['control_training_steps'], {k: 10 if smoke else 3 for k in ('SC', 'TR') if k in selected})
        self.assertEqual(report['control_validation_samples'], {k: v for k, v in {'SC': 4, 'TR': 4}.items() if k in selected})
        self.assertEqual(report['it_data_audit']['profile'], 'archived-it-v1')
        self.assertEqual(sender_calls, [[1, 60, 9]]*2 if 'IT' in selected else [])
        for rows, kwargs in calls.values():
            self.assertEqual(rows, self.test)
            self.assertEqual(kwargs['sampling'], self.config['sampling'])
            self.assertEqual((kwargs['backend'], kwargs['samples'], kwargs['cap']), ('vllm', 2, 17))
        self.assertEqual(len(trained), len(set(selected) & {'SC', 'TR'}))
        for training, heldout, kwargs in trained:
            self.assertEqual((training, heldout), (train, validation))
            self.assertEqual(kwargs['objective'], 'grpo')
            self.assertEqual(kwargs['epochs'], 3)
            self.assertEqual(kwargs['batch_size'], 64)
            if smoke:
                self.assertEqual(kwargs['max_steps'], 10)
                self.assertTrue(kwargs['smoke'])
                self.assertEqual(kwargs['validation_schedule'], 'initial')
            else:
                self.assertNotIn('max_steps', kwargs)
                self.assertEqual(kwargs['validation_schedule'], 'half_epoch')
            self.assertEqual(kwargs['validation_samples'], 4)
        self.assertEqual(json.loads((output/'pact_protocol.json').read_text())['test_sha256'],
                         _fingerprint(self.test))

    def test_default_entry_runs_four_controls_with_fixed_sampling_and_effective_donors(self):
        self._run_entry()

    def test_selected_cli_controls_use_correct_output_labels(self):
        for controls in (['IT'], ['DT'], ['TR'], ['DT', 'SC']):
            with self.subTest(controls=controls):
                self._run_entry(controls)

    def test_grpo_smoke_runs_selected_training_without_any_test(self):
        self._run_entry(['SC', 'TR'], smoke=True)

    def test_full_execution_needs_output_and_rejects_step_override(self):
        with redirect_stdout(StringIO()), patch('sys.stderr', new_callable=StringIO), \
                patch.object(pact_cli, 'prepare_inputs') as prepare:
            with self.assertRaises(SystemExit):
                pact_cli.main(['--experiment', '16'])
            with self.assertRaises(SystemExit):
                pact_cli.main(['--experiment', '16', '--prepare-only', '--control-steps', '3'])
            for controls in (['PACT'], ['IT', 'IT']):
                with self.assertRaises(SystemExit):
                    pact_cli.main(['--experiment', '16', '--prepare-only', '--controls', *controls])
            prepare.assert_not_called()

    def test_recipe_batches_and_explicit_override_in_preparation(self):
        cases = [(7, 'arc_c', None, 64), (11, 'gsm8k', None, 64),
                 (16, 'chemistry_cot', None, 32), (17, 'gsm8k', None, 64),
                 (11, 'gsm8k', 8, 8)]
        for experiment, task, override, expected in cases:
            with self.subTest(experiment=experiment, override=override):
                train = [row(i, 10+i) for i in range(65)]
                prepared = (train, [row(0, 90)], self.test, self.donors,
                            dict(self.config, task=task), self.audit, [1], [9])
                argv = ['--experiment', str(experiment), '--prepare-only', '--dt-samples', '2']
                if override is not None:
                    argv += ['--batch-size', str(override)]
                stream = StringIO()
                with patch.object(pact_cli, 'prepare_inputs', return_value=prepared), redirect_stdout(stream):
                    pact_cli.main(argv)
                report = json.loads(stream.getvalue())
                self.assertEqual(report['batch_size'], expected)
                self.assertEqual(report['micro_batch'], {7: 8, 11: 4}.get(experiment, 1))
                self.assertEqual(report['checkpoint_every'], 0)
                self.assertEqual(report['control_training_steps'],
                                 dict.fromkeys(('SC', 'TR'), 3*((65+expected-1)//expected)))

    def test_every_control_subset_skips_unselected_work_and_baseline(self):
        from itertools import combinations
        from pact_check import CONTROLS, _fingerprint
        train, val, test = [row(0, 10)], [row(0, 20)], [row(0, 30)]
        bridge = object()
        dummy = SimpleNamespace(training_digest=_fingerprint(train), sample_count=1, features=lambda p: p)
        for count in range(1, 5):
            for controls in combinations(CONTROLS, count):
                with self.subTest(controls=controls):
                    called = []
                    def evaluate(system, rows, *, feature_provider):
                        if system is bridge:
                            self.assertIsNotNone(feature_provider, 'Baseline must not be evaluated')
                            label = 'DT' if feature_provider == dummy.features else 'IT'
                        else:
                            label = system
                        called.append(label)
                        return dict(complete=True, correct=1, responses=1, backend='vllm')
                    def factory(label):
                        def fit(training, validation):
                            self.assertIn(label, controls)
                            return label, dict(trained=True, control=label,
                                train_digest=_fingerprint(train), validation_digest=_fingerprint(val),
                                checkpoint=label, uses_test_for_training=False)
                        return fit
                    pact = PACTCheck(bridge, train, val, test, evaluator=evaluate)
                    report = pact.run(controls=controls, samples_per_question=1,
                        dummy=dummy if 'DT' in controls else None,
                        donors={0: [1, 99, 9, 2]} if 'IT' in controls else None,
                        train_sc=factory('SC'), train_tr=factory('TR'))
                    self.assertCountEqual(called, controls)
                    self.assertEqual(set(report['raw']), set(controls))
                    self.assertNotIn('pact_score_pp', report)
                    self.assertNotIn('strongest_control', report)

    def test_sft_sc_only_dispatch_uses_epoch_loss_without_dt_or_tr(self):
        train, validation = [row(0, 10)], [row(0, 20)]
        prepared = (train, validation, self.test, self.donors,
                    dict(self.config, task='arc_c'), self.audit, [1], [9])
        base = SimpleNamespace(bridge=SimpleNamespace(receiver=object()), tokenizer=object())
        trained = []
        def fit(training, heldout, **kwargs):
            trained.append(kwargs)
            return dict(trained=True, control='SC', train_digest=_fingerprint(train),
                validation_digest=_fingerprint(validation), uses_test_for_training=False,
                checkpoint='SC/checkpoint', selected_step=1)
        def evaluate(rows, **kwargs):
            self.assertEqual(kwargs['task'], 'arc_c')  # TEST still uses the real task and vLLM.
            self.assertEqual(kwargs['backend'], 'vllm')
            return dict(complete=True, responses=len(rows)*kwargs['samples'], correct=0, backend='vllm')
        weight = self.root/'bridge'; weight.write_bytes(b'fixture')
        argv = ['--experiment', '7', '--controls', 'SC', '--output', str(self.root/'sc'), '--bridge', str(weight)]
        with patch.object(pact_cli, 'prepare_inputs', return_value=prepared), redirect_stdout(StringIO()), \
             patch('torch.cuda.is_available', return_value=True), \
             patch('pact_check._release_accelerator_memory'), \
             patch('platour_exps.require_vllm_python'), patch('platour_exps.verify_weight'), \
             patch('qwen3.pact_cli.resolve_backbone', side_effect=lambda role, *a, **kw: self.root/role), \
             patch('platour_api.PLatourSession.load', side_effect=AssertionError('Original sender must not load for SC-only')), \
             patch('platour_api.PLatourSession.load_self_communication', return_value=SimpleNamespace(fit=fit, evaluate=evaluate)), \
             patch('pact_check.build_dummy_statistics', side_effect=AssertionError('DT not selected')), \
             patch('platour_api.ReceiverSession.load', side_effect=AssertionError('TR not selected')):
            pact_cli.main(argv)
        self.assertEqual(len(trained), 1)
        self.assertEqual(trained[0]['validation_schedule'], 'epoch')
        self.assertEqual(trained[0]['validation_backend'], 'hf')
        self.assertEqual(trained[0]['micro_batch'], 8)
        self.assertEqual(trained[0]['checkpoint_interval'], 0)
        self.assertEqual(trained[0]['warmup_steps'], 1)
        self.assertIsNone(trained[0]['task'])
        report = json.loads((self.root/'sc/pact_check.json').read_text())
        self.assertEqual(report['controls'], ['SC'])
        self.assertEqual(report['control_training_steps'], {'SC': 3})
        self.assertNotIn('TR', report['raw'])


if __name__ == '__main__':
    unittest.main()
