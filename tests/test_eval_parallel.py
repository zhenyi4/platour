import json
import os
import unittest
from unittest.mock import patch
from qwen3.eval_parallel import gpu_groups, shard_rows, merged_metrics, child_arguments, SHARD_ENV


class ParallelEvaluationTests(unittest.TestCase):
    def test_default_topologies_and_single_gpu_limits(self):
        from argparse import ArgumentParser
        from qwen3.eval_parallel import add_arguments, default_gpus_per_replica
        parser = ArgumentParser()
        add_arguments(parser)
        defaults = parser.parse_args([])
        self.assertIsNone(defaults.n_gpu)  # Preserve legacy single-process placement.
        self.assertIsNone(defaults.gpus_per_replica)
        self.assertEqual(parser.parse_args(['--n_gpu', '1']).n_gpu, 1)
        visible = [str(i) for i in range(8)]
        for experiment in range(1, 31):
            per = default_gpus_per_replica(experiment)
            expected = 2 if experiment in (1, 2, 3, 4, 5, 6, 13, 14, 15) else 4 if experiment in (20, 25) else 1
            with self.subTest(experiment=experiment):
                self.assertEqual(per, expected)
                self.assertEqual(len(gpu_groups(8, per, visible)), 8 // expected)
                if per == 1:
                    self.assertEqual(gpu_groups(1, per, visible), [['0']])
                else:
                    with self.assertRaises(ValueError):
                        gpu_groups(1, per, visible)

    def test_gpu_masks_and_grouping(self):
        self.assertEqual(gpu_groups(4, 2, ['GPU-a', 'GPU-b', 'GPU-c', 'GPU-d']),
                         [['GPU-a', 'GPU-b'], ['GPU-c', 'GPU-d']])
        for n, per in [(0, 1), (3, 2), (9, 1), (2, 0)]:
            with self.assertRaises(ValueError):
                gpu_groups(n, per, list(range(8)))

    def test_shards_preserve_global_indices_and_seeds(self):
        rows = [dict(index=i) for i in range(11)]
        shards = []
        for rank in range(4):
            with patch.dict(os.environ, {SHARD_ENV: json.dumps(dict(rank=rank, world=4))}):
                part = shard_rows(rows)
            self.assertEqual(part, rows[rank::4])
            shards.extend((r['index'], sample, 1234 + 4*r['index'] + sample) for r in part for sample in range(4))
        self.assertEqual(sorted(shards), [(i, s, 1234+4*i+s) for i in range(11) for s in range(4)])

    def test_uneven_counts_and_clean80(self):
        parts = [dict(question_indices=[0, 2, 4], responses=6, correct=3, passed_questions=2,
                      clipped=1, clean80_correct=1, clean80_responses=2),
                 dict(question_indices=[1, 3], responses=4, correct=4, passed_questions=2,
                      clipped=0, clean80_correct=0, clean80_responses=0)]
        result = merged_metrics(parts, list(range(5)), 2)
        self.assertEqual(result['accuracy'], .7)
        self.assertEqual(result['pass_at_k'], .8)
        self.assertEqual(result['clean80_accuracy'], .5)
        self.assertEqual(result['clipping_rate'], .1)
        parts[1]['question_indices'] = [0, 3]
        with self.assertRaisesRegex(ValueError, 'coverage'):
            merged_metrics(parts, list(range(5)), 2)

    def test_incomplete_results_rejected(self):
        with self.assertRaisesRegex(ValueError, 'coverage'):
            merged_metrics([dict(question_indices=[0], responses=1, correct=1)], [0], 2)

    def test_child_cli_drops_launcher_flags_and_preserves_protocol(self):
        argv = ['evaluate', '--experiment', '18', '--n_gpu=8', '--gpus-per-replica', '2',
                '--sender', 'HF/model', '--output=old', '--eval-batch-size', '32']
        result = child_arguments(argv, {'--sender': '/tmp/model', '--output': '/tmp/run'})
        self.assertEqual(result, ['evaluate', '--experiment', '18', '--eval-batch-size', '32',
                                 '--sender', '/tmp/model', '--output', '/tmp/run'])

    def test_qwen35_empty_clean_subset(self):
        from types import SimpleNamespace
        from qwen35.cli import evaluate_vllm
        rows = [dict(index=99, gold='x')]
        decoder = SimpleNamespace(generate=lambda *a, **k: [dict(index=99, sample_index=0,
            sampling_seed=1333, tokens=2, response='x')])
        with patch('qwen35.cli.math_score', return_value=1):
            result = evaluate_vllm(decoder, SimpleNamespace(task='dapo'), rows, count=1, cap=10, subset=set())
        self.assertEqual(result['question_indices'], [99])
        self.assertEqual(result['clean80_responses'], 0)
        self.assertIsNone(result['clean80_accuracy'])

class LauncherTests(unittest.TestCase):
    def test_suite_selection_and_partial_macro(self):
        from qwen3.eval_parallel import evaluation_suites, three_suite_macro
        self.assertEqual(evaluation_suites('dapo', ['aime']), ('aime',))
        self.assertEqual(evaluation_suites('dapo'), ('aime', 'hmmt', 'imo'))
        for selected in ([], ['aime', 'aime'], ['lcb']):
            with self.assertRaises(ValueError):
                evaluation_suites('dapo', selected)
        self.assertIsNone(three_suite_macro({'aime': {'clean80_accuracy': .75, 'accuracy': .70}}))
        self.assertAlmostEqual(three_suite_macro({'aime': {'clean80_accuracy': .75},
                                                 'hmmt': {'accuracy': .60}, 'imo': {'accuracy': .45}}), .60)

    def test_sender_partitions(self):
        from collections import Counter
        from qwen3.eval_parallel import sender_device_map
        for family, depth, gpus, expected in [('qwen3', 94, 2, [47, 47]),
                                               ('qwen35', 60, 4, [12, 16, 16, 16])]:
            placement = sender_device_map(family, depth, gpus)
            sizes = Counter(device for key, device in placement.items() if '.layers.' in key)
            self.assertEqual([sizes[i] for i in range(gpus)], expected)

    def test_launcher_collects_uneven_replicas(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from qwen3.eval_parallel import launch
        seen = []
        class Worker:
            returncode = 0
            def __init__(self, cmd, env, **kwargs):
                spec = json.loads(env[SHARD_ENV]); seen.append(env['CUDA_VISIBLE_DEVICES'])
                ids = list(range(9))[spec['rank']::spec['world']]
                folder = Path(cmd[cmd.index('--output')+1]); folder.mkdir()
                (folder/'summary.json').write_text(json.dumps(dict(experiment=18,
                    checkpoint_source='published_release', selection=None,
                    question_indices=ids, correct=len(ids), responses=2*len(ids),
                    accuracy=.5, at_token_cap=0, backend='vllm', runtime={}, parity=None)))
            def poll(self): return 0
            def wait(self, **kwargs): return 0
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(n_gpu=4, gpus_per_replica=1, experiment=18,
                                   output=Path(tmp)/'out', sender='/sender', receiver='/receiver', weights='/weight')
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'3,5,6,7'}, clear=True), \
                 patch('torch.cuda.device_count', return_value=4), \
                 patch('qwen3.eval_parallel.prepare', return_value={'test':(list(range(9)),2)}), \
                 patch('qwen3.eval_parallel.subprocess.Popen', Worker), \
                 patch('sys.argv', ['platour-exps','evaluate','--experiment','18','--n-gpu','4','--output','unused']), \
                 patch('builtins.print'):
                launch(args)
            self.assertEqual(seen, ['3','5','6','7'])
            result = json.loads((args.output/'summary.json').read_text())
            self.assertTrue(result['complete'])
            self.assertEqual(result['responses'], 18)
            self.assertEqual(result['question_indices'], list(range(9)))

    def test_failed_worker_does_not_publish_summary(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from qwen3.eval_parallel import launch
        worker = SimpleNamespace(poll=lambda: 1, wait=lambda **kw: 1, returncode=1)
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(n_gpu=1, gpus_per_replica=1, experiment=18,
                output=Path(tmp)/'out', sender='/sender', receiver='/receiver', weights='/weight')
            with patch.dict(os.environ, {}, clear=True), patch('torch.cuda.device_count', return_value=1), \
                 patch('qwen3.eval_parallel.prepare', return_value={'test':([0],1)}), \
                 patch('qwen3.eval_parallel.subprocess.Popen', return_value=worker), \
                 patch('sys.argv', ['platour-exps','evaluate','--experiment','18']), patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError, 'replica failed'):
                    launch(args)
            self.assertFalse((args.output/'summary.json').exists())
            self.assertTrue((args.output/'failed.json').exists())


if __name__ == '__main__':
    unittest.main()
