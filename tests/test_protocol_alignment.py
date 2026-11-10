"""Cross-entry protocol regressions: scoring, defaults and native selection."""
import inspect
import unittest
from types import SimpleNamespace

from qwen3.scoring import evaluation_score
from qwen35.native_selection import contract, verify, install
from training_protocol import validation_backend, training_learning_rate


class ProtocolAlignmentTests(unittest.TestCase):
    def test_chemistry_requires_complete_final_answer_field(self):
        for response in ('A', '<answer>A', 'reasoning A </answer>', '<answer>A</answer><answer>B</answer>'):
            self.assertEqual(evaluation_score('chemistry_cot', response, 'A'), 0)
        self.assertEqual(evaluation_score('chemistry_cot', 'Reasoning B. <answer> A </answer>', 'A'), 1)

    def test_sft_uses_nll_even_when_task_is_named(self):
        self.assertEqual(validation_backend(None, 'sft', 'gsm8k'), 'hf')
        self.assertEqual(validation_backend(None, 'grpo', 'dapo'), 'vllm')
        with self.assertRaisesRegex(ValueError, 'NLL'):
            validation_backend('vllm', 'sft', 'gsm8k')
        self.assertEqual(training_learning_rate('grpo', 'full', None), 1e-6)

    def test_qwen35_api_formal_training_defaults(self):
        from platour_api import Qwen35Session
        params = inspect.signature(Qwen35Session.fit).parameters
        self.assertEqual(params['batch_size'].default, 64)
        self.assertEqual(params['response_cap'].default, 16384)
        self.assertIsNone(params['validation_samples'].default)  # resolved by task

    def test_native_entry_ignores_clipping_keeps_earliest_tie_and_full_budget(self):
        trainer = SimpleNamespace(early_stop_update=None, validation_update=None)
        install(trainer)
        state, reason = trainer.early_stop_update(None, 0, .5, .01, dict(max_steps=264))
        state, reason = trainer.early_stop_update(state, 33, .6, .99, dict(max_steps=264))
        self.assertEqual(state['best_step'], 33)
        for step in (66, 99, 132):
            state, reason = trainer.early_stop_update(state, step, .6, 0., dict(max_steps=264))
            self.assertEqual(state['best_step'], 33)
            self.assertIsNone(reason)
        _, reason = trainer.early_stop_update(state, 264, .5, 0., dict(max_steps=264))
        self.assertEqual(reason, 'max_steps')
        state, _ = trainer.validation_update(None, 0, dict(accuracy=.5, clipping_rate=0.), 'deepcoder', 295)
        state, _ = trainer.validation_update(state, 42, dict(accuracy=.7, clipping_rate=.99), 'deepcoder', 295)
        self.assertEqual(state['best_step'], 42)
        verify(contract())
        with self.assertRaisesRegex(ValueError, 'fresh run'):
            verify({})

    def test_native_chemistry_private_scorer_matches_public_field_requirement(self):
        import ast
        import re
        from qwen3.native_grpo import ASSETS, CHEMISTRY_SCORER, strict_chemistry_source
        source = strict_chemistry_source((ASSETS / 'base' / CHEMISTRY_SCORER).read_text())
        function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'score')
        scope = {'re': re}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<native Chemistry scorer>', 'exec'), scope)
        for response, expected in [('A', 0), ('<answer>A', 0), ('<answer>A</answer>', 1)]:
            result = scope['score'](dict(response=response, mode='nonthinking_cot', labels=['A','B','C','D'], gold='A'))
            self.assertEqual(result['acc'], expected)
