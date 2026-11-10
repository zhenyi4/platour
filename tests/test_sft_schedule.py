"""SFT schedule parity with HF, defaults, and signed resume configuration."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from transformers import get_linear_schedule_with_warmup
from training_protocol import (sft_schedule, set_sft_learning_rate,
                               training_learning_rate)


class SFTScheduleTests(unittest.TestCase):
    def test_stateless_schedule_matches_hf_at_every_step_and_resume(self):
        for total, warmup in ((54, 6), (465, 46), (90, 9), (348, 35), (351, 35), (1, 1), (9, 0)):
            for peak in (1e-4, 1e-5):
                with self.subTest(total=total, peak=peak):
                    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=peak)
                    reference = get_linear_schedule_with_warmup(optimizer, warmup, total)
                    actual = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=peak)
                    protocol = dict(objective='sft', learning_rate=peak, max_steps=total,
                                    **sft_schedule('sft', total, warmup))
                    for completed in range(total+1):
                        set_sft_learning_rate(actual, protocol, completed)
                        self.assertAlmostEqual(actual.param_groups[0]['lr'], optimizer.param_groups[0]['lr'])
                        optimizer.step()
                        reference.step()
                    # A new optimizer at an arbitrary restored step has the same LR.
                    set_sft_learning_rate(actual, protocol, total // 2)
                    self.assertGreaterEqual(actual.param_groups[0]['lr'], 0.)

    def test_defaults_explicit_overrides_and_grpo_preservation(self):
        for method in ('bridge', 'self', 'random', 'lora', 'full'):
            self.assertEqual(training_learning_rate('sft', method, None), 1e-5 if method == 'full' else 1e-4)
            self.assertEqual(training_learning_rate('grpo', method, None), 1e-6 if method == 'full' else 1e-5)
            self.assertEqual(training_learning_rate('sft', method, 3e-5), 3e-5)
        self.assertEqual(sft_schedule('sft', 11)['warmup_steps'], 2)
        self.assertEqual(sft_schedule('grpo', 11), {})
        for value in (-1, float('nan'), float('inf'), 0):
            with self.assertRaises(ValueError):
                training_learning_rate('sft', 'full', value)
        for value in (-1, 12, 1.5, True):
            with self.assertRaises(ValueError):
                sft_schedule('sft', 11, value)

    def test_pact_uses_published_warmup_for_matching_budgets(self):
        from training_protocol import qwen3_sft_warmup
        for task, total, warmup in [('arc_c', 54, 6), ('mmlu_redux', 465, 46),
                                    ('chemistry_no_cot', 90, 9), ('chemistry_cot', 90, 9),
                                    ('gsm8k', 348, 35), ('math500', 351, 35)]:
            self.assertEqual(qwen3_sft_warmup(task, total), warmup)
        self.assertEqual(qwen3_sft_warmup('custom', 11), 2)

    def test_old_constant_sft_protocol_cannot_resume_as_linear(self):
        from test_vllm_training import VLLMTrainingTests
        fixture = VLLMTrainingTests()
        fixture.setUp()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            fixture.fit(fixture.fresh(), root, 'sft')
            path = root/'protocol.json'
            protocol = json.loads(path.read_text())
            for key in ('scheduler', 'warmup_steps', 'schedule_version'):
                del protocol[key]
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(RuntimeError, 'resume'):
                fixture.fit(fixture.fresh(), root, 'sft')
