"""Historical SFT optional final TEST dispatch, terminal state and retry contract."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn

from qwen3.aligned_sft import complete_aligned_run, final_test_protocol
from qwen3.checkpoints import save_step


class HistoricalSFTTestTests(unittest.TestCase):
    def test_cli_passes_optional_test_and_checks_generation_runtime(self):
        from qwen3.cli import main, select
        config = select(7)
        rows = [dict(index=i, prompt_token_ids=[1,2], response_token_ids=[3], gold='A')
                for i in range(config['train_examples'])]
        for with_test in (False, True):
            with self.subTest(with_test=with_test), TemporaryDirectory() as folder:
                argv = ['platour-exps','train','--experiment','7','--train-backend','historical-sft',
                        '--sender','sender','--receiver','receiver','--train-data','train',
                        '--validation-data','val','--output',str(Path(folder)/'run')]
                if with_test:
                    argv += ['--test-data','test','--vllm-python','/test/vllm/bin/python']
                with patch('sys.argv',argv), patch.dict('os.environ',{'WORLD_SIZE':'4','RANK':'0','LOCAL_RANK':'0'}), \
                     patch('qwen3.cli.audit_inputs') as audit, \
                     patch('qwen3.cli.read_rows',side_effect=[rows,rows[:2],rows[:1]] if with_test else [rows,rows[:2]]), \
                     patch('platour_exps.require_vllm_python') as require, \
                     patch('qwen3.aligned_sft.train_aligned') as train:
                    main()
                self.assertEqual(train.call_args.kwargs['final_test'], rows[:1] if with_test else None)
                self.assertEqual(require.call_count, int(with_test))
                self.assertEqual('test' in audit.call_args.args[1], with_test)
                if with_test:
                    require.assert_called_once_with(Path('/test/vllm/bin/python'))

    def fixture(self, root, *, smoke=False):
        bridge = nn.Module(); bridge.P = nn.ModuleList([nn.Linear(1,1,bias=False)])
        optimizer = torch.optim.AdamW(bridge.P.parameters(), lr=.01)
        config = dict(method='sft', task='gsm8k', total_steps=30 if smoke else 3, smoke=smoke,
                      eval_backend='vllm', vllm_python='/test/vllm/bin/python',
                      eval_samples=2, response_cap=16)
        for step in (1,2,3):
            optimizer.zero_grad(); bridge.P[0].weight.sum().backward(); optimizer.step()
            save_step(root,step,bridge,optimizer,record=dict(step=step,heldout_nll=float(step)))
        # Earlier checkpoints have better loss; TEST must still use step 3.
        test=[dict(index=0,prompt_token_ids=[1,2],gold='3')]
        completion=dict(complete=True,steps=3,smoke=smoke)
        return bridge,optimizer,config,test,completion

    def test_final_checkpoint_only_no_test_no_decoder_and_smoke_label(self):
        for smoke in (False,True):
            for with_test in (False,True):
                with self.subTest(smoke=smoke,with_test=with_test), TemporaryDirectory() as folder:
                    root=Path(folder)
                    bridge,optimizer,config,test,completion=self.fixture(root,smoke=smoke)
                    before=bridge.P[0].weight.detach().clone()
                    def evaluate(model,tokenizer,rows,settings,**kwargs):
                        torch.testing.assert_close(model.P[0].weight,before,rtol=0,atol=0)
                        self.assertIsNone(model.P[0].weight.grad)
                        self.assertFalse(model.training)
                        self.assertEqual(kwargs['worker_python'],'/test/vllm/bin/python')
                        self.assertEqual(kwargs['count'],2)
                        self.assertFalse((root/'complete.json').exists())
                        return dict(correct=1,responses=2,accuracy=.5,backend='vllm')
                    with patch('qwen3.runner.evaluate_vllm',side_effect=evaluate) as decode:
                        complete_aligned_run(config,bridge,None,[],test if with_test else None,root,optimizer,completion)
                        if with_test:
                            # Simulate a crash after TEST summary but before completion.
                            (root/'complete.json').unlink()
                            complete_aligned_run(config,bridge,None,[],test,root,optimizer,completion)
                    self.assertEqual(decode.call_count,int(with_test))
                    self.assertTrue(json.loads((root/'complete.json').read_text())['complete'])
                    if with_test:
                        selected=json.loads((root/'selection.json').read_text())
                        self.assertEqual(selected['selected_step'],3)
                        self.assertFalse(selected['test_used_for_selection'])
                        summary=json.loads((root/'test/summary.json').read_text())
                        self.assertEqual(summary['smoke_only'],smoke)
                        self.assertEqual(summary['selected_step'],3)
                        self.assertTrue(bridge.training)

    def test_failed_test_restores_terminal_state_then_retries_partial_output(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            bridge,optimizer,config,test,completion=self.fixture(root)
            before=bridge.P[0].weight.detach().clone()
            saved=torch.load(root/'checkpoints/step-0003/training.pt',weights_only=True)['optimizer']
            rng=torch.load(root/'checkpoints/step-0003/rng.rank-00.pt',weights_only=True)['torch']
            def fail(*args,**kwargs):
                kwargs['output'].mkdir(parents=True)
                (kwargs['output']/'partial.json').write_text('{}')
                with torch.no_grad(): bridge.P[0].weight.add_(100.)
                torch.rand(20)
                raise RuntimeError('decoder interrupted')
            with patch('qwen3.runner.evaluate_vllm',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'decoder interrupted'):
                    complete_aligned_run(config,bridge,None,[],test,root,optimizer,completion)
            self.assertFalse((root/'complete.json').exists())
            torch.testing.assert_close(bridge.P[0].weight,before,atol=0,rtol=0)
            self.assertTrue(torch.equal(torch.get_rng_state(),rng))
            self.assertEqual(optimizer.state_dict()['param_groups'],saved['param_groups'])
            for key,value in saved['state'][0].items():
                torch.testing.assert_close(optimizer.state_dict()['state'][0][key],value,rtol=0,atol=0)
            with patch('qwen3.runner.evaluate_vllm',return_value=dict(correct=0,responses=2,accuracy=0.,backend='vllm')):
                complete_aligned_run(config,bridge,None,[],test,root,optimizer,completion)
            self.assertEqual(len(list((root/'recovery').glob('incomplete-test-vllm-*'))),1)
            self.assertTrue((root/'complete.json').exists())

    def test_incomplete_budget_corrupt_checkpoint_and_bad_coverage_never_complete(self):
        for failure in ('budget','checkpoint','coverage'):
            with self.subTest(failure=failure), TemporaryDirectory() as folder:
                root=Path(folder)
                bridge,optimizer,config,test,completion=self.fixture(root)
                if failure=='budget': completion['steps']=2
                if failure=='checkpoint': (root/'checkpoints/step-0003/bridge.pt').write_bytes(b'corrupt')
                with patch('qwen3.runner.evaluate_vllm',return_value=dict(correct=1,responses=1,accuracy=1.)):
                    with self.assertRaises((ValueError,RuntimeError)):
                        complete_aligned_run(config,bridge,None,[],test,root,optimizer,completion)
                self.assertFalse((root/'complete.json').exists())

    def test_test_data_and_decode_settings_are_bound_to_protocol(self):
        config=dict(eval_samples=2,response_cap=16)
        test=[dict(index=0,prompt_token_ids=[1,2],gold='3')]
        original=final_test_protocol(config,test)
        self.assertEqual(final_test_protocol(config,None),{})
        self.assertNotEqual(original,final_test_protocol(config,[dict(test[0],gold='4')]))
        self.assertNotEqual(original,final_test_protocol(dict(config,eval_samples=4),test))
        with self.assertRaises(ValueError): final_test_protocol(config,[])
