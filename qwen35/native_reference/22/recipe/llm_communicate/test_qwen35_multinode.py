"""CPU regressions for two-node layout, safe launch modes and zero-initialized bridge."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from qwen35_multinode_config import topology, rank_layout, configure_protocol, initialize_zero
from launch_qwen35_bridge_multinode import parse_args, options, torchrun_command, python_paths, prepare_control, complete_checkpoint
from qwen35_multinode_bridge import MonotoneFP8Bridge
from qwen35_bridge_opd_core import early_stop_update
from qwen35_bridge_grpo_core import sampled_log_probs, policy_loss
from test_qwen35_bridge_opd import model, Tests as HybridFixture


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HybridFixture.setUpClass()

    def parse(self, *extra):
        with patch.dict(os.environ, {'NNODES':'2','NPROC_PER_NODE':''}):
            return parse_args(['0','192.0.2.10',*extra])

    def test_two_node_replica_counts(self):
        self.assertEqual(topology('122b')['world_size'],8)
        self.assertEqual(topology('122b')['local_world_size'],4)
        self.assertEqual(topology('397b')['world_size'],4)
        self.assertEqual(topology('397b')['local_world_size'],2)
        self.assertEqual(topology('397b')['total_gpus'],16)
        with self.assertRaises(ValueError):topology('397b',gpus_per_node=2)

    def test_all_global_ranks_map_to_local_gpu_groups(self):
        for teacher in ('122b','397b'):
            p=topology(teacher);p['teacher_key']=teacher
            for rank in range(p['world_size']):
                node,local=divmod(rank,p['local_world_size'])
                env=dict(RANK=str(rank),WORLD_SIZE=str(p['world_size']),LOCAL_RANK=str(local),
                         LOCAL_WORLD_SIZE=str(p['local_world_size']),GROUP_RANK=str(node))
                out=rank_layout(p,env)
                self.assertEqual(out['node_rank'],node)
                self.assertEqual(out['primary'],local*p['teacher_model_parallel_size'])
                self.assertLess(out['devices'][-1],8)

    def test_reject_single_node_world_and_wrong_rank(self):
        p=topology('122b');p['teacher_key']='122b'
        env=dict(RANK='4',WORLD_SIZE='4',LOCAL_RANK='0',LOCAL_WORLD_SIZE='4',GROUP_RANK='1')
        with self.assertRaises(ValueError):rank_layout(p,env)
        env.update(WORLD_SIZE='8',GROUP_RANK='0')
        with self.assertRaises(ValueError):rank_layout(p,env)

    def test_safe_default_and_explicit_resume(self):
        args,layout=self.parse()
        self.assertEqual(args.mode,'probe')
        self.assertFalse(args.resume)
        args,layout=self.parse('--mode','train','--resume-step','33')
        self.assertTrue(args.resume)
        command=torchrun_command(args,layout,Path('/tmp/test-inv'),Path('/tmp/control.json'),python_paths())
        self.assertIn('--nnodes=2',command)
        self.assertIn('--nproc-per-node=4',command)
        self.assertIn('--max-restarts=0',command)

    def test_reject_loopback_placeholder_bad_port_and_wrong_process_count(self):
        for master in ('127.0.0.1','localhost','0.0.0.0','<node0_ip>'):
            with patch.dict(os.environ,{'NNODES':'2','NPROC_PER_NODE':''}),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):parse_args(['0',master])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):self.parse('--master-port','70000')
        with patch.dict(os.environ,{'NNODES':'2','NPROC_PER_NODE':'8'}),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):parse_args(['0','192.0.2.10'])

    def test_protocol_math_and_code_schedules(self):
        s=dict(layer_types=(['linear_attention']*3+['full_attention'])*8)
        for teacher,depth in [('122b',48),('397b',60)]:
            t=dict(layer_types=(['linear_attention']*3+['full_attention'])*(depth//4))
            for dataset,n,validation in [('dapo',16891,500),('lcb',780,100)]:
                args,layout=self.parse('--teacher',teacher,'--dataset',dataset)
                base=dict(examples_per_epoch=n,eval_questions=validation,batch_size=64)
                p=configure_protocol(base,s,t,options(args))
                self.assertEqual(p['initialization'],'zero')
                self.assertEqual(p['world_size'],layout['world_size'])
                self.assertEqual(p['vllm_replicas'],layout['world_size'])
                self.assertEqual(p['eval_steps'],[33,66,99,132] if dataset=='dapo' else [13,26,39,52])
                self.assertEqual(p['total_steps'],132 if dataset=='dapo' else 52)
                self.assertEqual(p['steps_per_epoch'],264 if dataset=='dapo' else 13)
                self.assertEqual(p['early_stop']['max_steps'],132 if dataset=='dapo' else 52)
                self.assertEqual(p['eval_samples'],4 if dataset=='dapo' else 16)
                self.assertTrue(all(a<=b for a,b in zip(p['layer_map'],p['layer_map'][1:])))
                self.assertTrue(all(t['layer_types'][j]=='full_attention' for i,j in enumerate(p['layer_map'])
                                    if s['layer_types'][i]=='full_attention'))
                self.assertEqual(p['teacher_gpu_groups'][-1][-1],7)

    def test_cpu_probe_never_qualifies_for_real_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,layout=self.parse('--root',tmp,'--mode','smoke')
            Path(tmp,'probe_audit.json').write_text(json.dumps(dict(passed=True,nccl_validated=False,
                cpu_simulation=True,topology=layout,source_manifest_sha256='test')))
            with self.assertRaises(RuntimeError):prepare_control(args,layout,Path(tmp)/'inv','test')

    def test_require_every_global_rank_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'train/checkpoints/step-0033';d.mkdir(parents=True)
            for name in ('bridge.pt','optimizer.pt','checkpoint.json'):(d/name).touch()
            for rank in range(7):(d/f'rng.rank-{rank:02d}.pt').touch()
            self.assertFalse(complete_checkpoint(root,33,8))
            (d/'rng.rank-07.pt').touch()
            self.assertTrue(complete_checkpoint(root,33,8))

    def test_zero_bridge_recovers_native_and_all_maps_get_gradients(self):
        bridge=MonotoneFP8Bridge(model(32,8),model(64,12))
        initialize_zero(bridge)
        self.assertTrue(all(torch.count_nonzero(p).item()==0 for p in bridge.bridge_parameters()))
        prompt,response=[1,4,5,6],[7,8,9,2]
        lp=sampled_log_probs(bridge,prompt,response)
        with torch.no_grad():
            logits=bridge.student(input_ids=torch.tensor([prompt+response[:-1]]),use_cache=False).logits[0,len(prompt)-1:].float()
            expected=logits.log_softmax(-1).gather(1,torch.tensor(response)[:,None])[:,0]
        torch.testing.assert_close(lp,expected,rtol=0,atol=1e-6)
        bridge.student_activation_checkpointing=True
        lp=sampled_log_probs(bridge,prompt,response)
        policy_loss(lp,1.,lp.detach())[0].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0
                            for p in bridge.bridge_parameters()))
        self.assertNotEqual(bridge.maps[2].weight.data_ptr(),bridge.maps[3].weight.data_ptr())
        self.assertTrue(all(p.grad is None and not p.requires_grad for m in (bridge.teacher,bridge.student)
                            for p in m.parameters()))


if __name__=='__main__':
    unittest.main()
