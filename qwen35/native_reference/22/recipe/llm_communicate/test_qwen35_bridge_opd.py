"""Check real hybrid-model reverse-KL gradients, masks and frozen backbones."""
import inspect
import unittest
from unittest.mock import patch
import torch
from qwen35_bridge_opd_core import initialize, opd_loss, weight_digest, first_tokens, TRAIN_SAMPLING, EVAL_SAMPLING
from qwen35_last_aligned_math_bridge import LastAlignedResidualBridge
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
import transformers.models.qwen3_5.modeling_qwen3_5 as modeling


def model(width,depth):
    config=Qwen3_5TextConfig(vocab_size=64,hidden_size=width,intermediate_size=width*2,
        num_hidden_layers=depth,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
        layer_types=(['linear_attention']*3+['full_attention'])*(depth//4),full_attention_interval=4,
        linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=8,linear_value_head_dim=8,
        linear_conv_kernel_dim=4,max_position_embeddings=256,eos_token_id=2,pad_token_id=0,
        rope_parameters=dict(rope_type='default',rope_theta=1000.,partial_rotary_factor=1.,
            mrope_interleaved=True,mrope_section=[2,2,4]))
    config._attn_implementation='sdpa'
    return Qwen3_5ForCausalLM(config).eval()


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        def reference(function):
            # Recent kernels packages wrap functions in an nn.Module; peel
            # that wrapper before unwrap, so CPU tests never call FLA Triton.
            if isinstance(function,torch.nn.Module):
                function=inspect.getclosurevars(function.forward).nonlocals['func']
            return inspect.unwrap(function)
        for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule',
                     'causal_conv1d_fn','causal_conv1d_update'):
            # Installed causal-conv1d also selects CUDA kernels on CPU inputs.
            # Use the reference implementation only inside these CPU fixtures.
            kernel_patch=patch.object(modeling,name,reference(getattr(modeling,name)))
            kernel_patch.start()
            cls.addClassCleanup(kernel_patch.stop)

    def bridge(self):
        torch.manual_seed(42)
        bridge=LastAlignedResidualBridge(model(32,4),model(64,8))
        initialize(bridge)
        return bridge

    def test_reverse_kl_full_vocabulary_and_gradients(self):
        bridge=self.bridge()
        prompt,response=[1,4,5,6],[7,8,9,2]
        loss=opd_loss(bridge,prompt,response,chunk_tokens=2)
        loss.backward()
        original=[p.grad.clone() for p in bridge.bridge_parameters()]
        for p in bridge.bridge_parameters(): p.grad=None
        ids=torch.tensor([prompt+response[:-1]])
        with torch.no_grad():
            teacher=bridge.teacher.model(input_ids=ids,attention_mask=torch.ones_like(ids),
                use_cache=False,output_hidden_states=True)
            features=tuple(h[:,:len(prompt)] for h in bridge.teacher_features(teacher.hidden_states))
            logits_t=bridge.teacher.lm_head(teacher.last_hidden_state[:,len(prompt)-1:]).float()
        mask=torch.tensor([[True,True,True,False]])
        with bridge.inject(features,mask):
            hidden=bridge.student.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
        logits_s=bridge.student.lm_head(hidden[:,len(prompt)-1:]).float()
        ls,lt=logits_s.log_softmax(-1),logits_t.log_softmax(-1)
        expected=(ls.exp()*(ls-lt)).sum(-1).mean()
        expected.backward()
        self.assertTrue(torch.allclose(loss,expected,atol=2e-7,rtol=2e-6))
        for p,gradient in zip(bridge.bridge_parameters(),original):
            self.assertTrue(torch.allclose(p.grad,gradient,atol=2e-7,rtol=2e-5))
            self.assertGreater(float(p.grad.norm()),0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for m in [bridge.student,bridge.teacher] for p in m.parameters()))

    def test_global_token_mean_with_uneven_shards(self):
        bridge=self.bridge()
        pairs=[([1,4,5],[6,7,2]),([1,8,9,10],[11,2])]
        gradients=[]
        for prompt,response in pairs:
            for p in bridge.bridge_parameters(): p.grad=None
            (opd_loss(bridge,prompt,response)*len(response)/5).backward()
            gradients.append([p.grad.clone() for p in bridge.bridge_parameters()])
        for p in bridge.bridge_parameters(): p.grad=None
        total=sum(opd_loss(bridge,prompt,response)*len(response)/5 for prompt,response in pairs)
        total.backward()
        for p,a,b in zip(bridge.bridge_parameters(),*gradients):
            self.assertTrue(torch.allclose(p.grad,a+b,atol=2e-7,rtol=2e-5))

    def test_initialization_digest_and_sampling(self):
        a,b=self.bridge(),self.bridge()
        self.assertEqual(weight_digest(a),weight_digest(b))
        self.assertEqual(a.layer_map,[4,5,6,7])
        bound=1e-4/(64**.5)
        self.assertTrue(all(0<float(p.abs().max())<=bound for p in a.bridge_parameters()))
        before=weight_digest(a)
        torch.optim.AdamW(a.bridge_parameters(),lr=1e-5,weight_decay=0).zero_grad()
        value=opd_loss(a,[1,4,5],[6,2]);value.backward()
        torch.optim.AdamW(a.bridge_parameters(),lr=1e-5,weight_decay=0).step()
        self.assertNotEqual(weight_digest(a),before)
        rows=[dict(sampling_seed=42),dict(sampling_seed=67)]
        logits=torch.arange(64,dtype=torch.float)[None].repeat(2,1)/10
        for sampling in [TRAIN_SAMPLING,EVAL_SAMPLING]:
            together=first_tokens(logits,rows,4096,sampling)
            for i,row in enumerate(rows):
                self.assertEqual(together[i],first_tokens(logits[i:i+1],[row],4096,sampling)[0])


if __name__=='__main__':
    unittest.main()
