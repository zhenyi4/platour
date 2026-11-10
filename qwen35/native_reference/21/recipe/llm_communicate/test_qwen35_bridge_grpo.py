"""Check group rewards, full selected-token gradients and clipped GRPO."""
import unittest
import torch
from test_qwen35_bridge_opd import Tests as HybridFixture
from qwen35_bridge_grpo_core import group_advantages,sampled_log_probs,policy_loss


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HybridFixture.setUpClass()

    def test_group_advantages_shuffled_and_missing_gold(self):
        rows=[]
        for index,correct in [(5,3),(7,8),(9,0),(11,None)]:
            for j in range(8):
                rows.append(dict(index=index,sample_index=j,acc=None if correct is None else int(j<correct),
                    gold_missing=correct is None,scoring_timed_out=False))
        shuffled=[rows[i] for i in torch.randperm(len(rows),generator=torch.Generator().manual_seed(42)).tolist()]
        values,counts=group_advantages(shuffled)
        self.assertEqual(counts,dict(groups=4,mixed_groups=1,all_correct_groups=1,all_wrong_groups=1,missing_gold_groups=1))
        rewards=torch.tensor([1.,1.,1.,0.,0.,0.,0.,0.])
        expected=(rewards-rewards.mean())/(rewards.std()+1e-6)
        self.assertTrue(torch.allclose(torch.tensor([values[(5,j)] for j in range(8)]),expected))
        self.assertTrue(all(values[(index,j)]==0 for index in [7,9,11] for j in range(8)))
        with self.assertRaises(AssertionError):group_advantages(rows[:-1])

    def test_selected_token_logprob_gradient_and_eos(self):
        bridge=HybridFixture().bridge()
        prompt,response=[1,4,5,6],[7,8,9,2]
        actual=sampled_log_probs(bridge,prompt,response,chunk_tokens=2)
        (-actual.mean()).backward()
        gradients=[p.grad.clone() for p in bridge.bridge_parameters()]
        for p in bridge.bridge_parameters():p.grad=None
        prefix=torch.tensor([prompt])
        with torch.no_grad():
            teacher=bridge.teacher.model(input_ids=prefix,attention_mask=torch.ones_like(prefix),
                use_cache=False,output_hidden_states=True)
            features=bridge.teacher_features(teacher.hidden_states)
        ids=torch.tensor([prompt+response[:-1]])
        with bridge.inject(features,torch.tensor([[True,True,True,False]])):
            hidden=bridge.student.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
        logits=bridge.student.lm_head(hidden[:,len(prompt)-1:]).float()
        expected=logits.log_softmax(-1).gather(-1,torch.tensor(response)[None,:,None])[0,:,0]
        (-expected.mean()).backward()
        self.assertTrue(torch.allclose(actual,expected,atol=2e-6,rtol=2e-6))
        for p,gradient in zip(bridge.bridge_parameters(),gradients):
            self.assertTrue(torch.allclose(p.grad,gradient,atol=2e-7,rtol=2e-5))
            self.assertGreater(float(p.grad.norm()),0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for m in [bridge.student,bridge.teacher] for p in m.parameters()))

    def test_grpo_gradient_and_detached_rollout_correction(self):
        logits=torch.tensor([[.2,-.7,1.1],[1.3,.1,-.2]],requires_grad=True)
        selected=logits.log_softmax(-1)[torch.arange(2),torch.tensor([0,2])]
        rollout=(selected.detach()-torch.tensor([.1,2.])).clone().requires_grad_()
        advantage=-.8
        loss,metrics=policy_loss(selected,advantage,rollout)
        loss.backward()
        actual=logits.grad.clone()
        self.assertIsNone(rollout.grad)
        logits.grad=None
        correction=torch.tensor([.1,2.]).exp().clamp_max(2.)
        expected=-(advantage*correction*logits.log_softmax(-1)[torch.arange(2),torch.tensor([0,2])]).mean()
        expected.backward()
        self.assertTrue(torch.allclose(actual,logits.grad,atol=1e-7))
        self.assertAlmostEqual(float(metrics['rollout_is_clipped_fraction']),.5)
        # Both signs: ordinary clip and negative-advantage dual clip.
        for advantage in [-1.,1.]:
            new=torch.tensor([-2.4,-1.5,.1],requires_grad=True)
            old=torch.full((3,),-2.)
            loss,_=policy_loss(new,advantage,old,old_log_probs=old)
            ratio=(new-old).exp()
            raw=-advantage*ratio
            expected=torch.maximum(raw,-advantage*ratio.clamp(.8,1.2))
            if advantage<0:expected=expected.clamp_max(-advantage*3.)
            self.assertTrue(torch.allclose(loss,expected.mean()))

    def test_activation_checkpoint_preserves_logprobs_and_all_projection_gradients(self):
        bridge=HybridFixture().bridge()
        prompt,response=[1,4,5,6],[7,8,9,2]
        expected=sampled_log_probs(bridge,prompt,response,chunk_tokens=2)
        (-expected.mean()).backward()
        gradients=[p.grad.clone() for p in bridge.bridge_parameters()]
        for p in bridge.bridge_parameters():p.grad=None
        bridge.student_activation_checkpointing=True
        actual=sampled_log_probs(bridge,prompt,response,chunk_tokens=2)
        # Backward happens after injection hooks and forward wrappers are removed.
        self.assertTrue(all(not layer._forward_pre_hooks for layer in bridge.student.model.layers))
        self.assertTrue(all('forward' not in layer.__dict__ for layer in bridge.student.model.layers))
        (-actual.mean()).backward()
        self.assertTrue(torch.equal(actual,expected))
        for p, gradient in zip(bridge.bridge_parameters(),gradients):
            self.assertTrue(torch.allclose(p.grad,gradient,atol=2e-7,rtol=2e-5))
            self.assertGreater(float(p.grad.norm()),0)
        self.assertFalse(bridge.student.training)
        self.assertTrue(all(p.grad is None for m in [bridge.student,bridge.teacher] for p in m.parameters()))


if __name__=='__main__':unittest.main()
