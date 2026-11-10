"""GRPO outcome advantages and differentiable frozen-backbone bridge policy."""
from collections import defaultdict
from contextlib import contextmanager
import torch
from torch.utils.checkpoint import checkpoint


def group_advantages(rows, group_size=8):
    groups=defaultdict(list)
    for row in rows:
        groups[row['index']].append(row)
    advantages={}
    counts=dict(groups=len(groups),mixed_groups=0,all_correct_groups=0,
                all_wrong_groups=0,missing_gold_groups=0)
    for index, group in groups.items():
        assert len(group)==group_size
        assert {r['sample_index'] for r in group}==set(range(group_size))
        assert len({r['gold_missing'] for r in group})==1
        if group[0]['gold_missing']:
            values=torch.zeros(group_size)
            counts['missing_gold_groups']+=1
        else:
            assert all(r['acc'] in (0,1) and not r['scoring_timed_out'] for r in group)
            rewards=torch.tensor([float(r['acc']) for r in group])
            # Match VERL GRPO: sample std (correction=1), epsilon=1e-6.
            values=(rewards-rewards.mean())/(rewards.std(correction=1)+1e-6)
            correct=int(rewards.sum())
            counts['all_correct_groups' if correct==group_size else
                   'all_wrong_groups' if correct==0 else 'mixed_groups']+=1
        for row, value in zip(group,values,strict=True):
            advantages[(index,row['sample_index'])]=float(value)
    return advantages,counts


@contextmanager
def checkpoint_student_blocks(bridge):
    """Checkpoint block internals after prefix injection, retaining eval mode.

    Injection is a pre-hook: it runs once before the wrapped forward, so the
    backward replay consumes the already-injected hidden tensor. It must not
    depend on hooks that have been removed after the forward context exits.
    """
    originals=[]
    try:
        for layer in bridge.student.model.layers:
            original=layer.forward
            def wrapped(*args, _forward=original, **kwargs):
                return checkpoint(_forward,*args,use_reentrant=False,
                                  preserve_rng_state=True,**kwargs)
            originals.append((layer, original, 'forward' in layer.__dict__))
            layer.forward=wrapped
        yield
    finally:
        for layer, original, had_instance_forward in originals:
            if had_instance_forward:
                layer.forward=original
            else:
                del layer.forward


def sampled_log_probs(bridge,prompt,response,chunk_tokens=32):
    """All response positions, including EOS; teacher encodes only the prompt."""
    if len(prompt)<2 or not response:
        raise ValueError('A prompt and a nonempty sampled response are required')
    device=bridge.student.device
    if hasattr(bridge,'response_hidden'):
        hidden=bridge.response_hidden(prompt,response)
    else:
        prefix=torch.tensor([prompt],device=device)
        with torch.no_grad():
            teacher=bridge.teacher.model(input_ids=prefix,attention_mask=torch.ones_like(prefix),
                position_ids=torch.arange(len(prompt),device=device)[None],use_cache=False,
                output_hidden_states=True)
            features=tuple(h.contiguous().clone() for h in bridge.teacher_features(teacher.hidden_states))
            del teacher
        ids=torch.tensor([prompt+response[:-1]],device=device)
        injection=torch.ones((1,len(prompt)),device=device,dtype=torch.bool)
        injection[:,-1]=False
        from contextlib import nullcontext
        checkpointing = (torch.is_grad_enabled() and
                         getattr(bridge, 'student_activation_checkpointing', False))
        context = checkpoint_student_blocks(bridge) if checkpointing else nullcontext()
        with context, bridge.inject(features,injection):
            hidden=bridge.student.model(input_ids=ids,attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(ids.shape[1],device=device)[None],use_cache=False).last_hidden_state[:,len(prompt)-1:]
    labels=torch.tensor(response,device=device)
    hidden=hidden[0]
    assert len(hidden)==len(labels)
    assert all(not p.requires_grad for p in bridge.student.lm_head.parameters())
    def chunk(h,tokens):
        logits=bridge.student.lm_head(h).float()
        return logits.gather(1,tokens[:,None]).squeeze(1)-logits.logsumexp(-1)
    parts=[]
    for start in range(0,len(response),chunk_tokens):
        h,tokens=hidden[start:start+chunk_tokens],labels[start:start+chunk_tokens]
        parts.append(checkpoint(chunk,h,tokens,use_reentrant=False,preserve_rng_state=False)
            if torch.is_grad_enabled() and h.requires_grad else chunk(h,tokens))
    return torch.cat(parts)


def policy_loss(log_probs,advantage,rollout_log_probs,old_log_probs=None,
                clip_ratio=.2,dual_clip=3.,correction_cap=2.):
    """VERL vanilla clipped GRPO, seq-mean-token-mean, token rollout IS.

    One actor minibatch and one PPO epoch are fully on-policy: VERL uses
    actor log_probs.detach() as old policy in this case. Native vLLM sampled
    log probabilities are distinct and provide detached token IS correction.
    No reference-policy KL, OPD KL or entropy term is added.
    """
    assert log_probs.ndim==1 and log_probs.numel()>0
    old=log_probs.detach() if old_log_probs is None else old_log_probs.detach()
    rollout=torch.as_tensor(rollout_log_probs,device=log_probs.device,dtype=log_probs.dtype)
    assert old.shape==rollout.shape==log_probs.shape
    assert torch.isfinite(log_probs).all() and torch.isfinite(rollout).all()
    advantage=torch.as_tensor(advantage,device=log_probs.device,dtype=log_probs.dtype)
    assert advantage.numel()==1 and torch.isfinite(advantage)
    log_ratio=(log_probs-old).clamp(-20.,20.)
    ratio=log_ratio.exp()
    raw=-advantage*ratio
    clipped=-advantage*ratio.clamp(1-clip_ratio,1+clip_ratio)
    upper=torch.maximum(raw,clipped)
    bounded=torch.where(advantage<0,torch.minimum(-advantage*dual_clip,upper),upper)
    correction=(old-rollout).clamp(-20.,20.).exp().clamp_max(correction_cap).detach()
    loss=(bounded*correction).mean()
    metrics=dict(pg_clip_fraction=(clipped>raw).float().mean().detach(),
        ppo_kl=(-log_ratio).mean().detach(),
        rollout_is_mean=correction.mean().detach(),
        rollout_is_clipped_fraction=((old-rollout)>torch.log(log_probs.new_tensor(correction_cap))).float().mean().detach(),
        actor_rollout_logprob_abs_error=(old-rollout).abs().mean().detach())
    return loss,metrics


def grpo_loss(bridge,prompt,response,advantage,rollout_log_probs):
    return policy_loss(sampled_log_probs(bridge,prompt,response),advantage,rollout_log_probs)
