"""Standalone full-parameter and rank-128 LoRA Qwen3.5-4B GRPO actors."""
from contextlib import nullcontext
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from qwen35_bridge_opd_core import save,weight_digest


class LoRALinear(nn.Module):
    def __init__(self,base,rank=128,alpha=128):
        super().__init__()
        assert base.bias is None
        self.base=base.requires_grad_(False)
        self.rank,self.alpha=rank,alpha
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))

    def effective_weight(self):
        return (self.base.weight.float()+(self.lora_B@self.lora_A)*(self.alpha/self.rank)).to(self.base.weight.dtype)

    def forward(self,x):
        delta=F.linear(F.linear(x,self.lora_A.to(x.dtype)),self.lora_B.to(x.dtype))
        return self.base(x)+delta*(self.alpha/self.rank)


class MasterAdamW(torch.optim.Optimizer):
    """BF16 actor/gradients, FP32 Adam moments and FP32 master weights.

    FP32 masters prevent lr=1e-6 updates from being lost to BF16 rounding.
    Exact masters/moments are included in the optimizer checkpoint.
    """
    def __init__(self,params,lr):
        super().__init__(params,dict(lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.))

    @torch.no_grad()
    def allocate_state(self):
        for group in self.param_groups:
            for p in group['params']:
                if not self.state[p]:
                    self.state[p].update(step=0,master=p.float().clone(),exp_avg=torch.zeros_like(p,dtype=torch.float32),
                        exp_avg_sq=torch.zeros_like(p,dtype=torch.float32))

    def load_state_dict(self,payload):
        # Optimizer's default loader casts moments to the parameter dtype.
        # That would round FP32 masters/moments to BF16; map them explicitly.
        old=[p for g in payload['param_groups'] for p in g['params']]
        current=[p for g in self.param_groups for p in g['params']]
        assert len(old)==len(current)
        self.state.clear()
        for key,p in zip(old,current,strict=True):
            values=payload['state'].get(key,{})
            if values:
                self.state[p]={n:(v.to(device=p.device,dtype=torch.float32).clone() if torch.is_tensor(v) else v)
                               for n,v in values.items()}
        for old_group,new_group in zip(payload['param_groups'],self.param_groups,strict=True):
            new_group.update({k:v for k,v in old_group.items() if k!='params'})

    @torch.no_grad()
    def step(self,closure=None):
        assert closure is None
        for group in self.param_groups:
            beta1,beta2=group['betas']
            for p in group['params']:
                if p.grad is None:continue
                state=self.state[p]
                if not state:
                    state.update(step=0,master=p.float().clone(),exp_avg=torch.zeros_like(p,dtype=torch.float32),
                        exp_avg_sq=torch.zeros_like(p,dtype=torch.float32))
                state['step']+=1
                grad=p.grad.float()
                state['exp_avg'].lerp_(grad,1-beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad,grad,value=1-beta2)
                denom=state['exp_avg_sq'].sqrt().div_(math.sqrt(1-beta2**state['step'])).add_(group['eps'])
                state['master'].addcdiv_(state['exp_avg'],denom,value=-group['lr']/(1-beta1**state['step']))
                p.copy_(state['master'])


class NativeActor(nn.Module):
    trainable_actor=True
    def __init__(self,student,mode,rank=128,alpha=128):
        super().__init__()
        self.student=student
        self.mode,self.rank,self.alpha=mode,rank,alpha
        self.layer_map=[]
        self.student_activation_checkpointing=True
        self.target_modules=[]
        if mode=='full':
            student.requires_grad_(True)
        else:
            assert mode=='lora'
            student.requires_grad_(False)
            for name,module in list(student.model.named_modules()):
                if isinstance(module,nn.Linear):
                    parent,leaf=name.rsplit('.',1)
                    setattr(student.model.get_submodule(parent),leaf,LoRALinear(module,rank,alpha))
                    self.target_modules.append('model.'+name)
            expected=student.config.num_hidden_layers*7 if student.config.model_type in ('llama','qwen3') else 248
            assert len(self.target_modules)==expected,len(self.target_modules)
        self._params=tuple(p for p in student.parameters() if p.requires_grad)

    def initialize_for_grpo(self,seed=42):
        return dict(kind='native pretrained actor actor' if self.mode=='full' else
            'LoRA A Kaiming uniform; B exactly zero; native frozen base',
            seed=seed,mode=self.mode,sft_checkpoint_loaded=False,teacher_model_loaded=False)

    def bridge_parameters(self):return self._params

    def bridge_state_tensors(self):
        return {n:p for n,p in self.student.named_parameters() if self.mode=='full' or p.requires_grad}

    def bridge_checkpoint_state(self):
        return dict(mode=self.mode,rank=self.rank,alpha=self.alpha,
            actor={n:t.detach().cpu().clone() for n,t in self.bridge_state_tensors().items()})

    def load_checkpoint(self,path):
        payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        assert payload['mode']==self.mode and payload['rank']==self.rank and payload['alpha']==self.alpha
        self.restore_state(payload['actor'])

    @torch.no_grad()
    def restore_state(self,state):
        current=self.bridge_state_tensors()
        assert current.keys()==state.keys()
        for n,p in current.items():p.copy_(state[n])

    def make_optimizer(self,lr):
        return MasterAdamW(self._params,lr) if self.mode=='full' else torch.optim.AdamW(self._params,
            lr=lr,weight_decay=0.,betas=(.9,.999))

    def prepare_optimizer_for_stress(self,optimizer):
        if self.mode=='full':
            optimizer.allocate_state()
        else:
            for p in self._params:
                if not optimizer.state[p]:
                    optimizer.state[p]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))

    def assert_trainable_parameters(self):
        expected={id(p) for p in self._params}
        for p in self.student.parameters():
            assert p.requires_grad==(id(p) in expected)
            if not p.requires_grad:assert p.grad is None

    def response_hidden(self,prompt,response):
        from qwen35_bridge_grpo_core import checkpoint_student_blocks
        ids=torch.tensor([prompt+response[:-1]],device=self.student.device)
        context=checkpoint_student_blocks(self) if torch.is_grad_enabled() and self.student_activation_checkpointing else nullcontext()
        with context:
            return self.student.model(input_ids=ids,attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(ids.shape[1],device=ids.device)[None],
                use_cache=False).last_hidden_state[:,len(prompt)-1:]

    @torch.no_grad()
    def merged_state(self):
        if self.mode=='full':return self.student.state_dict()
        result={}
        for name,t in self.student.state_dict().items():
            if name.endswith(('.lora_A','.lora_B')):continue
            if '.base.weight' in name:
                parent=name[:-len('.base.weight')]
                result[parent+'.weight']=self.student.get_submodule(parent).effective_weight()
            else:result[name]=t
        return result

    def generate_native_actor(self,engine,rows,digest,sampling,cap,directory,return_logprobs):
        from qwen35_native_actor_vllm_worker import tensor_digest
        directory.mkdir(parents=True,exist_ok=True)
        start=time.monotonic()
        # Empty ranks still participate in barriers, but have no decode work.
        if not rows:
            result=dict(complete=True,rows=[],seconds=0.,bridge_state_sha256=digest)
            save(directory/'responses.json',result)
            return result,0.
        path=engine.directory/'current_actor_weights.pt'
        if getattr(engine,'actor_identity',None)!=digest:
            if self.mode=='full':
                from qwen35_full_actor_sync import shared_full_actor_snapshot
                path,actor_digest=shared_full_actor_snapshot(self,engine,digest)
            else:
                state=self.bridge_state_tensors()
                actor_digest=tensor_digest(state)
                from safetensors.torch import save_file
                path=engine.directory/'current_lora'
                path.mkdir(exist_ok=True)
                tensors={}
                for name,t in state.items():
                    qualified=('base_model.model.'+name+'.weight' if self.student.config.model_type in ('llama','qwen3') else
                        'base_model.model.model.language_model.'+name[len('model.'):]+'.weight')
                    if '.in_proj_qkv.' in qualified:
                        parts=[t]*3 if name.endswith('.lora_A') else list(t.split([2048,2048,4096],dim=0))
                        for channel,part in zip(('q','k','v'),parts,strict=True):
                            tensors[qualified.replace('.in_proj_qkv.','.in_proj_'+channel+'.')]=part.detach().cpu().contiguous().clone()
                    else:
                        tensors[qualified]=t.detach().cpu().contiguous()
                save_file(tensors,str(path/'adapter_model.safetensors'))
                save(path/'adapter_config.json',dict(peft_type='LORA',task_type='CAUSAL_LM',
                    base_model_name_or_path=str(self.student.config._name_or_path),r=self.rank,
                    lora_alpha=self.alpha,lora_dropout=0.,bias='none',inference_mode=True,
                    target_modules=sorted(({n.rsplit('.',1)[-1] for n in self.target_modules}-{'in_proj_qkv'})|
                        (set() if self.student.config.model_type in ('llama','qwen3') else {'in_proj_q','in_proj_k','in_proj_v'})),
                    use_rslora=False,use_dora=False))
                # Source tensors use trainable actor names; separate audit is
                # needed because the PEFT file has checkpoint-compatible names.
                torch.save({n:t.detach().cpu() for n,t in state.items()},path/'actor_source.pt')
                del state
            engine.actor_identity=digest
            engine.actor_digest=actor_digest
            engine.actor_path=str(path)
        manifest=directory/'manifest.json'
        save(manifest,dict(rows=rows,tokens=cap,sampling=sampling,return_logprobs=return_logprobs,
            actor_weights=engine.actor_path,actor_state_sha256=engine.actor_digest,actor_training_mode=self.mode,
            bridge_state_sha256=digest,prefill_backend='native vLLM',decode_backend='native vLLM'))
        export=time.monotonic()-start
        result=engine.call(dict(manifest=str(manifest),result=str(directory/'responses.json')))
        assert result['bridge_state_sha256']==digest and len(result['rows'])==len(rows)
        return result,export+result.get('actor_sync_seconds',0.)

    def metadata(self):
        return dict(recipe='qwen35_native_actor',mode=self.mode,teacher_model_loaded=False,
            bridge_projection_present=False,student_frozen=self.mode=='lora',layer_map=[],
            trainable_parameters=sum(p.numel() for p in self._params),rank=self.rank if self.mode=='lora' else None,
            lora_alpha=self.alpha if self.mode=='lora' else None,lora_dropout=0.,target_modules=self.target_modules,
            forward_dtype='bfloat16',optimizer_state_dtype='float32',
            full_actor_optimizer='FP32 master AdamW' if self.mode=='full' else 'FP32 AdamW',
            lora_forward='native frozen W(x) + (alpha/r) B(A(x)); BF16 forward, FP32 factor masters' if self.mode=='lora' else None,
            weight_sync='native vLLM LoRA adapter' if self.mode=='lora' else 'disk -> native vLLM load_weights; exact all-tensor audit',
            inference='native vLLM prefill and all response tokens; no bridge cache transfer')


def load_bridge(root,protocol,checkpoint=None):
    if protocol.get('model_family')=='qwen3':
        from transformers import AutoModelForCausalLM
        if protocol.get('hf_attention_math_only',False):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        student=AutoModelForCausalLM.from_pretrained(protocol['student_checkpoint'],dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
        actor=NativeActor(student,protocol['actor_training_mode'],protocol.get('lora_rank',128),
            protocol.get('lora_alpha',128)).eval()
        if 'lora_parity_limits' in protocol:actor.parity_limits=protocol['lora_parity_limits']
        actor.bf16_backend_acceptance=protocol.get('bf16_backend_acceptance')
        if checkpoint is not None:actor.load_checkpoint(checkpoint)
        return actor,actor.metadata()
    from qwen35_direct_math_bridge import accelerated_models
    from qwen35_student import load_student
    runtime=accelerated_models()
    actor=NativeActor(load_student(Path(protocol['student_checkpoint'])),protocol['actor_training_mode'],
        protocol.get('lora_rank',128),protocol.get('lora_alpha',128)).eval()
    actor.full_actor_snapshot_retention=protocol.get('full_actor_snapshot_retention')
    if checkpoint is not None:actor.load_checkpoint(checkpoint)
    runtime.update(actor.metadata())
    return actor,runtime


def smoke_audit(actor,engine,rows,directory,pad_id,eos):
    from qwen35_bridge_grpo_core import sampled_log_probs
    from qwen35_bridge_opd_core import TRAIN_SAMPLING
    directory.mkdir(parents=True,exist_ok=True)
    initial=weight_digest(actor)
    backup={n:t.detach().cpu().clone() for n,t in actor.bridge_state_tensors().items()}
    cpu_rng,cuda_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
    optimizer=actor.make_optimizer(1e-6 if actor.mode=='full' else 1e-5)
    row=min(rows,key=lambda r:len(r['prompt_token_ids']))
    # Real actor update before parity probing. Audit optimizer is discarded.
    norms=[]
    try:
        save(directory/'live.json',dict(stage='synthetic_actor_updates',mode=actor.mode))
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss=-sampled_log_probs(actor,row['prompt_token_ids'],[16,17,18,19]).mean()
            loss.backward()
            actor.assert_trainable_parameters()
            norm=torch.nn.utils.clip_grad_norm_(actor.bridge_parameters(),1.,error_if_nonfinite=True)
            assert norm>0
            norms.append(float(norm))
            optimizer.step()
            save(directory/'live.json',dict(stage='synthetic_actor_updates',completed=step+1,gradient_norms=norms))
        trained=weight_digest(actor)
        assert trained!=initial
        changed=sum(int(torch.count_nonzero(t.detach().cpu()!=backup[n])) for n,t in actor.bridge_state_tensors().items())
        assert changed>0
        path=directory/'roundtrip.pt'
        save(directory/'live.json',dict(stage='checkpoint_roundtrip'))
        torch.save(actor.bridge_checkpoint_state(),path)
        actor.restore_state(backup); actor.load_checkpoint(path)
        assert weight_digest(actor)==trained
        if actor.mode=='full':
            # The roundtrip checkpoint is also the native weight sync source;
            # do not serialize the same 8.4 GB actor a second time.
            engine.actor_identity=trained
            engine.actor_digest=trained
            engine.actor_path=str(path)
        else:path.unlink()
        probes=[dict(**r,sample_index=0,sampling_seed=1234+4*r['index']) for r in rows[:3]]
        save(directory/'live.json',dict(stage='post_update_vllm_parity'))
        result,_=actor.generate_native_actor(engine,probes,trained,TRAIN_SAMPLING,16,directory/'post_update_parity',True)
        assert all(x['all_actor_weights_equal'] for x in result['actor_weight_audits'])
        errors=[]
        with torch.no_grad():
            for r in result['rows']:
                lp=sampled_log_probs(actor,r['prompt_token_ids'],r['response_token_ids'])
                errors.extend((lp-torch.tensor(r['rollout_log_probs'],device=lp.device)).abs().tolist())
        values=torch.tensor(errors)
        parity=dict(mean_abs_logprob_error=sum(errors)/len(errors),max_abs_logprob_error=max(errors),
            p95_abs_logprob_error=float(torch.quantile(values,.95)),p99_abs_logprob_error=float(torch.quantile(values,.99)),
            tokens=len(errors),errors_above_point1=sum(x>.1 for x in errors))
        ratios=[]
        with torch.no_grad():
            for r in result['rows']:
                lp=sampled_log_probs(actor,r['prompt_token_ids'],r['response_token_ids'])
                ratios.extend((lp-torch.tensor(r['rollout_log_probs'],device=lp.device)).exp().tolist())
        parity.update(importance_ratio_min=min(ratios),importance_ratio_max=max(ratios))
        # BF16 logits have spacing .125 at 16..32 and .25 at 32..64.
        # A blanket max-error .2 rejects a single rounded logit even with
        # exact weights. Gate the full distribution and token IS instead.
        limits=getattr(actor,'parity_limits',dict(mean=.02,p95=.05,p99=.2,max=.3,importance_min=.7,importance_max=1.35))
        strict_passed=(parity['mean_abs_logprob_error']<limits['mean'] and
                parity['p95_abs_logprob_error']<limits['p95'] and
                parity['p99_abs_logprob_error']<limits['p99'] and
                parity['max_abs_logprob_error']<limits['max'] and
                parity['importance_ratio_min']>limits['importance_min'] and
                parity['importance_ratio_max']<limits['importance_max'])
        alternate=getattr(actor,'bf16_backend_acceptance',None)
        backend_accepted=False
        if alternate is not None:
            backend_accepted=(parity['mean_abs_logprob_error']<alternate['mean'] and
                parity['p95_abs_logprob_error']<alternate['p95'] and
                parity['p99_abs_logprob_error']<alternate['p99'] and
                parity['max_abs_logprob_error']<alternate['max'] and
                parity['importance_ratio_min']>alternate['importance_min'] and
                parity['importance_ratio_max']<alternate['importance_max'])
        assert strict_passed or backend_accepted,parity
        if actor.mode=='full':path.unlink()
        save(directory/'audit.json',dict(passed=True,mode=actor.mode,post_update_hf_vllm_parity=parity,
            actor_weights_exactly_synced=True,checkpoint_roundtrip=True,
            strict_parity_passed=strict_passed,bf16_backend_accepted=backend_accepted,
            bf16_backend_acceptance=alternate,
            changed_actor_coordinates=changed,gradient_norms=norms,teacher_model_loaded=False,
            bridge_projection_present=False,audit_optimizer_discarded=True,parity_limits=limits))
    finally:
        optimizer.zero_grad(set_to_none=True)
        actor.restore_state(backup)
        engine.actor_identity=None
        del optimizer,backup
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
        assert weight_digest(actor)==initial
        torch.cuda.empty_cache()


if __name__=='__main__':
    raise SystemExit('Use run_qwen35_native_actor_grpo_smoke.py')
