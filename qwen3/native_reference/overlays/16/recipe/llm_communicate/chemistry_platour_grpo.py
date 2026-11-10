"""Qwen3 dense/500-train-sample Gaussian Chemistry PLatour GRPO recipes.

Default command prepares and runs SMOKE ONLY for all three variants.
Formal training requires the explicit `train` stage and uses a new directory.
"""
import argparse
import ast
from collections import OrderedDict
from contextlib import nullcontext
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import types

import torch
from torch import nn
from torch.nn import functional as F
from qwen35_bridge_opd_core import ROOT,save,weight_digest,TRAIN_SAMPLING
from arc_bridge_sft_common import sha
from direct_residual_chat_bridge import DirectResidualChatBridge
from qwen35_virtual_teacher_bridge import VirtualTeacherBridge

HERE=Path(__file__).resolve().parent
SCRIPT=Path(__file__).resolve()
PREVIOUS_SMOKES=ROOT/'outputs/llm_communicate/chemistry_8b_17b_random_to_17b_platour_grpo_smokes'
BASE=ROOT/'outputs/llm_communicate/chemistry_8b_17b_random_to_17b_platour_grpo_zero_linear_smokes'
REFERENCE=ROOT/'outputs/llm_communicate/chemistry_235b_fp8_to_17b_platour_grpo_uniform_endpoints_gate1_g8_bs32_2k_lr1e5_ep3_eval_half_avg8_8gpu'
VARIANTS={'8b':'Qwen3-8B','17b':'Qwen3-1.7B','random':None}


class RightPadding:
    """Right-pad HF forwards; export only contiguous real prefix K/V."""
    def prefill(self,ids,mask,extra=None):
        lengths=mask.long().sum(-1).tolist()
        compact=torch.full_like(ids,self.pad_token_id);compact_mask=torch.zeros_like(mask)
        for i,n in enumerate(lengths):
            compact[i,:n]=ids[i,mask[i].bool()];compact_mask[i,:n]=1
        ids,mask=compact,compact_mask
        values=self.features(ids,mask)
        injection=mask.bool().clone()
        for i,n in enumerate(lengths):injection[i,n-1]=False
        if extra is not None:
            expanded=torch.full((len(ids),ids.shape[1]+extra.shape[1]),self.pad_token_id,device=ids.device,dtype=ids.dtype)
            expanded_mask=torch.zeros_like(expanded)
            for i,n in enumerate(lengths):
                expanded[i,:n]=ids[i,:n];expanded[i,n:n+extra.shape[1]]=extra[i]
                expanded_mask[i,:n+extra.shape[1]]=1
            ids,mask=expanded,expanded_mask;lengths=[n+extra.shape[1] for n in lengths]
        with self.inject(values,injection):
            out=self.student.model(input_ids=ids,attention_mask=mask,
                position_ids=torch.arange(ids.shape[1],device=ids.device)[None].expand_as(ids),use_cache=True)
        last=out.last_hidden_state[torch.arange(len(ids),device=ids.device),torch.tensor(lengths,device=ids.device)-1]
        logits=self.student.lm_head(last)[:,None]
        assert torch.isfinite(logits).all()
        for layer in out.past_key_values.layers:
            for name in ['keys','values']:
                old=getattr(layer,name);aligned=torch.zeros_like(old)
                for i,n in enumerate(lengths):aligned[i,:,-n:]=old[i,:,:n]
                setattr(layer,name,aligned)
            assert torch.isfinite(layer.keys).all() and torch.isfinite(layer.values).all()
        return types.SimpleNamespace(logits=logits,past_key_values=out.past_key_values)

    def response_hidden(self,prompt,response):
        from qwen35_bridge_grpo_core import checkpoint_student_blocks
        values=self.prompt_features(prompt)
        ids=torch.tensor([prompt+response[:-1]],device=self.student.device)
        injection=torch.ones((1,len(prompt)),device=ids.device,dtype=torch.bool);injection[:,-1]=False
        context=checkpoint_student_blocks(self) if torch.is_grad_enabled() and self.student_activation_checkpointing else nullcontext()
        with context,self.inject(values,injection):
            return self.student.model(input_ids=ids,attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(ids.shape[1],device=ids.device)[None],use_cache=False).last_hidden_state[:,len(prompt)-1:]


class DenseBridge(RightPadding,DirectResidualChatBridge):
    def __init__(self,student,teacher,pad):
        super().__init__(student,teacher,variant='teacher_residual',initialization='zero',lm_chunk_tokens=128)
        self.pad_token_id=pad;self._prompt_features=OrderedDict()
        self.native_vllm_first_token=True;self.student_activation_checkpointing=True

    def initialize_for_grpo(self,seed=42):
        with torch.no_grad():
            for p in self.maps.parameters():p.zero_()
        self.initialization='zero'
        return dict(kind='exact-zero linear bridge',seed=seed,constant_gate=1.,
            projection_initialization='exact zero',sft_checkpoint_loaded=False)

    def remember(self,key,features):
        self._prompt_features[key]=features;self._prompt_features.move_to_end(key)
        while len(self._prompt_features)>8:self._prompt_features.popitem(last=False)

    def features(self,ids,mask):
        lengths=mask.long().sum(-1).tolist()
        compact=torch.full_like(ids,self.pad_token_id);right_mask=torch.zeros_like(mask)
        for i,n in enumerate(lengths):compact[i,:n]=ids[i,mask[i].bool()];right_mask[i,:n]=1
        with torch.no_grad():
            out=self.teacher.model(input_ids=compact,attention_mask=right_mask,
                position_ids=torch.arange(ids.shape[1],device=ids.device)[None].expand_as(ids),
                use_cache=False,output_hidden_states=True)
            values=[]
            for h in self.teacher_features(out.hidden_states):
                aligned=torch.zeros(ids.shape[0],ids.shape[1],h.shape[-1],device=ids.device,dtype=h.dtype)
                for i,n in enumerate(lengths):
                    assert torch.isfinite(h[i,:n]).all()
                    aligned[i,mask[i].bool()]=h[i,:n]
                values.append(aligned)
            values=tuple(values)
            for i in range(len(ids)):
                active=mask[i].bool()
                self.remember(tuple(ids[i,active].tolist()),tuple(h[i:i+1,active].clone() for h in values))
        return values

    def prompt_features(self,prompt):
        key=tuple(prompt)
        if key not in self._prompt_features:
            ids=torch.tensor([prompt],device=self.student.device);self.features(ids,torch.ones_like(ids))
        values=self._prompt_features[key];self._prompt_features.move_to_end(key)
        if not torch.is_inference_mode_enabled() and any(h.is_inference() for h in values):
            values=tuple(h.clone() for h in values);self.remember(key,values)
        return values


def classes_for_mask(mask,prefix,suffix):
    lengths=mask.long().sum(-1)[:,None]
    relative=mask.long().cumsum(-1)-1
    count=prefix+suffix-1
    classes=torch.full_like(relative,count)
    classes=torch.where(mask.bool()&(relative<prefix),relative,classes)
    suffix_active=mask.bool()&(relative>=lengths-suffix)&(relative<lengths-1)
    classes=torch.where(suffix_active,prefix+relative-(lengths-suffix),classes)
    return classes


class GaussianBridge(RightPadding,VirtualTeacherBridge):
    def __init__(self,student,stats,prefix,suffix,pad):
        nn.Module.__init__(self)
        self.student=student.eval().requires_grad_(False)
        self.layer_map=list(range(student.config.num_hidden_layers))
        self.prefix,self.suffix=list(prefix),list(suffix)
        self.classes=len(prefix)+len(suffix)-1;self.pad_token_id=pad
        self.register_buffer('template_token_ids',torch.tensor(prefix+suffix[:-1],device=student.device))
        self.register_buffer('initial_mean',stats['mean'].to(student.device))
        self.register_buffer('initial_std',stats['std'].to(student.device))
        assert self.initial_mean.shape==self.initial_std.shape==(self.classes,28,1)
        self.virtual_hidden=nn.Parameter(torch.empty(self.classes,28,2048,device=student.device,dtype=torch.float32))
        self.maps=nn.ModuleList([nn.Linear(2048,2048,bias=False,device=student.device,dtype=torch.float32) for _ in range(28)])
        self.native_vllm_first_token=True;self.student_activation_checkpointing=True

    def initialize_for_grpo(self,seed=42):
        noise=torch.randn(tuple(self.virtual_hidden.shape),generator=torch.Generator().manual_seed(seed))
        with torch.no_grad():
            self.virtual_hidden.copy_(self.initial_mean+self.initial_std*noise.to(self.student.device))
            for projection in self.maps:projection.weight.zero_()
        return dict(kind='500 training-sample template/layer scalar moments -> trainable Gaussian hidden',
            seed=seed,virtual_hidden_shape=list(self.virtual_hidden.shape),projection_bound=0.,
            projection_initialization='exact zero; same P as real 1.7B sender',
            teacher_model_loaded=False,constant_gate=1.,sft_checkpoint_loaded=False)

    def features(self,ids,mask):
        classes=classes_for_mask(mask,len(self.prefix),len(self.suffix))
        active=classes.lt(self.classes)&mask.bool()
        assert torch.equal(ids[active],self.template_token_ids[classes[active]])
        return tuple(classes for _ in self.layer_map)

    def prompt_features(self,prompt):
        ids=torch.tensor([prompt],device=self.student.device)
        return self.features(ids,torch.ones_like(ids))

    def build_stress_prompt(self,source,tokens):
        middle=source[len(self.prefix):-len(self.suffix)]
        width=tokens-len(self.prefix)-len(self.suffix)
        assert middle and width>0
        result=self.prefix+(middle*((width+len(middle)-1)//len(middle)))[:width]+self.suffix
        assert len(result)==tokens
        return result

    def residual(self,index,hidden,classes,injection_mask):
        n=injection_mask.shape[1]
        active=injection_mask.bool()&classes.lt(self.classes)
        projected=self.maps[index](self.virtual_hidden[:,index])
        delta=torch.zeros_like(hidden[:,:n])
        delta[active]=projected[classes[active]].to(hidden.dtype)
        return F.pad(delta,(0,0,0,hidden.shape[1]-n)) if hidden.shape[1]>n else delta

    def metadata(self):
        return dict(algorithm='trainable_template_Gaussian_virtual_teacher',layer_map=self.layer_map,
            layer_mapping='identity',constant_gate=1.,teacher_model_loaded=False,teacher_forward_calls=0,
            virtual_hidden_shape=list(self.virtual_hidden.shape),statistics_samples=500,
            statistics_pooling='questions and hidden channels, separate template position and decoder block input',
            hidden_space='decoder block input before RMSNorm',injection='template positions in prompt[:-1] only',
            question_direct_delta_zero=True,student_frozen=True,matrix_trainable=True,projection_trainable=True,
            projection_dtype='float32',virtual_hidden_dtype='float32',
            trainable_parameters=sum(p.numel() for p in self.bridge_parameters()))

    @torch.no_grad()
    def training_statistics(self):
        deltas=torch.stack([p(self.virtual_hidden[:,i]) for i,p in enumerate(self.maps)])
        return dict(virtual_hidden_norm=float(self.virtual_hidden.norm()),
            virtual_hidden_grad_norm=0. if self.virtual_hidden.grad is None else float(self.virtual_hidden.grad.norm()),
            projection_norm=math.sqrt(sum(float(p.weight.square().sum()) for p in self.maps)),
            projection_grad_norm=math.sqrt(sum(float(p.weight.grad.square().sum()) for p in self.maps if p.weight.grad is not None)),
            effective_delta_norm=float(deltas.norm()),
            effective_delta_mean_layer_rms=float(deltas.square().mean(-1).sqrt().mean()))


def load_bridge(root,p):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from run_qwen3_gsm8k_c2c_batched import configure_attention
    configure_attention()
    options=dict(dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,
        device_map={'':torch.cuda.current_device()})
    student=AutoModelForCausalLM.from_pretrained(p['student_checkpoint'],**options).eval()
    pad=AutoTokenizer.from_pretrained(p['student_checkpoint'],local_files_only=True).pad_token_id
    if p['virtual_teacher']:
        assert sha(root/'gaussian_statistics.pt')==p['gaussian_statistics_sha256']
        stats=torch.load(root/'gaussian_statistics.pt',map_location='cpu',weights_only=True)
        bridge=GaussianBridge(student,stats,p['template_prefix_ids'],p['template_suffix_ids'],pad).eval()
    else:
        teacher=AutoModelForCausalLM.from_pretrained(p['teacher_checkpoint'],**options).eval()
        assert student is not teacher
        bridge=DenseBridge(student,teacher,pad).eval()
    assert bridge.layer_map==p['layer_map']
    assert sum(t.numel() for t in bridge.bridge_parameters())==p['trainable_parameters']
    return bridge,dict(bridge.metadata(),variant=p['variant'],teacher_response_forward=False,
        native_vllm_first_token=True,padding='right padding, valid-only KV export',
        teacher_prompt_feature_cache_size=0 if p['virtual_teacher'] else 8)


def restore(bridge,state):
    bridge.maps.load_state_dict(state['P'])
    if hasattr(bridge,'virtual_hidden'):
        with torch.no_grad():bridge.virtual_hidden.copy_(state['virtual_hidden'])


def smoke_audit(bridge,engine,rows,directory,pad):
    from qwen35_bridge_grpo_core import sampled_log_probs
    from qwen3_4b_06b_grpo import generate_native_first
    directory.mkdir(parents=True,exist_ok=True)
    before=weight_digest(bridge)
    state=bridge.bridge_checkpoint_state() if hasattr(bridge,'bridge_checkpoint_state') else {'P':{k:v.cpu().clone() for k,v in bridge.maps.state_dict().items()}}
    rng,cuda_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
    params=bridge.bridge_parameters();history=[]
    try:
        row=min(rows,key=lambda r:len(r['prompt_token_ids']))
        ids=torch.tensor([row['prompt_token_ids']],device=bridge.student.device)
        with torch.no_grad():
            for p in bridge.maps.parameters():p.zero_()
            native=bridge.student(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True,logits_to_keep=1).logits
            bridged=bridge.prefill(ids,torch.ones_like(ids)).logits
            zero_error=float((native-bridged).abs().max())
            assert zero_error==0.,zero_error
        restore(bridge,state)
        optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=0.,betas=(.9,.999))
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss=-sampled_log_probs(bridge,row['prompt_token_ids'],[16,17,18,19]).mean()
            loss.backward()
            pn=math.sqrt(sum(float(p.grad.square().sum()) for p in bridge.maps.parameters() if p.grad is not None))
            vn=float(bridge.virtual_hidden.grad.norm()) if hasattr(bridge,'virtual_hidden') else None
            assert pn>0 and (vn is None or (vn==0. if step==0 else vn>0))
            for model in [bridge.student]+([bridge.teacher] if hasattr(bridge,'teacher') else []):
                assert all(not p.requires_grad and p.grad is None for p in model.parameters())
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
            history.append(dict(step=step+1,projection_gradient_norm=pn,virtual_hidden_gradient_norm=vn))
        trained=weight_digest(bridge)
        assert trained!=before
        roundtrip=directory/'roundtrip.pt'
        updated=bridge.bridge_checkpoint_state() if hasattr(bridge,'bridge_checkpoint_state') else {'P':{k:v.cpu().clone() for k,v in bridge.maps.state_dict().items()}}
        torch.save(updated,roundtrip)
        with torch.no_grad():next(iter(bridge.maps.parameters())).add_(1.)
        if hasattr(bridge,'virtual_hidden'):
            with torch.no_grad():bridge.virtual_hidden.add_(1.)
        assert weight_digest(bridge)!=trained
        restore(bridge,torch.load(roundtrip,map_location='cpu',weights_only=True,mmap=True))
        assert weight_digest(bridge)==trained
        roundtrip.unlink();del updated,optimizer
        restore(bridge,state)
        with torch.no_grad():
            generator=torch.Generator(device=bridge.student.device).manual_seed(42)
            for p in bridge.maps.parameters():
                p.uniform_(-1e-4/math.sqrt(p.shape[-1]),1e-4/math.sqrt(p.shape[-1]),generator=generator)
                p.mul_(1000.)
        chosen=[min(rows,key=lambda r:len(r['prompt_token_ids'])),max(rows,key=lambda r:len(r['prompt_token_ids'])),rows[0]]
        chosen=[dict(r,sample_index=0,sampling_seed=42+i) for i,r in enumerate(chosen)]
        result,_=generate_native_first(bridge,engine,chosen,weight_digest(bridge),dict(TRAIN_SAMPLING,temperature=0.),
            16,directory/'handoff',pad,return_logprobs=True)
        errors=[];ratios=[]
        with torch.no_grad():
            for row in result['rows']:
                lp=sampled_log_probs(bridge,row['prompt_token_ids'],row['response_token_ids'])
                d=lp-torch.tensor(row['rollout_log_probs'],device=lp.device)
                errors.extend(d.abs().tolist());ratios.extend(d.exp().tolist())
        parity=dict(mean_abs_logprob_error=sum(errors)/len(errors),max_abs_logprob_error=max(errors),
            importance_ratio_min=min(ratios),importance_ratio_max=max(ratios),tokens=len(errors))
        parity['strict_passed']=parity['mean_abs_logprob_error']<.05 and parity['max_abs_logprob_error']<.2
        parity['bf16_backend_accepted']=parity['mean_abs_logprob_error']<.05 and parity['max_abs_logprob_error']<.4 and min(ratios)>.65 and max(ratios)<1.5
        save(directory/'parity.json',parity)
        assert parity['strict_passed'] or parity['bf16_backend_accepted'],parity
        save(directory/'audit.json',dict(passed=True,zero_projection_native_max_logit_error=zero_error,
            gradient_steps=history,frozen_backbones=True,teacher_model_loaded=hasattr(bridge,'teacher'),
            checkpoint_roundtrip_exact=True,all_trainable_tensors_restored_after_audit=True,
            mixed_length_padding_tested=True,nonzero_bridge_probe_scale=1000.,parity=parity))
    finally:
        restore(bridge,state)
        for p in params:p.grad=None
        torch.set_rng_state(rng);torch.cuda.set_rng_state(cuda_rng)
        assert weight_digest(bridge)==before
        torch.cuda.empty_cache()


def common_ends(prompts):
    first=prompts[0]['prompt_token_ids'];prefix=len(first);suffix=len(first)
    for row in prompts:
        ids=row['prompt_token_ids'];n=0
        while n<min(prefix,len(ids)) and first[n]==ids[n]:n+=1
        prefix=n;n=0
        while n<min(suffix,len(ids)) and first[-1-n]==ids[-1-n]:n+=1
        suffix=n
    assert prefix>0 and suffix>1 and all(len(r['prompt_token_ids'])>prefix+suffix for r in prompts)
    return first[:prefix],first[-suffix:]


def snapshot(root):
    names=['chemistry_platour_grpo.py','train_qwen35_bridge_opd_vllm.py','run_qwen35_bridge_opd_vllm.py',
        'qwen35_bridge_grpo_core.py','qwen35_bridge_opd_core.py','qwen35_virtual_teacher_bridge.py',
        'qwen3_4b_06b_grpo.py','prepare_qwen35_bridge_vllm.py','qwen35_bridge_opd_vllm_worker.py',
        'qwen35_hybrid_memory_connector.py','qwen35_bridge_logits_processors.py',
        'chemistry_grpo_score_worker.py','eval_qwen3_17b_chemistry.py','run_qwen3_gsm8k_c2c_batched.py',
        'qwen35_student.py','qwen35_direct_math_bridge.py']
    files=[HERE/n for n in names]+[ROOT/'scripts'/n for n in ['direct_residual_chat_bridge.py','shared_subspace_chat_bridge.py','shared_subspace_residual_bridge.py']]
    save(root/'source_sha256.json',{str(f.relative_to(ROOT)):sha(f) for f in files})
    for f in files:
        dest=root/'source_snapshot'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)


def prepare(root,variants):
    from transformers import AutoConfig,AutoTokenizer
    from hidden_bridge import bridge_layer_map
    root.mkdir(parents=True,exist_ok=True)
    original=json.loads((REFERENCE/'protocol.json').read_text())
    prompts=json.loads((REFERENCE/'train_prompts.json').read_text())
    prefix,suffix=common_ends(prompts)
    test=json.loads((REFERENCE/'eval_prompts.json').read_text())
    for r in test:
        ids=r['prompt_token_ids'];assert ids[:len(prefix)]==prefix and ids[-len(suffix):]==suffix
    selected=torch.randperm(1890,generator=torch.Generator().manual_seed(42)).tolist()[:500]
    assert len(set(selected))==500 and not {prompts[i]['id'] for i in selected}&{r['id'] for r in test}
    for variant in variants:
        job=root/variant
        assert not (job/'protocol.json').exists(),'Use a fresh suite root'
        job.mkdir(exist_ok=True)
        for split in ['train','eval']:
            source=REFERENCE/f'{split}_prompts.json'
            assert sha(source)==original[f'{split}_prompts_sha256'];shutil.copyfile(source,job/source.name)
        p=dict(original)
        for key in list(p):
            if key.startswith(('teacher_','fp8_')) or key in ['chemistry_reference_root','chemistry_reference_protocol_sha256']:
                p.pop(key)
        sender=VARIANTS[variant]
        config=AutoConfig.from_pretrained(ROOT/'data/models'/(sender or 'Qwen3-1.7B'),local_files_only=True)
        random=variant=='random';width=config.hidden_size
        p.update(bridge_recipe='chemistry_dense_gaussian_platour_grpo',variant=variant,
            teacher=sender or 'trainable_gaussian_virtual_teacher',teacher_checkpoint=str(ROOT/'data/models'/sender) if sender else None,
            teacher_family='qwen3',teacher_weight_precision='bf16' if sender else None,
            teacher_model_parallel_size=1,teacher_model_loaded=not random,virtual_teacher=random,
            teacher_response_forward=False,teacher_hidden_size=width,
            world_size=8,gpu_ids=list(range(8)),decoder_gpu_ids=list(range(8)),vllm_replicas=8,
            layer_map=bridge_layer_map(28,config.num_hidden_layers,'uniform_endpoints'),
            layer_mapping='uniform_endpoints' if sender else 'identity_template_virtual',
            mapping_rule=f'round(i*{config.num_hidden_layers-1}/27), zero-based block input',
            trainable_parameters=28*2048*width+((len(prefix)+len(suffix)-1)*28*2048 if random else 0),
            initialization='exact-zero linear bridge; no SFT or GRPO checkpoint',projection_initialization='exact zero',
            training_scope='SMOKE ONLY by default; formal train stage must be explicitly invoked',
            wandb_project='chemistry-platour-grpo',wandb_run_name=f'chemistry_{variant}_to_17b_platour_grpo',
            reference_235b=str(REFERENCE),reference_235b_protocol_sha256=sha(REFERENCE/'protocol.json'))
        if random:
            p.update(template_prefix_ids=prefix,template_suffix_ids=suffix,
                virtual_hidden_shape=[len(prefix)+len(suffix)-1,28,2048],gaussian_statistics_sha256=None,
                gaussian_samples=selected,gaussian_pooling='training questions and hidden channels per template position and layer',
                matrix_trainable=True,projection_trainable=True,injection='template positions in prompt[:-1] only',
                gaussian_statistics_scope='500 original Chemistry training prompts; no response tokens; test excluded')
        else:
            tokenizer=AutoTokenizer.from_pretrained(p['teacher_checkpoint'],local_files_only=True)
            for r in prompts+test:
                assert tokenizer.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True,
                    enable_thinking=False,return_dict=False)==r['prompt_token_ids']
        assert all(p[k]==original[k] for k in ['batch_size','group_size','epochs','total_steps','learning_rate',
            'max_response_length','train_sampling','eval_sampling','eval_samples','eval_steps',
            'train_prompts_sha256','eval_prompts_sha256'])
        save(job/'protocol.json',p);snapshot(job)
        if random and (PREVIOUS_SMOKES/'random/statistics_complete.json').exists():
            previous=json.loads((PREVIOUS_SMOKES/'random/protocol.json').read_text())
            stats=PREVIOUS_SMOKES/'random/gaussian_statistics.pt'
            assert all(p[k]==previous[k] for k in ['gaussian_samples','template_prefix_ids','template_suffix_ids','train_prompts_sha256'])
            assert sha(stats)==previous['gaussian_statistics_sha256']
            shutil.copyfile(stats,job/'gaussian_statistics.pt')
            shutil.copyfile(PREVIOUS_SMOKES/'random/statistics_complete.json',job/'statistics_complete.json')
            p['gaussian_statistics_sha256']=sha(job/'gaussian_statistics.pt');save(job/'protocol.json',p)
            save(job/'statistics_reuse.json',dict(source=str(stats),same_500_training_samples=True,
                exact_statistics_copy=True,gaussian_initialization_unchanged=True,only_projection_initialization_changed=True))
    save(root/'plan.json',dict(variants=list(variants),execution_scope='SMOKE ONLY',
        random_statistics_samples=500,random_statistics_source='training only',
        template_prefix_tokens=len(prefix),template_suffix_tokens=len(suffix),
        gaussian_hidden_shape=[len(prefix)+len(suffix)-1,28,2048],automatically_start_training=False))


def statistics(job):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from run_qwen3_gsm8k_c2c_batched import configure_attention
    configure_attention();torch.set_num_threads(4)
    p=json.loads((job/'protocol.json').read_text());rows=json.loads((job/'train_prompts.json').read_text())
    assert p['virtual_teacher'] and len(set(p['gaussian_samples']))==500
    model=AutoModelForCausalLM.from_pretrained(p['student_checkpoint'],dtype=torch.bfloat16,
        attn_implementation='sdpa',device_map={'':0},local_files_only=True).eval().requires_grad_(False)
    pad=AutoTokenizer.from_pretrained(p['student_checkpoint'],local_files_only=True).pad_token_id
    count=len(p['template_prefix_ids'])+len(p['template_suffix_ids'])-1
    sums=torch.zeros(count,28,device='cuda',dtype=torch.float64);squares=torch.zeros_like(sums)
    with torch.inference_mode():
        for offset in range(0,500,8):
            group=[rows[i] for i in p['gaussian_samples'][offset:offset+8]]
            lengths=[len(r['prompt_token_ids']) for r in group];width=max(lengths)
            ids=torch.full((len(group),width),pad,device='cuda',dtype=torch.long);mask=torch.zeros_like(ids)
            for i,row in enumerate(group):ids[i,:lengths[i]]=torch.tensor(row['prompt_token_ids'],device='cuda');mask[i,:lengths[i]]=1
            hidden=model.model(input_ids=ids,attention_mask=mask,position_ids=torch.arange(width,device='cuda')[None].expand_as(ids),
                use_cache=False,output_hidden_states=True).hidden_states
            classes=classes_for_mask(mask,len(p['template_prefix_ids']),len(p['template_suffix_ids']))
            for layer in range(28):
                values=torch.stack([hidden[layer][i,classes[i]<count] for i in range(len(group))]).double()
                assert values.shape==(len(group),count,2048) and torch.isfinite(values).all()
                sums[:,layer]+=values.sum((0,2));squares[:,layer]+=values.square().sum((0,2))
            save(job/'statistics_live.json',dict(completed=min(offset+8,500),total=500))
    mean=sums/(500*2048);std=(squares/(500*2048)-mean.square()).clamp_min(0).sqrt()
    target=job/'gaussian_statistics.pt'
    state=dict(mean=mean.float().cpu().unsqueeze(-1),std=std.float().cpu().unsqueeze(-1),
        samples=p['gaussian_samples'],sample_ids=[rows[i]['id'] for i in p['gaussian_samples']],training_only=True,
        hidden_space='decoder block input before RMSNorm',pooling=p['gaussian_pooling'],
        template_token_ids=p['template_prefix_ids']+p['template_suffix_ids'][:-1])
    assert torch.isfinite(state['mean']).all() and torch.isfinite(state['std']).all()
    torch.save(state,target);p['gaussian_statistics_sha256']=sha(target);save(job/'protocol.json',p)
    save(job/'statistics_complete.json',dict(complete=True,samples=500,training_only=True,
        shape=list(state['mean'].shape),sha256=sha(target),test_questions_used=0))


def runtime_environment():
    for key in ['PLATOUR_BRIDGE_PYTHON','PLATOUR_GENERATION_PYTHON','PLATOUR_SCORING_PYTHON']:
        os.environ[key]=str(ROOT/'.runtime/math500/bin/python')
    os.environ.update(WANDB_MODE='offline',WANDB_DISABLE_CODE='true',VLLM_NO_USAGE_STATS='1',
        OMP_NUM_THREADS='4',VLLM_WORKER_MULTIPROC_METHOD='spawn',DO_NOT_TRACK='1')
    os.environ.pop('WANDB_DISABLED',None)


def suite(root,variants):
    from run_qwen35_bridge_opd_vllm import queue
    runtime_environment()
    lock=(root/'suite.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if 'random' in variants and not (root/'random/statistics_complete.json').exists():
        save(root/'phase.json',dict(phase='statistics',variant='random',time=time.time()))
        with (root/'statistics.log').open('a') as log:
            subprocess.run([sys.executable,'-u',str(SCRIPT),'statistics','--root',str(root)],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True)
    results={}
    for variant in variants:
        job=root/variant
        save(root/'phase.json',dict(phase='smoke',variant=variant,time=time.time()))
        if not (job/'smoke_audit.json').exists():queue(job,'grpo',smoke_only=True)
        done=json.loads((job/'smoke/complete.json').read_text())
        audit=json.loads((job/'smoke/platour_audit/audit.json').read_text())
        assert done['complete'] and done['replica_exact_weights'] and done['singleton_tail_verified'] and audit['passed']
        assert not (job/'train').exists(),'Smoke must not create a formal training directory'
        results[variant]=dict(complete=True,steps=done['steps'],world_size=done['world_size'],
            nonzero_gradient_updates=done['nonzero_gradient_updates'],singleton_tail_verified=True,
            audit=audit,formal_training_started=False)
        save(root/'results.json',results)
    if '17b' in variants and 'random' in variants:
        a=torch.load(root/'17b/smoke/initial/bridge.pt',map_location='cpu',weights_only=True,mmap=True)['P']
        b=torch.load(root/'random/smoke/initial/bridge.pt',map_location='cpu',weights_only=True,mmap=True)['P']
        assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
        save(root/'matched_initial_projection_audit.json',dict(passed=True,
            dense_17b_and_random_projections_bitwise_identical=True,gaussian_changes_teacher_only=True))
    save(root/'phase.json',dict(phase='complete',smoke_only=True,variants=list(variants),time=time.time()))


def formal(root,variant,destination):
    """Explicit future entry: fresh training from the audited initialization."""
    from run_qwen35_bridge_opd_vllm import queue
    source=root/variant;destination=destination.resolve()
    assert not destination.exists() and (source/'smoke_audit.json').exists()
    destination.mkdir(parents=True)
    for name in ['protocol.json','train_prompts.json','eval_prompts.json']:
        shutil.copyfile(source/name,destination/name)
    if variant=='random':shutil.copyfile(source/'gaussian_statistics.pt',destination/'gaussian_statistics.pt')
    p=json.loads((destination/'protocol.json').read_text())
    p.update(reuse_completed_smoke_root=str(source.resolve()),training_scope='Formal 3-epoch GRPO from fresh audited initialization',
        wandb_run_name=destination.name)
    save(destination/'protocol.json',p);snapshot(destination);runtime_environment()
    queue(destination,'grpo')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','start','queue','statistics','train'])
    parser.add_argument('--root',type=Path,default=BASE)
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    parser.add_argument('--variant',choices=VARIANTS)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=args.root.resolve();runtime_environment()
    if args.stage=='statistics':statistics(root/'random');return
    if args.stage=='queue':suite(root,args.variants);return
    if args.stage=='train':
        assert args.variant and args.output,'Explicit variant and new --output are required for formal training'
        formal(root,args.variant,args.output);return
    if not (root/'plan.json').exists():prepare(root,args.variants)
    else:assert json.loads((root/'plan.json').read_text())['variants']==args.variants
    if args.stage=='prepare':print(root);return
    cmd=[sys.executable,'-u',str(SCRIPT),'queue','--root',str(root),'--variants',*args.variants]
    with (root/'controller.log').open('a') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    save(root/'controller.json',dict(pid=child.pid,command=cmd,time=time.time(),smoke_only=True))
    print(json.dumps(dict(pid=child.pid,root=str(root),smoke_only=True)))


if __name__=='__main__':main()
