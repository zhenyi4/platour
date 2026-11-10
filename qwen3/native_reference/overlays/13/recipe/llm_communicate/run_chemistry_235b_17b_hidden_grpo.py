"""Eight GPUs: frozen Qwen3-235B FP8 -> 1.7B, Chemistry PLatour GRPO."""
import argparse
from collections import OrderedDict
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types
from unittest.mock import patch

from qwen35_bridge_opd_core import ROOT,save,weight_digest
from arc_bridge_sft_common import sha

HERE=Path(__file__).resolve().parent
SCRIPT=Path(__file__).resolve()
REFERENCE=ROOT/'outputs/llm_communicate/chemistry_17b_lora_grpo_r128_a128_g8_bs32_2k_lr1e5_ep3_eval_half_avg8_8gpu'
BASE=ROOT/'outputs/llm_communicate/chemistry_235b_fp8_to_17b_platour_grpo_uniform_endpoints_gate1_g8_bs32_2k_lr1e5_ep3_eval_half_avg8_8gpu'


def load_bridge(root,protocol,devices):
    import torch
    from qwen3_235b_trivia_hidden_sft import load_bridge as fp8_loader
    from run_qwen3_gsm8k_c2c_batched import configure_attention
    from qwen35_bridge_grpo_core import checkpoint_student_blocks
    configure_attention()
    with patch('transformers.modeling_utils.caching_allocator_warmup',lambda *a,**k:None):
        bridge=fp8_loader(root,devices=devices,student_device=devices[0])
    bridge.native_vllm_first_token=True
    bridge.student_activation_checkpointing=True
    bridge._prompt_features=OrderedDict()

    def remember(self,key,features):
        self._prompt_features[key]=features;self._prompt_features.move_to_end(key)
        while len(self._prompt_features)>8:self._prompt_features.popitem(last=False)

    def features(self,ids,mask,positions=None):
        lengths=mask.long().sum(-1).tolist()
        # FP8 teacher uses right padding: no query is fully masked. Return
        # only actual prompt features, aligned to the student's input mask.
        teacher_ids=torch.full_like(ids,self.pad_token_id)
        teacher_mask=torch.zeros_like(mask)
        for i,n in enumerate(lengths):
            teacher_ids[i,:n]=ids[i,mask[i].bool()];teacher_mask[i,:n]=1
        teacher_positions=torch.arange(ids.shape[1],device=ids.device)[None].expand_as(ids)
        with torch.no_grad():
            out=self.teacher.model(input_ids=teacher_ids.to(devices[0]),attention_mask=teacher_mask.to(devices[0]),
                position_ids=teacher_positions.to(devices[0]),use_cache=False,output_hidden_states=True)
            values=[]
            for layer,h in enumerate(self.teacher_features(out.hidden_states)):
                aligned=torch.zeros(ids.shape[0],ids.shape[1],h.shape[-1],device=self.student.device,dtype=h.dtype)
                for i,n in enumerate(lengths):
                    valid=h[i,:n].to(self.student.device)
                    assert torch.isfinite(valid).all(),('nonfinite_teacher_feature',layer,i,n)
                    aligned[i,mask[i].bool()]=valid
                values.append(aligned)
            values=tuple(values)
            for i in range(len(ids)):
                active=mask[i].bool()
                key=tuple(ids[i,active].tolist())
                self.remember(key,tuple(h[i:i+1,active].clone() for h in values))
        return values

    def prompt_features(self,prompt):
        key=tuple(prompt)
        if key not in self._prompt_features:
            ids=torch.tensor([prompt],device=self.student.device)
            self.features(ids,torch.ones_like(ids))
        values=self._prompt_features[key];self._prompt_features.move_to_end(key)
        # Rollout prefill runs in inference_mode. Autograd needs ordinary
        # constant tensors when saving teacher features for dL/dP.
        if not torch.is_inference_mode_enabled() and any(h.is_inference() for h in values):
            values=tuple(h.clone() for h in values);self.remember(key,values)
        return values

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
            ids,mask=expanded,expanded_mask
            lengths=[n+extra.shape[1] for n in lengths]
        with self.inject(values,injection):
            out=self.student.model(input_ids=ids,attention_mask=mask,
                position_ids=torch.arange(ids.shape[1],device=ids.device)[None].expand_as(ids),use_cache=True)
        last=out.last_hidden_state[torch.arange(len(ids),device=ids.device),torch.tensor(lengths,device=ids.device)-1]
        logits=self.student.lm_head(last)[:,None]
        assert torch.isfinite(logits).all(),'Nonfinite student prefill logits'
        # Existing KV exporter slices from the right. Repack only the valid
        # K/V after forward; zero fill is discarded before vLLM consumes it.
        for layer in out.past_key_values.layers:
            for name in ['keys','values']:
                old=getattr(layer,name);aligned=torch.zeros_like(old)
                for i,n in enumerate(lengths):aligned[i,:,-n:]=old[i,:,:n]
                setattr(layer,name,aligned)
            assert torch.isfinite(layer.keys).all() and torch.isfinite(layer.values).all(),'Nonfinite student KV'
        return types.SimpleNamespace(logits=logits,past_key_values=out.past_key_values)

    def response_hidden(self,prompt,response):
        values=self.prompt_features(prompt)
        ids=torch.tensor([prompt+response[:-1]],device=self.student.device)
        injection=torch.ones((1,len(prompt)),device=self.student.device,dtype=torch.bool);injection[:,-1]=False
        context=checkpoint_student_blocks(self) if torch.is_grad_enabled() and self.student_activation_checkpointing else nullcontext()
        with context,self.inject(values,injection):
            return self.student.model(input_ids=ids,attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(ids.shape[1],device=ids.device)[None],use_cache=False).last_hidden_state[:,len(prompt)-1:]

    for name,function in [('remember',remember),('features',features),('prompt_features',prompt_features),
                          ('prefill',prefill),('response_hidden',response_hidden)]:
        setattr(bridge,name,types.MethodType(function,bridge))
    assert all(not p.requires_grad for model in [bridge.teacher,bridge.student] for p in model.parameters())
    return bridge,dict(bridge.metadata(),teacher_device_ids=devices,student_device=str(bridge.student.device),
        native_vllm_first_token=True,teacher_prompt_feature_cache_size=8,teacher_response_forward=False,
        student_attention='SDPA math',fp8_teacher_weights=True)


def verify_handoff(bridge,engine,rows,directory,pad):
    import torch
    from qwen3_4b_06b_grpo import generate_native_first
    from qwen35_bridge_grpo_core import sampled_log_probs
    from qwen35_bridge_opd_core import TRAIN_SAMPLING
    directory.mkdir(parents=True,exist_ok=True)
    before=weight_digest(bridge)
    original=[p.detach().cpu().clone() for p in bridge.bridge_parameters()]
    chosen=[min(rows,key=lambda r:len(r['prompt_token_ids'])),max(rows,key=lambda r:len(r['prompt_token_ids'])),rows[0]]
    chosen=[dict(r,sample_index=0,sampling_seed=42+i) for i,r in enumerate(chosen)]
    try:
        with torch.no_grad():
            for p in bridge.bridge_parameters():p.mul_(1000.)
        digest=weight_digest(bridge)
        result,_=generate_native_first(bridge,engine,chosen,digest,dict(TRAIN_SAMPLING,temperature=0.),
            16,directory/'probe',pad,return_logprobs=True)
        errors=[];ratios=[]
        with torch.no_grad():
            for row in result['rows']:
                actual=sampled_log_probs(bridge,row['prompt_token_ids'],row['response_token_ids'])
                delta=actual-torch.tensor(row['rollout_log_probs'],device=actual.device)
                errors.extend(delta.abs().tolist());ratios.extend(delta.exp().tolist())
        report=dict(mean_abs_logprob_error=sum(errors)/len(errors),max_abs_logprob_error=max(errors),
            importance_ratio_min=min(ratios),importance_ratio_max=max(ratios),tokens=len(errors),
            mixed_length_batch=True,padding_removed_from_export=True,nonzero_bridge_probe_scale=1000.,
            original_projection_digest=before,teacher_response_forward=False)
        report['strict_parity_passed']=report['mean_abs_logprob_error']<.05 and report['max_abs_logprob_error']<.2
        report['bf16_backend_accepted']=report['mean_abs_logprob_error']<.05 and report['max_abs_logprob_error']<.4 and min(ratios)>.65 and max(ratios)<1.5
        report['passed']=report['strict_parity_passed'] or report['bf16_backend_accepted']
        save(directory/'parity.json',report)
        assert report['passed'],report
    finally:
        with torch.no_grad():
            for p,value in zip(bridge.bridge_parameters(),original,strict=True):p.copy_(value)
        assert weight_digest(bridge)==before
        torch.cuda.empty_cache()


def prepare(root):
    assert not (root/'protocol.json').exists(),'Use a fresh output directory'
    root.mkdir(parents=True,exist_ok=True)
    p=json.loads((REFERENCE/'protocol.json').read_text())
    for name in ['train','eval']:
        source=REFERENCE/f'{name}_prompts.json'
        assert sha(source)==p[f'{name}_prompts_sha256']
        shutil.copyfile(source,root/source.name)
    from transformers import AutoTokenizer
    teacher=ROOT/'data/models/Qwen3-235B-A22B-FP8'
    tokenizer=AutoTokenizer.from_pretrained(teacher,local_files_only=True)
    for name in ['train','eval']:
        for row in json.loads((root/f'{name}_prompts.json').read_text()):
            assert tokenizer.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,
                enable_thinking=False,return_dict=False)==row['prompt_token_ids']
    from hidden_bridge import bridge_layer_map
    for key in list(p):
        if key.startswith(('lora_','bf16_backend','parity_limits')) or key in ['actor_training_mode','hf_attention_math_only']:
            p.pop(key)
    p.update(bridge_recipe='qwen235_17b_chemistry_hidden_grpo',teacher='Qwen3-235B-A22B-FP8',
        teacher_checkpoint=str(teacher),teacher_family='qwen3_235b_fp8',teacher_model_loaded=True,
        teacher_weight_precision='fp8',teacher_hidden_size=4096,student_hidden_size=2048,
        teacher_model_parallel_size=2,teacher_layer_split=[47,47],
        teacher_gpu_groups=[[0,1],[2,3],[4,5],[6,7]],world_size=4,
        gpu_ids=list(range(8)),decoder_gpu_ids=[0,2,4,6],vllm_replicas=4,
        trainable_parameters=28*2048*4096,layer_map=bridge_layer_map(28,94,'uniform_endpoints'),
        layer_mapping='uniform_endpoints',mapping_rule='round(i*93/27), decoder block input, zero-based',
        constant_gate=1.,frozen_backbones=True,projection_dtype='float32',projection='linear_no_bias',
        initialization='uniform +/-1e-4/sqrt(4096), seed42, fresh bridge and pretrained base models',
        injection='prompt[:-1] only',teacher_response_forward=False,
        teacher_prompt_feature_cache_size=8,student_activation_checkpointing=True,
        prefill_padding='Right padding for teacher and student; repack valid KV for existing exporter; discard all padding before vLLM',
        native_vllm_first_token=True,vllm_worker_script='qwen35_bridge_opd_vllm_worker.py',
        vllm_memory_fraction=.10,fp8_kernel_cache=str(ROOT/'.runtime/qwen235-grpo-hf-cache'),
        training_rollout_backend='Batched HF bridge prompt prefill, deduplicated by question, native vLLM full-KV decode',
        evaluation_backend='Same hidden bridge and native vLLM all response tokens',
        wandb_project='chemistry-platour-grpo',wandb_run_name=root.name,
        chemistry_reference_root=str(REFERENCE),chemistry_reference_protocol_sha256=sha(REFERENCE/'protocol.json'))
    same=['batch_size','group_size','epochs','learning_rate','weight_decay','betas','grad_clip','scheduler',
        'warmup_steps','max_prompt_length','max_response_length','train_sampling','eval_sampling','eval_samples',
        'eval_questions','eval_steps','eval_seed_stride','total_steps','train_prompts_sha256','eval_prompts_sha256']
    old=json.loads((REFERENCE/'protocol.json').read_text())
    assert all(p[k]==old[k] for k in same)
    save(root/'protocol.json',p)
    save(root/'matched_chemistry_config.json',dict(passed=True,matched_fields=same,
        reference=str(REFERENCE),training_data_and_prompt_ids_exactly_identical=True))
    files=[SCRIPT,*[HERE/n for n in ['qwen3_235b_trivia_hidden_sft.py','train_qwen35_bridge_opd_vllm.py',
        'run_qwen35_bridge_opd_vllm.py','qwen35_bridge_opd_core.py','qwen35_bridge_grpo_core.py',
        'qwen3_4b_06b_grpo.py','qwen35_bridge_opd_vllm_worker.py','prepare_qwen35_bridge_vllm.py',
        'qwen35_hybrid_memory_connector.py','chemistry_grpo_score_worker.py','eval_qwen3_17b_chemistry.py',
        'run_qwen3_gsm8k_c2c_batched.py']],ROOT/'scripts/direct_residual_chat_bridge.py']
    save(root/'source_sha256.json',{str(f.relative_to(ROOT)):sha(f) for f in files})
    for f in files:
        dest=root/'source_snapshot'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=BASE)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--queue',action='store_true')
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    for key in ['PLATOUR_BRIDGE_PYTHON','PLATOUR_GENERATION_PYTHON','PLATOUR_SCORING_PYTHON']:
        os.environ[key]=str(ROOT/'.runtime/math500/bin/python')
    os.environ.update(WANDB_MODE='offline',WANDB_DISABLE_CODE='true',VLLM_NO_USAGE_STATS='1',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',DO_NOT_TRACK='1')
    os.environ.pop('WANDB_DISABLED',None)
    if not args.queue:prepare(root)
    if args.prepare_only:print(root);return
    if args.detach:
        cmd=[sys.executable,'-u',str(SCRIPT),'--queue','--root',str(root)]
        with (root/'controller.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        save(root/'controller.json',dict(pid=child.pid,command=cmd,time=time.time()))
        print(json.dumps(dict(pid=child.pid,root=str(root))));return
    from run_qwen35_bridge_opd_vllm import queue
    queue(root,'grpo')


if __name__=='__main__':main()
