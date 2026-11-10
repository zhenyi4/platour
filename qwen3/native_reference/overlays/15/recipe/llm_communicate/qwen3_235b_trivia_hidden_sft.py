"""Qwen3-235B-A22B-FP8 -> Qwen3-1.7B TriviaQA hidden bridge SFT."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(HERE)]
from trivia_bridge_common import INSTRUCTION,normalize_answer,score
from generate_qwen35_train_suite import save,jsonl,read

BASE=ROOT/'outputs/llm_communicate/triviaqa_235b_fp8_to_17b_hidden_short10k_sft_ep3_bs64'
TEACHER=ROOT/'data/models/Qwen3-235B-A22B-FP8'
STUDENT=ROOT/'data/models/Qwen3-1.7B'
REFERENCE=HERE/'assets/qwen3_native_eval/triviaqa_answer_sentence_short10k_sft_ep3_bs64_greedy.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(base):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer
    from hidden_bridge import bridge_layer_map
    from nemotron_chat_data import INDEX_DTYPE
    base.mkdir(parents=True,exist_ok=True)
    if (base/'protocol.json').exists():
        p=json.loads((base/'protocol.json').read_text())
        assert sha(Path(p['data_root'])/'split.json')==p['split_sha256']
        assert sha(Path(p['data_root'])/'train.jsonl')==p['source_sha256']
        return p
    artifact=json.loads(REFERENCE.read_text());original=artifact['jobs']['4b_to_17b_hidden']['protocol']
    native=AutoTokenizer.from_pretrained(STUDENT,local_files_only=True)
    teacher=AutoTokenizer.from_pretrained(TEACHER,local_files_only=True)
    assert native.get_vocab()==teacher.get_vocab() and native.chat_template==teacher.chat_template
    raw={split:pq.read_table(ROOT/'data/triviaqa'/f'{split}.parquet').to_pylist() for split in ('train','validation')}
    assert len(raw['train'])==87622 and len(raw['validation'])==11313
    assert not {r['question_id'] for r in raw['train']}&{r['question_id'] for r in raw['validation']}
    def target(answer):
        candidates=[answer['value'],*answer['aliases']];accepted=set(answer['normalized_aliases'])
        valid=[v for v in candidates if 1<=len(v.split())<=5 and normalize_answer(v) in accepted and not any(c in v for c in '{}\n\r')]
        if not valid:return None
        return valid[0] if valid[0]==answer['value'] else min(valid,key=lambda x:(len(x.split()),len(x),x))
    selected=artifact['selection']['source_indices']
    assert len(selected)==len(set(selected))==10000
    def record(row,index,source_index,answer=None):
        text=row['question']+'\n\n'+INSTRUCTION
        prompt=native.apply_chat_template([dict(role='user',content=text)],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        other=teacher.apply_chat_template([dict(role='user',content=text)],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        assert prompt==other and len(prompt)<=1024
        gold=row['answer']['value'] if answer is None else answer
        response='The answer is \\boxed{'+gold+'}'
        ids=native(response+'<|im_end|>',add_special_tokens=False)['input_ids']
        assert ids[-1]==151645 and len(prompt)+len(ids)<=3072
        r=dict(index=index,id=row['question_id'],source_index=source_index,question=row['question'],gold=gold,
            aliases=row['answer']['aliases'],normalized_aliases=row['answer']['normalized_aliases'],
            input_text=text,prompt_token_ids=prompt,response=response,model_response=response,
            response_token_ids=ids,tokens=len(ids),acc=1,finish_reason='stop')
        if answer is not None:assert score(r)['acc']==1
        return r
    training=[]
    for i,source in enumerate(selected):
        gold=target(raw['train'][source]['answer']);assert gold is not None
        training.append(record(raw['train'][source],i,source,gold))
    validation=[record(r,i,i) for i,r in enumerate(raw['validation'])]
    data=base/'data';data.mkdir(exist_ok=True)
    jsonl(data/'train.jsonl',training);jsonl(data/'validation.jsonl',validation)
    order=torch.randperm(10000,generator=torch.Generator().manual_seed(42)).tolist()
    split=dict(train=order[:9900],validation=order[9900:]);save(data/'split.json',split)
    assert sha(data/'split.json')==original['split_sha256'],'Reference split differs'
    assert sha(data/'train.jsonl')==original['source_sha256'],'Reference training targets or tokens differ'
    offset=0;index=[];heldout=set(split['validation'])
    with (data/'tokens.bin').open('wb') as binary:
        for r in training:
            ids=r['prompt_token_ids']+r['response_token_ids'];np.asarray(ids,dtype=np.uint32).tofile(binary)
            index.append((offset,len(r['prompt_token_ids']),len(ids),0,int(r['index'] in heldout),r['index']));offset+=len(ids)
    np.asarray(index,dtype=INDEX_DTYPE).tofile(data/'index.bin')
    save(data/'complete.json',dict(records=10000,source_indices=selected,
        maximum_prompt_tokens=max(len(r['prompt_token_ids']) for r in training),
        maximum_response_tokens=max(r['tokens'] for r in training),exact_reference_data=True))
    sc=json.loads((STUDENT/'config.json').read_text());tc=json.loads((TEACHER/'config.json').read_text())
    assert (sc['num_hidden_layers'],sc['hidden_size'])==(28,2048)
    assert (tc['num_hidden_layers'],tc['hidden_size'])==(94,4096)
    p=dict(original)
    p.pop('hardware_batch_tuning',None)
    p.update(sender='Qwen3-235B-A22B-FP8',teacher_checkpoint=str(TEACHER),student_checkpoint=str(STUDENT),
        teacher_revision='39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35',
        student_revision=json.loads((ROOT/'outputs/model_downloads/qwen3_17b/manifest.json').read_text())['revision'],
        teacher_family='qwen3_235b_fp8',teacher_quantization=tc['quantization_config'],
        teacher_weight_precision='fp8',base_dtype='bfloat16',projection_dtype='float32',
        layer_map=bridge_layer_map(28,94,'uniform_endpoints'),mapping_rule='round(i*93/27), zero-based decoder block input',
        trainable_parameters=28*2048*4096,teacher_gpu_ids=[1,2],student_gpu_id=0,gpu_ids=[0,1,2],
        world_size=1,teacher_model_parallel_size=2,teacher_layer_split=[47,47],
        micro_batch=16,gradient_accumulation_steps=4,data_root=str(data),
        split_sha256=sha(data/'split.json'),source_sha256=sha(data/'train.jsonl'),
        reference_artifact=str(REFERENCE),reference_artifact_sha256=sha(REFERENCE),
        fp8_moe_backend='vLLM fused_experts Triton; frozen FP8 weights and dynamic block activation quantization',
        tracking='local JSON logs only; W&B disabled',native_baseline_root=None)
    save(base/'protocol.json',p)
    files=[Path(__file__),HERE/'train_arc_hidden_sft.py',HERE/'eval_arc_bridge_sft.py',HERE/'arc_bridge_sft_common.py',
        HERE/'arc_bridge_vllm_worker.py',HERE/'trivia_bridge_common.py',ROOT/'scripts/direct_residual_chat_bridge.py',
        ROOT/'scripts/shared_subspace_chat_bridge.py',ROOT/'scripts/shared_subspace_residual_bridge.py',ROOT/'scripts/nemotron_chat_data.py']
    save(base/'source_sha256.json',{str(f.relative_to(ROOT)):sha(f) for f in files})
    for f in files:
        dest=base/'source_snapshot'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
    print(json.dumps(dict(prepared=True,root=str(base),trainable_parameters=p['trainable_parameters'],exact_reference_data=True)),flush=True)
    return p


def accelerate_fp8_experts(teacher):
    import types
    import torch
    from transformers.integrations.finegrained_fp8 import FP8Expert
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    def forward(module,hidden_states,top_k_index,top_k_weights):
        assert not torch.is_grad_enabled(),'Frozen FP8 teacher is inference only'
        with torch.cuda.device(hidden_states.device):
            return fused_experts(hidden_states.contiguous(),module.gate_up_proj,module.down_proj,
                top_k_weights.float().contiguous(),top_k_index.int().contiguous(),
                quant_config=fp8_w8a8_moe_quant_config(module.gate_up_proj_scale_inv,
                    module.down_proj_scale_inv,block_shape=module.block_size))
    count=0
    for module in teacher.modules():
        if isinstance(module,FP8Expert):
            assert module.gate_up_proj.dtype==module.down_proj.dtype==torch.float8_e4m3fn
            module.forward=types.MethodType(forward,module);count+=1
    assert count==94,count
    return count


def load_bridge(base,checkpoint=None,devices=None,student_device=None,backbones_only=False):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from transformers.integrations import finegrained_fp8 as fp8
    from direct_residual_chat_bridge import DirectResidualChatBridge
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
    from torch.nn import functional as F
    p=json.loads((Path(base)/'protocol.json').read_text())
    if devices is None:
        if p.get('distributed_sft') and 'LOCAL_RANK' in os.environ:
            primary=2*int(os.environ['LOCAL_RANK'])
            devices=[primary,primary+1]
            student_device=primary
        else:
            devices=p['teacher_gpu_ids']
    if student_device is None:student_device=p.get('student_gpu_id',0)
    primary,secondary=devices
    # Use the local Triton fallback for attention linear FP8; no unpinned runtime kernel download.
    fp8._quantization_kernel=False
    placement={'model.embed_tokens':primary,'model.rotary_emb':primary,'model.norm':secondary,'lm_head':secondary}
    placement.update({f'model.layers.{i}':primary if i<47 else secondary for i in range(94)})
    options=dict(dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
    student=AutoModelForCausalLM.from_pretrained(p['student_checkpoint'],device_map={'':student_device},**options).eval()
    teacher=AutoModelForCausalLM.from_pretrained(p['teacher_checkpoint'],device_map=placement,**options).requires_grad_(False).eval()
    scales=0
    for name,param in teacher.named_parameters():
        if name.endswith('_scale_inv'):param.data=param.data.float();scales+=1
    assert scales>0
    from transformers.integrations.finegrained_fp8 import FP8Expert
    modules=[m for m in teacher.modules() if isinstance(m,FP8Expert)]
    references=[]
    with torch.no_grad():
        for module in (modules[0],modules[-1]):
            device=module.gate_up_proj.device
            x=torch.randn(4,4096,device=device,dtype=torch.bfloat16)
            ids=torch.arange(8,device=device)[None].expand(4,-1).contiguous()
            weights=torch.full((4,8),1/8,device=device,dtype=torch.float32)
            references.append((module,x,ids,weights,module(x,ids,weights)))
    count=accelerate_fp8_experts(teacher)
    parity=[]
    with torch.no_grad():
        for module,x,ids,weights,reference in references:
            actual=module(x,ids,weights)
            relative=float((actual.float()-reference.float()).norm()/reference.float().norm())
            assert torch.isfinite(actual).all() and relative<.03,(str(x.device),relative)
            parity.append(dict(device=str(x.device),relative_l2=relative,passed=True,
                input_dtype=str(x.dtype),output_dtype=str(actual.dtype),weight_dtype=str(module.gate_up_proj.dtype)))
    del references,modules
    if backbones_only:
        runtime_name=f'teacher_runtime.rank-{int(os.environ["RANK"]):02d}.json' if 'RANK' in os.environ else 'teacher_runtime.json'
        save(Path(base)/runtime_name,dict(fp8_expert_modules=count,fp8_scale_parameters=scales,
            teacher_placement=placement,student_device=student_device,teacher_weights_kept_fp8=True,fp8_kernel_parity=parity))
        return student,teacher
    class Bridge(DirectResidualChatBridge):
        def features(self,ids,mask,positions=None):
            with torch.no_grad():
                kwargs=dict(input_ids=ids.to(primary),attention_mask=mask.to(primary),use_cache=False,output_hidden_states=True)
                if positions is not None:kwargs['position_ids']=positions.to(primary)
                hidden=self.teacher.model(**kwargs).hidden_states
                return tuple(h.detach().to(student_device).contiguous().clone() for h in self.teacher_features(hidden))

        def chat_loss(self,batch,evaluate_lm=False):
            ids,mask=batch['prefix_ids'],batch['prefix_mask']
            features=self.features(ids,mask.long())
            injection=mask.clone();last=mask.long().sum(1)-1
            injection[torch.arange(len(mask),device=mask.device),last]=False
            with self.inject(features,injection):
                hidden=self.student.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).last_hidden_state
            labels=batch['labels'][:,1:];active=labels!=-100
            predictors=hidden[:,:-1][active];targets=labels[active]
            def chunk(h,y):return F.cross_entropy(self.student.lm_head(h).float(),y,reduction='sum')
            pieces=[]
            for offset in range(0,len(targets),self.lm_chunk_tokens):
                h=predictors[offset:offset+self.lm_chunk_tokens];y=targets[offset:offset+self.lm_chunk_tokens]
                pieces.append(activation_checkpoint(chunk,h,y,use_reentrant=False) if torch.is_grad_enabled() else chunk(h,y))
            loss=torch.stack(pieces).sum()/len(targets)
            return loss.new_zeros(()),loss,dict(lm_nll=loss,weighted_lm_loss=loss)

        def prefill_rows(self,rows,tokenizer):
            width=max(len(r['prompt_token_ids']) for r in rows)
            ids=torch.full((len(rows),width),self.pad_token_id,device=student_device,dtype=torch.long)
            mask=torch.zeros_like(ids);injection=torch.zeros((len(rows),width-1),device=student_device,dtype=torch.bool)
            for i,r in enumerate(rows):
                n=len(r['prompt_token_ids']);ids[i,:n]=torch.tensor(r['prompt_token_ids'],device=student_device);mask[i,:n]=1;injection[i,:n-1]=True
            positions=torch.arange(width,device=student_device)[None].expand(len(rows),-1)
            features=self.features(ids[:,:-1],injection.long(),positions[:,:-1])
            with self.inject(features,injection):
                return self.student(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=True,logits_to_keep=1)

        def metadata(self):
            return dict(super().metadata(),teacher_precision='fp8',teacher='Qwen3-235B-A22B-FP8',student='Qwen3-1.7B',constant_gate=1.)
    bridge=Bridge(student,teacher,variant='teacher_residual',initialization='zero',lm_chunk_tokens=128).eval()
    bridge.pad_token_id=AutoTokenizer.from_pretrained(STUDENT,local_files_only=True).pad_token_id
    assert bridge.layer_map==p['layer_map']
    assert sum(param.numel() for param in bridge.bridge_parameters())==p['trainable_parameters']
    if checkpoint:bridge.load_projectors(checkpoint)
    else:assert all(param.count_nonzero()==0 for param in bridge.bridge_parameters())
    runtime_name=f'teacher_runtime.rank-{int(os.environ["RANK"]):02d}.json' if 'RANK' in os.environ else 'teacher_runtime.json'
    save(Path(base)/runtime_name,dict(fp8_expert_modules=count,fp8_scale_parameters=scales,
        teacher_placement=placement,student_device=student_device,teacher_weights_kept_fp8=True,
        fp8_kernel_parity=parity,torch=torch.__version__))
    return bridge


def environment():
    env=os.environ.copy()
    env.update(PYTHONPATH=str(ROOT/'scripts')+os.pathsep+str(HERE),CUDA_VISIBLE_DEVICES='0,1,2',
        ARC_SFT_PYTHON=sys.executable,ARC_SFT_GPU_IDS='0,1,2',WANDB_MODE='disabled',WANDB_DISABLED='true',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',VLLM_NO_USAGE_STATS='1')
    return env


def run(base):
    p=prepare(base)
    import signal
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={int(line.split(',')[0]):int(line.split(',')[1]) for line in usage.splitlines()}
    assert all(used[i]<512 for i in p['gpu_ids']),f'GPUs {p["gpu_ids"]} must be idle'
    with (base/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stages=[('smoke','train_arc_hidden_sft.py',['--smoke']),('train','train_arc_hidden_sft.py',[]),('eval','eval_arc_bridge_sft.py',[])]
        for stage,script,args in stages:
            for name,digest in json.loads((base/'source_sha256.json').read_text()).items():
                assert sha(ROOT/name)==digest,f'Source changed since preparation: {name}'
            if (base/stage/'complete.json').exists() or (stage=='eval' and (base/'eval/summary.json').exists()):continue
            save(base/'phase.json',dict(phase=stage,time=time.time()))
            with (base/f'{stage}.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u',str(HERE/script),'--root',str(base),*args],cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(base/f'{stage}.controller.json',dict(pid=child.pid,time=time.time()))
                try:
                    code=child.wait()
                except BaseException:
                    import signal
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    save(base/'phase.json',dict(phase='interrupted',stage=stage,time=time.time()))
                    raise
            if code:
                save(base/'phase.json',dict(phase='failed',stage=stage,exit_code=code,time=time.time()))
                raise RuntimeError(f'{stage} failed: {base}/{stage}.log')
        save(base/'phase.json',dict(phase='complete',time=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=BASE)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--queue',action='store_true')
    args=parser.parse_args();base=args.root.absolute()
    if args.prepare_only:prepare(base);return
    if args.detach:
        prepare(base)
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--queue','--root',str(base)]
        with (base/'controller.log').open('a') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(base/'controller.json',dict(pid=proc.pid,time=time.time(),command=command));print(json.dumps(dict(pid=proc.pid,root=str(base))));return
    run(base)


if __name__=='__main__':main()
