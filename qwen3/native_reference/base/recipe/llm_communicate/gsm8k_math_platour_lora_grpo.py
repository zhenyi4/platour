"""GSM8K/MATH zero-bridge PLatour and native LoRA; default is smoke only."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

import torch
from qwen35_bridge_opd_core import ROOT, save, quarter_epoch_schedule
from arc_bridge_sft_common import sha
import chemistry_platour_grpo as dense

HERE=Path(__file__).resolve().parent
SCRIPT=Path(__file__).resolve()
BASE=ROOT/'outputs/llm_communicate/gsm8k_math_17b_grpo_zero_bridge_smokes'
VARIANTS=['8b','17b','random','lora','235b']


def load_bridge(root,p,devices):
    if p['variant']=='235b':
        from run_chemistry_235b_17b_hidden_grpo import load_bridge as loader
        bridge,runtime=loader(root,p,devices)
        bridge.initialize_for_grpo=types.MethodType(dense.DenseBridge.initialize_for_grpo,bridge)
        return bridge,runtime
    bridge,runtime=dense.load_bridge(root,p)
    if p.get('projection_type')=='low_rank':
        from low_rank_platour_grpo import attach
        bridge=attach(bridge,p);runtime.update(bridge.metadata())
    return bridge,runtime


def snapshot(job):
    dense.snapshot(job)
    hashes=json.loads((job/'source_sha256.json').read_text())
    names=['gsm8k_math_platour_lora_grpo.py','run_chemistry_235b_17b_hidden_grpo.py',
        'qwen3_235b_trivia_hidden_sft.py','qwen35_native_actor_grpo.py',
        'qwen35_native_actor_vllm_worker.py','qwen35_bridge_opd_score_worker.py',
        'generate_qwen35_train_suite.py','llama31_math_score_worker.py','llama31_math_grpo.py']
    for name in names:
        f=HERE/name; rel=f.relative_to(ROOT)
        dest=job/'source_snapshot'/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(f,dest);hashes[str(rel)]=sha(f)
    save(job/'source_sha256.json',hashes)


def prepare(root,tasks,variants):
    from transformers import AutoTokenizer, AutoConfig
    from hidden_bridge import bridge_layer_map
    from generate_qwen35_train_suite import SUFFIX
    import pyarrow.parquet as pq
    root.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(ROOT/'data/models/Qwen3-1.7B',local_files_only=True)
    lora_reference=ROOT/'outputs/llm_communicate/chemistry_17b_lora_grpo_r128_a128_g8_bs32_2k_lr1e5_ep3_eval_half_avg8_8gpu/protocol.json'
    lora=json.loads(lora_reference.read_text())
    for task in tasks:
        if task=='gsm8k':
            source=ROOT/'data/gsm8k/main/train-00000-of-00001.parquet'
            eval_source=ROOT/'data/gsm8k/main/test-00000-of-00001.parquet'
            def gsm_rows(path):
                return [dict(index=i,question=r['question'],input_text=r['question']+SUFFIX,
                    gold=r['answer'].rsplit('####',1)[1].strip()) for i,r in enumerate(pq.read_table(path).to_pylist())]
            raw,raw_eval=gsm_rows(source),gsm_rows(eval_source)
        else:
            origin=ROOT/'outputs/llm_communicate/llama31_70b_to_8b_hidden_math_grpo_g8_bs64_4k_lr1e5_ep1_eval5_8gpu_repair_v2'
            source,eval_source=origin/'train_prompts.json',origin/'eval_prompts.json'
            raw=json.loads(source.read_text());raw_eval=json.loads(eval_source.read_text())
        def prompts(rows,split):
            result=[]
            for i,r in enumerate(rows):
                assert r['input_text']==r['question']+SUFFIX
                messages=[dict(role='user',content=r['input_text'])]
                ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
                assert 2<=len(ids)<=2048
                result.append(dict(index=i,id=str(r.get('id',f'{task}-{split}-{i}')),
                    question=r['question'],input_text=r['input_text'],gold=r['gold'],
                    task=task,messages=messages,prompt_token_ids=ids))
            return result
        train,test=prompts(raw,'train'),prompts(raw_eval,'test')
        assert len(train)==(7473 if task=='gsm8k' else 7500)
        assert len(test)==(1319 if task=='gsm8k' else 500)
        assert not {r['question'] for r in train}&{r['question'] for r in test}
        prefix,suffix=dense.common_ends(train)
        for r in test:
            ids=r['prompt_token_ids'];assert ids[:len(prefix)]==prefix and ids[-len(suffix):]==suffix
        selected=torch.randperm(len(train),generator=torch.Generator().manual_seed(42)).tolist()[:500]
        cap=2048 if task=='gsm8k' else 4096
        steps,eval_steps=quarter_epoch_schedule(len(train),64,3,2)
        for variant in variants:
            job=root/task/variant;assert not (job/'protocol.json').exists(),'Use a fresh root'
            job.mkdir(parents=True,exist_ok=True)
            save(job/'train_prompts.json',train);save(job/'eval_prompts.json',test)
            p=dict(lora)
            p.update(dataset=task,variant=variant,epochs=3,examples_per_epoch=len(train),
                batch_size=64,responses_per_full_batch=512,ppo_mini_batch_size_prompts=64,
                ppo_mini_batch_size_responses=512,steps_per_epoch=steps,total_steps=steps*3,
                max_response_length=cap,eval_samples=2 if task=='gsm8k' else 4,
                eval_seed_stride=2 if task=='gsm8k' else 4,eval_questions=len(test),
                eval_steps=eval_steps,checkpoint_steps=eval_steps,vllm_max_model_len=2048+cap,
                score_worker_script='llama31_math_score_worker.py',math_verify_boxed_only=True,
                reward='Boxed math_verify gold accuracy; missing-gold groups have zero advantage',
                train_prompts_sha256=sha(job/'train_prompts.json'),eval_prompts_sha256=sha(job/'eval_prompts.json'),
                source_train_sha256=sha(source),source_test_sha256=sha(eval_source),
                wandb_project=f'{task}-grpo',wandb_run_name=f'{task}_{variant}_to_17b_grpo',
                training_scope='SMOKE ONLY; formal training requires explicit train command')
            if variant!='lora':
                for key in list(p):
                    if key.startswith(('lora_','bf16_backend','parity_limits')) or key in ['actor_training_mode','hf_attention_math_only']:
                        p.pop(key)
                sender={'8b':'Qwen3-8B','17b':'Qwen3-1.7B','235b':'Qwen3-235B-A22B-FP8','random':None}[variant]
                config=AutoConfig.from_pretrained(ROOT/'data/models'/(sender or 'Qwen3-1.7B'),local_files_only=True)
                random=variant=='random';width=config.hidden_size
                p.update(bridge_recipe='gsm8k_math_platour_grpo',virtual_teacher=random,
                    teacher=sender or 'trainable_gaussian_virtual_teacher',
                    teacher_checkpoint=str(ROOT/'data/models'/sender) if sender else None,
                    teacher_family='qwen3_235b_fp8' if variant=='235b' else 'qwen3',
                    teacher_model_loaded=not random,teacher_hidden_size=width,student_hidden_size=2048,
                    teacher_weight_precision='fp8' if variant=='235b' else 'bf16',
                    layer_map=bridge_layer_map(28,config.num_hidden_layers,'uniform_endpoints'),
                    layer_mapping='identity_template_virtual' if random else 'uniform_endpoints',
                    trainable_parameters=28*2048*width+((len(prefix)+len(suffix)-1)*28*2048 if random else 0),
                    initialization='exact-zero linear bridge; fresh base; no SFT or GRPO checkpoint',
                    projection_initialization='exact zero',constant_gate=1.,projection_dtype='float32',
                    student_activation_checkpointing=True,native_vllm_first_token=True,
                    teacher_response_forward=False,vllm_worker_script='qwen35_bridge_opd_vllm_worker.py')
                if random:
                    p.update(template_prefix_ids=prefix,template_suffix_ids=suffix,
                        virtual_hidden_shape=[len(prefix)+len(suffix)-1,28,2048],gaussian_statistics_sha256=None,
                        gaussian_samples=selected,gaussian_pooling='training questions and hidden channels per template position and layer',
                        gaussian_statistics_scope=f'500 {task} training prompts only; no responses; no test',
                        matrix_trainable=True,projection_trainable=True)
                if variant=='235b':
                    p.update(teacher_model_parallel_size=2,teacher_layer_split=[47,47],
                        teacher_gpu_groups=[[0,1],[2,3],[4,5],[6,7]],world_size=4,
                        decoder_gpu_ids=[0,2,4,6],vllm_replicas=4,vllm_memory_fraction=.10,
                        fp8_kernel_cache=str(ROOT/'.runtime/qwen235-grpo-hf-cache'))
                if sender:
                    teacher_tok=AutoTokenizer.from_pretrained(p['teacher_checkpoint'],local_files_only=True)
                    assert teacher_tok.get_vocab()==tok.get_vocab() and teacher_tok.chat_template==tok.chat_template
            save(job/'protocol.json',p);snapshot(job)
    save(root/'plan.json',dict(tasks=tasks,variants=variants,smoke_only=True,formal_training_started=False))


def queue(root,tasks,variants):
    from run_qwen35_bridge_opd_vllm import queue as run
    dense.runtime_environment()
    lock=(root/'suite.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    results={}
    for task in tasks:
        for variant in variants:
            job=root/task/variant
            save(root/'phase.json',dict(phase='statistics' if variant=='random' else 'smoke',task=task,variant=variant,time=time.time()))
            if variant=='random' and not (job/'statistics_complete.json').exists():
                with (job/'statistics.log').open('a') as log:
                    subprocess.run([sys.executable,'-u',str(SCRIPT),'statistics','--job',str(job)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            save(root/'phase.json',dict(phase='smoke',task=task,variant=variant,time=time.time()))
            if not (job/'smoke_audit.json').exists():run(job,'grpo',smoke_only=True)
            done=json.loads((job/'smoke/complete.json').read_text())
            assert done['complete'] and done['replica_exact_weights'] and done['singleton_tail_verified']
            if variant!='lora':
                audit=json.loads((job/'smoke/platour_audit/audit.json').read_text());assert audit['passed']
                initial=torch.load(job/'smoke/initial/bridge.pt',map_location='cpu',weights_only=True,mmap=True)
                assert all(torch.count_nonzero(v)==0 for v in initial['P'].values())
            else:audit=json.loads((job/'smoke/native_actor_audit/audit.json').read_text())
            assert not (job/'train').exists()
            results[f'{task}/{variant}']=dict(passed=True,complete=done,audit=audit,formal_training_started=False)
            save(root/'results.json',results)
    save(root/'phase.json',dict(phase='complete',smoke_only=True,runs=len(results),time=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','start','queue','statistics','train'])
    parser.add_argument('--root',type=Path,default=BASE)
    parser.add_argument('--tasks',nargs='+',choices=['gsm8k','math'],default=['gsm8k','math'])
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS)
    parser.add_argument('--job',type=Path)
    parser.add_argument('--output',type=Path)
    a=parser.parse_args();root=a.root.resolve();dense.runtime_environment()
    if a.stage=='statistics':dense.statistics(a.job);return
    if a.stage=='queue':queue(root,a.tasks,a.variants);return
    if a.stage=='train':
        from run_qwen35_bridge_opd_vllm import queue as run
        assert a.job and a.output and not a.output.exists() and (a.job/'smoke_audit.json').exists()
        a.output.mkdir(parents=True)
        for name in ['protocol.json','train_prompts.json','eval_prompts.json','gaussian_statistics.pt']:
            if (a.job/name).exists():shutil.copyfile(a.job/name,a.output/name)
        p=json.loads((a.output/'protocol.json').read_text())
        p.update(reuse_completed_smoke_root=str(a.job.resolve()),training_scope='Formal fresh 3-epoch GRPO',wandb_run_name=a.output.name)
        save(a.output/'protocol.json',p);snapshot(a.output);run(a.output,'grpo');return
    if not (root/'plan.json').exists():prepare(root,a.tasks,a.variants)
    else:assert json.loads((root/'plan.json').read_text())==dict(tasks=a.tasks,variants=a.variants,smoke_only=True,formal_training_started=False)
    if a.stage=='prepare':print(root);return
    command=[sys.executable,'-u',str(SCRIPT),'queue','--root',str(root),'--tasks',*a.tasks,'--variants',*a.variants]
    with (root/'controller.log').open('a') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save(root/'controller.json',dict(pid=child.pid,command=command,smoke_only=True,time=time.time()))
    print(json.dumps(dict(pid=child.pid,root=str(root),smoke_only=True)))


if __name__=='__main__':main()
