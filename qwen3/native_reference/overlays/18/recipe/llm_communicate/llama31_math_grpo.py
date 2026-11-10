"""Matched Llama-3.1 70B->8B hidden bridge and 8B LoRA GRPO recipes."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from qwen35_bridge_opd_core import ROOT, TRAIN_SAMPLING, LLAMA31_EVAL_SAMPLING, save, quarter_epoch_schedule

STUDENT=ROOT/'data/models/Llama-3.1-8B-Instruct'
TEACHER=ROOT/'data/models/Llama-3.1-70B-Instruct'
STUDENT_REVISION='0e9e39f249a16976918f6564b8830bc894c89659'
TEACHER_REVISION='1605565b47bb9346c5515c34102e054115b4f98b'
DATA_REVISION='6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'
SUFFIX='\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
HERE=Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def layer_map():
    from hidden_bridge import bridge_layer_map
    return bridge_layer_map(32,80,'uniform_endpoints')


def make_hidden_bridge(student,teacher):
    import torch
    from direct_residual_chat_bridge import DirectResidualChatBridge
    class LlamaHiddenBridge(DirectResidualChatBridge):
        def features(self,ids,mask):
            teacher_device=self.teacher.device
            with torch.no_grad():
                positions=(mask.cumsum(-1)-1).clamp_min(0)
                result=self.teacher.model(input_ids=ids.to(teacher_device),attention_mask=mask.to(teacher_device),
                    position_ids=positions.to(teacher_device),use_cache=False,output_hidden_states=True)
                return tuple(h.detach().to(self.student.device).contiguous().clone()
                             for h in self.teacher_features(result.hidden_states))

        def prefill(self,ids,mask,extra=None):
            features=self.features(ids,mask)
            injection=mask.bool().clone(); injection[:,-1]=False
            if extra is not None:
                ids=torch.cat([ids,extra],1);mask=torch.cat([mask,torch.ones_like(extra)],1)
            with self.inject(features,injection):
                return self.student(input_ids=ids,attention_mask=mask,
                    position_ids=(mask.cumsum(-1)-1).clamp_min(0),use_cache=True,logits_to_keep=1)

        def response_hidden(self,prompt,response):
            from qwen35_bridge_grpo_core import checkpoint_student_blocks
            device=self.student.device
            prefix=torch.tensor([prompt],device=device)
            features=self.features(prefix,torch.ones_like(prefix))
            ids=torch.tensor([prompt+response[:-1]],device=device)
            injection=torch.ones_like(prefix,dtype=torch.bool);injection[:,-1]=False
            context=checkpoint_student_blocks(self) if torch.is_grad_enabled() and self.student_activation_checkpointing else nullcontext()
            with context,self.inject(features,injection):
                return self.student.model(input_ids=ids,attention_mask=torch.ones_like(ids),
                    position_ids=torch.arange(ids.shape[1],device=device)[None],use_cache=False).last_hidden_state[:,len(prompt)-1:]
    bridge=LlamaHiddenBridge(student,teacher,variant='teacher_residual',initialization='zero').eval()
    bridge.student_activation_checkpointing=True
    bridge.native_vllm_first_token=True
    return bridge


def load_bridge(root,protocol,devices):
    import torch
    from transformers import AutoModelForCausalLM
    options=dict(dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
    student=AutoModelForCausalLM.from_pretrained(protocol['student_checkpoint'],
        device_map={'':devices[0]},**options).eval()
    if protocol['bridge_recipe']=='llama31_lora':
        from qwen35_native_actor_grpo import NativeActor
        bridge=NativeActor(student,'lora',rank=128,alpha=128).eval()
        bridge.parity_limits=protocol.get('lora_parity_limits',dict(mean=.02,p95=.075,p99=.15,max=.2,importance_min=.8,importance_max=1.25))
        runtime=bridge.metadata();runtime.update(recipe='llama31_lora',parity_limits=bridge.parity_limits)
    else:
        assert len(devices)==2
        teacher=AutoModelForCausalLM.from_pretrained(protocol['teacher_checkpoint'],
            device_map={'':devices[1]},**options).eval()
        bridge=make_hidden_bridge(student,teacher)
        assert bridge.layer_map==protocol['layer_map']
        runtime=dict(recipe='llama31_hidden',teacher_device=str(teacher.device),student_device=str(student.device),
            layer_map=bridge.layer_map,teacher_dtype='bfloat16',student_dtype='bfloat16')
    assert sum(p.numel() for p in bridge.bridge_parameters())==protocol['trainable_parameters']
    return bridge,runtime


def last_complete_boxed(text):
    """Find the final balanced boxed expression without parsing repeated prose."""
    end=len(text)
    while True:
        start=text.rfind('\\boxed',0,end)
        if start<0:return None
        opening=start+len('\\boxed')
        while opening<len(text) and text[opening].isspace():opening+=1
        if opening<len(text) and text[opening]=='{':
            depth=0
            for i in range(opening,len(text)):
                if i and text[i-1]=='\\':continue
                if text[i]=='{':depth+=1
                elif text[i]=='}':
                    depth-=1
                    if depth==0:return text[start:i+1]
        end=start


def score(row):
    from math_verify import parse,verify,LatexExtractionConfig
    from math_verify.errors import TimeoutException
    if not row['gold'].strip():
        return dict(**row,score=None,acc=None,pred=None,gold_missing=True,scoring_timed_out=False)
    box=last_complete_boxed(row['response'])
    if box is None:
        return dict(**row,score=0,acc=0,pred='',gold_missing=False,scoring_timed_out=False)
    # Exact textual equality needs no symbolic parser and cannot time out.
    content=box[box.index('{')+1:-1].strip()
    if content==row['gold'].strip():
        return dict(**row,score=1,acc=1,pred=content,gold_missing=False,scoring_timed_out=False)
    config=(LatexExtractionConfig(boxed_match_priority=0,try_extract_without_anchor=False),)
    for timeout in (10,60):
        try:
            gold=parse('\\boxed{'+row['gold']+'}',extraction_config=config,parsing_timeout=timeout,raise_on_error=True)
            pred=parse(box,extraction_config=config,parsing_timeout=timeout,raise_on_error=True)
            acc=int(verify(gold,pred,timeout_seconds=timeout,raise_on_error=True))
            return dict(**row,score=acc,acc=acc,pred=str(pred),gold_missing=False,scoring_timed_out=False)
        except TimeoutException:
            continue
        except Exception:
            # Malformed mathematical predictions are incorrect answers.
            return dict(**row,score=0,acc=0,pred=content,gold_missing=False,scoring_timed_out=False)
    return dict(**row,score=-1,acc=None,pred=None,gold_missing=False,scoring_timed_out=True)


def training_rewards(rows):
    """Keep answer accuracy for reporting; unfinished responses get zero reward."""
    return [dict(r,reward=None if r['gold_missing'] else
        (0 if r['finish_reason']=='length' else r['acc'])) for r in rows]


def prepare_data(directory,train_source=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    directory.mkdir(parents=True,exist_ok=True)
    source=directory/'math_train.json'
    if train_source is not None:
        content=Path(train_source).read_text()
        train=json.loads(content) if content.lstrip().startswith('[') else [json.loads(x) for x in content.splitlines() if x.strip()]
        provenance=dict(source=str(Path(train_source).absolute()),sha256=sha(train_source))
    elif source.exists():
        train=json.loads(source.read_text());provenance=json.loads((directory/'math_train_source.json').read_text())
    else:
        from importlib.util import spec_from_file_location,module_from_spec
        spec=spec_from_file_location('math_reward',ROOT/'third_party/verl/verl/utils/reward_score/math_reward.py')
        reward=module_from_spec(spec);spec.loader.exec_module(reward)
        repo='DigitalLearningGmbH/MATH-lighteval'
        revision=HfApi().dataset_info(repo).sha
        data=load_dataset(repo,revision=revision,split='train')
        train=[]
        for i,row in enumerate(data):
            boxed=reward.last_boxed_only_string(row['solution'])
            gold=reward.remove_boxed(boxed) if boxed else ''
            train.append(dict(index=i,question=row['problem'],gold=gold,input_text=row['problem']+SUFFIX))
        provenance=dict(source=repo,revision=revision,split='train')
    assert len(train)==7500
    save(source,train);save(directory/'math_train_source.json',provenance)
    evaluation=load_dataset('HuggingFaceH4/MATH-500',revision=DATA_REVISION,split='test')
    tokenizer=AutoTokenizer.from_pretrained(STUDENT,local_files_only=True)
    teacher_tokenizer=AutoTokenizer.from_pretrained(TEACHER,local_files_only=True)
    assert tokenizer.get_vocab()==teacher_tokenizer.get_vocab()
    def tokenize(rows,evaluation=False):
        result=[]
        for i,r in enumerate(rows):
            question=r['problem'] if evaluation else r['question']
            gold=r['answer'] if evaluation else r['gold']
            text=question+SUFFIX
            ids=tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,return_dict=False)
            teacher_ids=teacher_tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,return_dict=False)
            assert ids==teacher_ids and 2<=len(ids)<=2048
            result.append(dict(index=i,question=question,gold=gold,input_text=text,prompt_token_ids=ids))
        return result
    training=tokenize(train); validation=tokenize(evaluation,True)
    assert len(validation)==500
    assert not {r['question'] for r in training}&{r['question'] for r in validation}
    save(directory/'train_prompts.json',training);save(directory/'eval_prompts.json',validation)
    return directory/'math_train.json',directory/'eval_prompts.json'


def prepare(base,mode,train_source=None):
    base=Path(base).absolute()
    if (base/'protocol.json').exists():
        raise RuntimeError('Use a fresh output directory; --queue uses an existing prepared recipe')
    official=json.loads((STUDENT/'generation_config.json').read_text())
    assert official['temperature']==LLAMA31_EVAL_SAMPLING['temperature'] and official['top_p']==LLAMA31_EVAL_SAMPLING['top_p']
    assert official['eos_token_id']==[128001,128008,128009]
    train_source,eval_source=prepare_data(base,train_source)
    steps,eval_steps=quarter_epoch_schedule(7500,64,1)
    protocol=dict(model_family='llama31',bridge_recipe='llama31_hidden',method='grpo',teacher='Llama-3.1-70B-Instruct',student='Llama-3.1-8B-Instruct',
        teacher_checkpoint=str(TEACHER),student_checkpoint=str(STUDENT),
        teacher_checkpoint_revision=TEACHER_REVISION,teacher_weight_precision='bf16',vocab_size=128256,
        teacher_hidden_size=8192,student_hidden_size=4096,
        formula='h_s[i] + P_i(h_t[layer_map[i]])',layer_mapping='uniform_endpoints',layer_map=layer_map(),
        mapping_rule='round(i * 79 / 31), zero-based decoder block indices',mapping_group_size=1,
        monotone_constraint=True,attention_types_preserved=True,
        hidden_space='decoder block input before RMSNorm',injection='prompt[:-1] only',
        constant_gate=1.,initialization='uniform +/-1e-4/sqrt(8192), seed42; no SFT/pretrain checkpoint',
        frozen_backbones=True,projection_dtype='float32',trainable_parameters=1073741824,
        dataset='math',objective='GRPO binary boxed MATH reward; group-normalized advantage',
        reward='VERL math-verify acc, 0/1; missing-gold groups have zero advantage',math_verify_boxed_only=True,
        optimizer='AdamW',learning_rate=1e-5,weight_decay=0.,betas=[.9,.999],grad_clip=1.,
        warmup_steps=0,scheduler='constant',epochs=1,examples_per_epoch=7500,batch_size=64,group_size=8,
        responses_per_full_batch=512,steps_per_epoch=steps,total_steps=steps,micro_batch=1,
        world_size=4,teacher_model_parallel_size=2,teacher_gpu_groups=[[0,1],[2,3],[4,5],[6,7]],decoder_gpu_ids=[0,2,4,6],gpu_ids=list(range(8)),ppo_epochs=1,ppo_mini_batch_size_prompts=64,
        ppo_mini_batch_size_responses=512,loss_agg_mode='seq-mean-token-mean',clip_ratio=.2,clip_ratio_c=3.,
        norm_adv_by_std=True,advantage_std_correction=1,advantage_epsilon=1e-6,
        use_kl_loss=False,use_kl_in_reward=False,entropy_coefficient=0.,teacher_response_forward=False,
        rollout_correction='detached token-level IS, cap 2; native vLLM sampled log probabilities',
        old_policy='actor log_probs.detach(), one on-policy minibatch and one PPO epoch (VERL default)',
        loss_distribution='Unfiltered actor log probabilities at sampled response tokens',
        enable_thinking=False,max_prompt_length=2048,max_response_length=4096,
        train_sampling=TRAIN_SAMPLING,eval_sampling=LLAMA31_EVAL_SAMPLING,
        sampling_source='https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/0e9e39f249a16976918f6564b8830bc894c89659/generation_config.json',
        eval_steps=eval_steps,eval_questions=500,eval_samples=4,total_evaluations=1+len(eval_steps),
        evaluations_per_epoch=4,checkpoint_steps=eval_steps,save_evaluation_bridges=False,
        validation_before_training=True,initial_eval_step=0,eval_seed='1234 + 4*index + sample_index',
        eval_coverage='MATH500 x4 before training, then every quarter epoch',
        eval_selection='Store bridge, optimizer and rank RNG at each quarterly evaluation',
        training_rollout_backend='HF bridge prefill + native vLLM full-KV decode',
        evaluation_backend='HF bridge prefill + native vLLM full-KV decode',
        vllm_tensor_parallel_size=1,vllm_replicas=4,vllm_memory_fraction=.25,
        vllm_max_model_len=6144,vllm_max_num_seqs=16,vllm_decode_batch_size=0,
        vllm_attention_backend='FLASH_ATTN',vllm_flash_attn_version=2,
        vllm_allocation_note='One GPU for 70B BF16 teacher, one for 8B actor and vLLM in each pair; smoke profiles resident worst case',
        source_sha256=sha(train_source),
        eval_source_sha256=sha(eval_source),
        train_prompts_sha256=hashlib.sha256((base/'train_prompts.json').read_bytes()).hexdigest(),
        eval_prompts_sha256=hashlib.sha256((base/'eval_prompts.json').read_bytes()).hexdigest(),
        filter='None; retain clipped, incorrect and empty-gold training samples',resume=False,
        train_seed_formula='42 + 100000*zero_based_epoch + 1000*global_step + 8*question_index + sample_index',
        data_order='Fresh torch.randperm(7500), seed 42 + zero_based_epoch',
        wandb_project='hidden-bridge-math-grpo',
        wandb_run_name='llama31_70b_to_8b_uniform_endpoints_constant_gate1_grpo_g8_bs64_4k_lr1e5_1epoch_eval5_vllm_8gpu_repair_v2')
    protocol.update(student_checkpoint_revision=STUDENT_REVISION,transfer_mode='hidden',
        native_vllm_first_token=True,student_activation_checkpointing=True,eos_token_ids=[128001,128008,128009],
        teacher_placement='dedicated second GPU per pair; BF16',teacher_model_loaded=True,
        lora_parity_limits=dict(mean=.02,p95=.075,p99=.15,max=.2,importance_min=.8,importance_max=1.25),
        smoke_parity_note='Llama BF16: p95 abs logprob < .075; mean < .02; max < .2; token IS between .8 and 1.25',
        score_worker_script='llama31_math_score_worker.py',
        train_reward_policy='zero_for_length_limit',recovery_checkpoint_every=5,
        scorer_policy='last complete balanced boxed expression; exact equality fast path; math-verify 10s then 60s retry',
        repair_version=2,
        reward='binary final-box correctness; length-limited training responses get reward 0; evaluation accuracy retains clipped responses',
        training_source=json.loads((base/'math_train_source.json').read_text()),
        eval_dataset_revision=DATA_REVISION,reference_run='qwen3_30b_a3b_to_4b_uniform_endpoints_constant_gate1_grpo_g8_bs64_4k_lr1e5_1epoch_eval5_vllm_8gpu')
    if mode=='lora':
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        config=LlamaConfig.from_pretrained(STUDENT,local_files_only=True)
        with torch.device('meta'): model=LlamaForCausalLM(config)
        targets=[(n,m) for n,m in model.model.named_modules() if isinstance(m,torch.nn.Linear)]
        protocol.update(bridge_recipe='llama31_lora',teacher='none',teacher_checkpoint=None,
            teacher_model_loaded=False,teacher_placement=None,teacher_model_parallel_size=1,teacher_gpu_groups=[[i] for i in range(8)],
            teacher_checkpoint_revision=None,teacher_hidden_size=None,
            decoder_gpu_ids=list(range(8)),world_size=8,vllm_replicas=8,
            layer_map=[],layer_mapping='standalone_no_teacher',mapping_rule=None,constant_gate=None,
            actor_training_mode='lora',lora_rank=128,lora_alpha=128,lora_dropout=0.,
            lora_target_modules=[n for n,m in targets],
            trainable_parameters=sum(128*(m.in_features+m.out_features) for n,m in targets),
            formula='frozen W(x) + (alpha/r) B(A(x))',projection_dtype=None,
            initialization='LoRA A Kaiming uniform; B=0; seed42; no SFT/pretraining checkpoint',
            vllm_worker_script='qwen35_native_actor_vllm_worker.py',
            training_rollout_backend='native vLLM LoRA, update and audit adapter tensors each rollout',
            evaluation_backend='native vLLM with current LoRA adapter',
            wandb_run_name='llama31_8b_inst_lora_r128_alpha128_grpo_g8_bs64_4k_lr1e5_1epoch_eval5_vllm_8gpu_repair_v2')
    save(base/'protocol.json',protocol)
    save(base/'mapping.json',dict(teacher_layers_0based=protocol['layer_map']))
    files=[Path(__file__),ROOT/'scripts/direct_residual_chat_bridge.py',
        *[HERE/n for n in ('train_qwen35_bridge_opd_vllm.py','run_qwen35_bridge_opd_vllm.py',
            'qwen35_bridge_opd_core.py','qwen35_bridge_grpo_core.py','qwen35_native_actor_grpo.py',
            'qwen35_native_actor_vllm_worker.py','qwen35_bridge_opd_vllm_worker.py',
            'qwen35_hybrid_memory_connector.py','llama31_math_score_worker.py','prepare_qwen35_bridge_vllm.py',
            'qwen3_30b_4b_math_grpo.py','qwen3_4b_06b_grpo.py')]]
    protocol['verify_source_hashes']=True
    save(base/'protocol.json',protocol)
    save(base/'source_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
    for p in files:
        target=base/'source_snapshot'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    return protocol


def summarize_validation(run_dir):
    """Read local results, keeping the best mean accuracy and pass@4 separately."""
    run_dir=Path(run_dir)
    entries=[]
    for path in sorted((run_dir/'evaluation').glob('step-*/summary.json')):
        r=json.loads(path.read_text())
        if r.get('complete'):
            entries.append(dict(step=r['step'],avg_at_4=r['accuracy'],pass_at_4=r['pass_at_k'],
                samples_per_question=r['samples_per_question'],summary=str(path)))
    if not entries:return dict(evaluations=0,best_avg_at_4=None,best_pass_at_4=None)
    result=dict(evaluations=len(entries),history=entries,
        best_avg_at_4=max(entries,key=lambda r:(r['avg_at_4'],-r['step'])),
        best_pass_at_4=max(entries,key=lambda r:(r['pass_at_4'],-r['step'])))
    save(run_dir/'best_validation.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['bridge','lora'],default='bridge')
    parser.add_argument('--root',type=Path)
    parser.add_argument('--train-source',type=Path,help='Optional canonical 7500-row MATH JSON/JSONL with question/gold')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--best-val',action='store_true',help='Read best avg@4 and pass@4 from local validation JSON; no training')
    parser.add_argument('--queue',action='store_true',help='Use an already prepared run without rewriting provenance')
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args()
    for key in ('PLATOUR_BRIDGE_PYTHON','PLATOUR_GENERATION_PYTHON','PLATOUR_SCORING_PYTHON'):
        os.environ.setdefault(key,str(ROOT/'.runtime/math500/bin/python'))
    os.environ.setdefault('WANDB_MODE','offline')
    base=(args.root or ROOT/'outputs/llm_communicate'/f'llama31_{"70b_to_8b_hidden" if args.mode=="bridge" else "8b_lora_r128_alpha128"}_math_grpo_g8_bs64_4k_lr1e5_ep1_eval5_8gpu_repair_v2').absolute()
    if args.best_val:
        print(json.dumps(summarize_validation(base/'train'),indent=2));return
    if not args.queue and not (base/'protocol.json').exists(): prepare(base,args.mode,args.train_source)
    else:
        protocol=json.loads((base/'protocol.json').read_text())
        assert protocol['bridge_recipe']==('llama31_hidden' if args.mode=='bridge' else 'llama31_lora')
        for key in ('train','eval'):
            assert sha(base/f'{key}_prompts.json')==protocol[f'{key}_prompts_sha256']
    if args.prepare_only:
        print(json.dumps(dict(prepared=True,root=str(base),mode=args.mode)));return
    if args.detach:
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--queue','--mode',args.mode,'--root',str(base)]
        if args.smoke_only:command.append('--smoke-only')
        with (base/'controller.log').open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(base/'controller.json',dict(pid=process.pid,command=command,started=time.time()))
        print(json.dumps(dict(pid=process.pid,root=str(base))));return
    from run_qwen35_bridge_opd_vllm import queue
    queue(base,'grpo',smoke_only=args.smoke_only)


if __name__=='__main__':main()
