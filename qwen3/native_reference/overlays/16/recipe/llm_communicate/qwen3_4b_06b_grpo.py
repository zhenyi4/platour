"""Matched 4B->0.6B hidden GRPO: SFT epoch3 initialization versus exact zeros."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

from c2c_sft_17b_common import ROOT, DATA, PYTHON, CPU, environment
from qwen35_bridge_opd_core import TRAIN_SAMPLING, QWEN3_EVAL_SAMPLING, quarter_epoch_schedule, weight_digest
import generate_qwen35_train_suite as suite

HERE = ROOT / 'recipe/llm_communicate'
SFT = ROOT / 'outputs/llm_communicate/qwen3_4b_to_06b_native_c2c_hidden_sft'
BASE = ROOT / 'outputs/llm_communicate/qwen3_4b_to_06b_hidden_sft_vs_zero_grpo_ep3'


def state_digest(state, zero=False):
    import torch
    result = hashlib.sha256()
    for name, value in state.items():
        value = torch.zeros_like(value) if zero else value
        result.update(name.encode()); result.update(str(tuple(value.shape)).encode())
        result.update(value.contiguous().numpy().tobytes())
    return result.hexdigest()


def load_bridge(base, protocol):
    import torch
    from transformers import AutoModelForCausalLM
    from direct_residual_chat_bridge import DirectResidualChatBridge
    class Bridge(DirectResidualChatBridge):
        def initialize_for_grpo(self, seed=42):
            assert not getattr(self, 'grpo_initialized', False), 'Initialization is applied exactly once'
            assert all(p.count_nonzero() == 0 for p in self.bridge_parameters())
            if protocol['initialization_mode'] == 'sft':
                checkpoint = Path(protocol['sft_checkpoint'])
                assert suite.sha(checkpoint) == protocol['sft_checkpoint_file_sha256']
                self.load_projectors(checkpoint)
            actual = weight_digest(self)
            assert actual == protocol['expected_initial_bridge_state_sha256']
            self.grpo_initialized = True
            return dict(kind='SFT epoch3 checkpoint' if protocol['initialization_mode']=='sft' else 'all-zero linear',
                seed=seed, constant_gate=1., scale=0. if protocol['initialization_mode']=='zero' else None,
                sft_checkpoint_loaded=protocol['initialization_mode']=='sft',
                sft_checkpoint=protocol.get('sft_checkpoint'),
                sft_checkpoint_file_sha256=protocol.get('sft_checkpoint_file_sha256'),
                optimizer_resumed=False, grpo_step_starts_at=0, bridge_state_sha256=actual)
        def metadata(self):
            value = super().metadata()
            if getattr(self, 'grpo_initialized', False):
                value.update(grpo_initialization=protocol['initialization_mode'],
                    initialization='SFT epoch3 projector weights' if protocol['initialization_mode']=='sft' else 'all-zero projection matrices',
                    sft_parent_file_sha256=protocol.get('sft_checkpoint_file_sha256'), constant_gate=1.)
            return value
    options = dict(dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True,
                   device_map={'':torch.cuda.current_device()})
    student = AutoModelForCausalLM.from_pretrained(protocol['student_checkpoint'], **options)
    teacher = AutoModelForCausalLM.from_pretrained(protocol['teacher_checkpoint'], **options)
    bridge = Bridge(student, teacher, variant='teacher_residual', initialization='zero', lm_chunk_tokens=128).eval()
    bridge.native_vllm_first_token = True
    assert bridge.layer_map == protocol['layer_map']
    assert sum(p.numel() for p in bridge.bridge_parameters()) == 73400320
    return bridge, dict(student=protocol['student'], teacher=protocol['teacher'], teacher_dtype='bfloat16',
        student_dtype='bfloat16', projector_dtype='float32', transfer_mode='hidden',
        layer_map=bridge.layer_map, first_response_token_backend='native_vllm', cuda_device=torch.cuda.current_device())


def generate_native_first(bridge, engine, rows, digest, sampling, cap, directory, pad, return_logprobs=False):
    """Deduplicate question prefills and import prompt[:-1], preserving all-token vLLM RNG."""
    import torch
    from train_qwen35_bridge_opd_vllm import prefill, prompt_tensors
    from prepare_qwen35_bridge_vllm import cache_to_cpu, export_memory
    directory.mkdir(parents=True, exist_ok=True)
    memory = engine.directory / 'memory'; memory.mkdir(exist_ok=True)
    assert not list(memory.glob('*.safetensors')), 'Refuse stale prefix memory'
    started = time.monotonic()
    # Prefix memory depends on prompt and current bridge weights, not sample seed.
    unique = {}; transfers = {}
    for row in rows:
        key = tuple(row['prompt_token_ids'])
        unique.setdefault(key, row)
    chosen = list(unique.values())
    with torch.inference_mode():
        for offset in range(0, len(chosen), 8):
            batch = chosen[offset:offset+8]
            ids, mask = prompt_tensors(batch, pad, 'cuda')
            out = prefill(bridge, ids, mask)
            cache = cache_to_cpu(out.past_key_values)
            assert all(key.endswith('.kv') for key in cache), 'Dense Qwen3 receiver only'
            # Left-padding stays on the left. Remove the uncached final question
            # token before slicing each example's actual prefix in export_memory.
            cache = {name: value[:, :, :-1] for name, value in cache.items()}
            for i, row in enumerate(batch):
                prefix = row['prompt_token_ids'][:-1]
                transfer = export_memory(cache, dict(prompt_token_ids=prefix), digest, 'full_hidden', memory, i)
                transfer['checkpoint_sha256'] = digest
                transfers[tuple(row['prompt_token_ids'])] = transfer
            del out, cache, ids, mask
    exported = [dict(**row, transfer=transfers[tuple(row['prompt_token_ids'])]) for row in rows]
    torch.cuda.synchronize(); seconds = time.monotonic()-started
    manifest = directory / 'manifest.json'
    suite.save(manifest, dict(rows=exported, tokens=cap, sampling=sampling, bridge_state_sha256=digest,
        return_logprobs=return_logprobs, first_response_token_backend='native_vllm',
        identity_kind='SHA256 of ordered actual FP32 projection tensor names, shapes and bytes',
        unique_prompt_prefills=len(chosen), requested_responses=len(rows), export_seconds=seconds))
    torch.cuda.empty_cache()
    result = engine.call(dict(manifest=str(manifest), result=str(directory / 'responses.json')))
    assert result['bridge_state_sha256'] == digest and len(result['rows']) == len(rows)
    for actual, expected in zip(result['rows'], rows, strict=True):
        assert all(actual[key] == value for key, value in expected.items())
        assert actual['bridge_state_sha256'] == digest
        assert 0 < len(actual['response_token_ids']) == actual['tokens'] <= cap
        if return_logprobs: assert len(actual['rollout_log_probs']) == actual['tokens']
    for path in memory.glob('*.safetensors'): path.unlink()
    return result, seconds


def verify_vllm_handoff(bridge, engine, rows, directory, pad, eos):
    """Nonzero, mixed-length prompt batch: same-history HF/vLLM log probabilities."""
    import torch
    from train_qwen35_bridge_opd_vllm import prefill, prompt_tensors
    original = [p.detach().cpu().clone() for p in bridge.bridge_parameters()]
    before = weight_digest(bridge)
    directory.mkdir(parents=True, exist_ok=True)
    chosen = [min(rows,key=lambda r:len(r['prompt_token_ids'])),
              max(rows,key=lambda r:len(r['prompt_token_ids'])), rows[0]]
    chosen = [dict(**r, sample_index=0, sampling_seed=42+i) for i, r in enumerate(chosen)]
    try:
        if all(not p.count_nonzero() for p in bridge.bridge_parameters()):
            with torch.no_grad():
                rng = torch.Generator(device=bridge.student.device).manual_seed(42)
                for p in bridge.bridge_parameters(): p.uniform_(-.01/math.sqrt(2560), .01/math.sqrt(2560), generator=rng)
        identity = weight_digest(bridge)
        result, _ = generate_native_first(bridge,engine,chosen,identity,
            dict(TRAIN_SAMPLING,temperature=0.),16,directory/'probe',pad,return_logprobs=True)
        ids, mask = prompt_tensors(chosen,pad,'cuda')
        errors, first_match = [], []
        with torch.inference_mode():
            out = prefill(bridge,ids,mask)
            first = out.logits[:, -1].argmax(-1).tolist()
            first_match = [v==r['response_token_ids'][0] for v,r in zip(first,result['rows'],strict=True)]
            del out, ids, mask
            for row in result['rows']:
                ids,mask = prompt_tensors([row],pad,'cuda')
                out = prefill(bridge,ids,mask)
                for token,lp in zip(row['response_token_ids'][:8],row['rollout_log_probs'][:8],strict=True):
                    logits = out.logits[0,-1].float()
                    errors.append(abs(float(logits[token]-logits.logsumexp(-1))-lp))
                    out = bridge.student(input_ids=torch.tensor([[token]],device='cuda'),
                        past_key_values=out.past_key_values,use_cache=True,logits_to_keep=1)
        report = dict(passed=sum(errors)/len(errors)<.05 and max(errors)<.2,
            mean_abs_logprob_error=sum(errors)/len(errors),max_abs_logprob_error=max(errors),
            first_greedy_matches=sum(first_match),probes=len(chosen),teacher_forced_tokens=8,
            nonzero_bridge=True,left_padding_tested=True,native_vllm_first_response_token=True,
            zero_probe_scale=.01 if all(not p.count_nonzero() for p in original) else None)
        suite.save(directory/'parity.json',report); assert report['passed'], report
    finally:
        with torch.no_grad():
            for p,value in zip(bridge.bridge_parameters(),original,strict=True): p.copy_(value)
        assert weight_digest(bridge)==before
        torch.cuda.empty_cache()


def prepare(root):
    import torch
    from transformers import AutoTokenizer
    from hidden_bridge import bridge_layer_map
    root.mkdir(parents=True,exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'data/models/Qwen3-0.6B',local_files_only=True)
    jobs=[]
    for task,count,cap,prompt_cap,k,n in [('gsm8k',7473,2048,1024,2,1319),('math',7500,4096,2048,4,500)]:
        source=DATA/task/'Qwen3-4B/scored.jsonl'
        rows=suite.read(source); assert len(rows)==count and [r['index'] for r in rows]==list(range(count))
        train=[]
        for r in rows:
            ids=tokenizer.apply_chat_template([dict(role='user',content=r['input_text'])],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False)
            if hasattr(ids,'keys'): ids=ids['input_ids']
            assert ids==r['prompt_token_ids'] and 2<=len(ids)<=prompt_cap
            train.append({key:r[key] for key in ['index','question','input_text','gold','prompt_token_ids']})
        sft_base=SFT/'hidden'/f'4b_to_06b_{task}'
        sft_done=json.loads((sft_base/'train/complete.json').read_text()); assert sft_done['complete'] and sft_done['epochs']==3
        checkpoint=Path(sft_done['selected_checkpoint']); assert suite.sha(checkpoint)==sft_done['checkpoint_sha256']
        config=json.loads(checkpoint.with_name('subspace_config.json').read_text())
        mapping=bridge_layer_map(28,36,'uniform_endpoints'); assert config['layer_map']==mapping
        assert config['student_dim']==1024 and config['teacher_dim']==2560 and config['initialization']=='all-zero projection matrices'
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)['P']
        expected={'sft':state_digest(state),'zero':state_digest(state,zero=True)}
        prompts=json.loads((sft_base/'eval_prompts.json').read_text()); assert len(prompts)==n
        assert not {tuple(r['prompt_token_ids']) for r in train}&{tuple(r['prompt_token_ids']) for r in prompts}
        steps,eval_steps=quarter_epoch_schedule(count,64,3)
        for init in ['sft','zero']:
            name=f'{task}_{init}_grpo';base=root/name;base.mkdir(exist_ok=True)
            suite.immutable(base/'train_prompts.json',train);suite.immutable(base/'eval_prompts.json',prompts)
            protocol=dict(model_family='qwen3',bridge_recipe='qwen3_4b_06b_hidden',method='grpo',transfer_mode='hidden',
                teacher='Qwen3-4B',student='Qwen3-0.6B',teacher_checkpoint=str(ROOT/'data/models/Qwen3-4B'),
                student_checkpoint=str(ROOT/'data/models/Qwen3-0.6B'),teacher_weight_precision='bf16',
                formula='h_s[i] + P_i(h_t[layer_map[i]])',layer_mapping='uniform_endpoints',layer_map=mapping,
                hidden_space='decoder block input before RMSNorm',injection='prompt[:-1] only',constant_gate=1.,
                initialization_mode=init,expected_initial_bridge_state_sha256=expected[init],
                sft_checkpoint=str(checkpoint) if init=='sft' else None,
                sft_checkpoint_file_sha256=suite.sha(checkpoint) if init=='sft' else None,
                sft_epochs=3 if init=='sft' else 0,resume_training=False,optimizer_resumed=False,
                teacher_hidden_size=2560,student_hidden_size=1024,vocab_size=151936,trainable_parameters=73400320,
                projection_dtype='float32',frozen_backbones=True,teacher_response_forward=False,
                dataset=task,examples_per_epoch=count,epochs=3,batch_size=64,group_size=8,
                responses_per_full_batch=512,steps_per_epoch=steps,total_steps=steps*3,
                world_size=8,gpu_ids=list(range(8)),micro_batch=1,student_activation_checkpointing=True,
                optimizer='AdamW',learning_rate=1e-5,scheduler='constant',warmup_steps=0,weight_decay=0.,grad_clip=1.,
                ppo_epochs=1,loss_agg_mode='seq-mean-token-mean',clip_ratio=.2,clip_ratio_c=3.,
                rollout_correction='detached token-level IS cap2, native-vLLM raw sampling logprobs',
                norm_adv_by_std=True,advantage_std_correction=1,advantage_epsilon=1e-6,
                use_kl_loss=False,use_kl_in_reward=False,entropy_coefficient=0.,
                train_sampling=TRAIN_SAMPLING,eval_sampling=QWEN3_EVAL_SAMPLING,enable_thinking=False,
                max_prompt_length=prompt_cap,max_response_length=cap,math_verify_boxed_only=True,
                eval_samples=k,eval_questions=n,eval_seed_stride=k,eval_steps=eval_steps,
                checkpoint_steps=eval_steps,total_evaluations=13,validation_before_training=True,
                save_evaluation_bridges=False,native_vllm_first_token=True,
                training_rollout_backend='HF bridge question prefill, native vLLM samples every response token',
                evaluation_backend='HF bridge question prefill, native vLLM samples every response token',
                vllm_memory_fraction=.25,vllm_max_model_len=prompt_cap+cap,vllm_max_num_seqs=32,
                vllm_decode_batch_size=0,vllm_tensor_parallel_size=1,prefill_unique_questions=True,
                filter='None; retain clipped/incorrect rollouts; missing-gold groups get zero advantage',
                reward='VERL boxed-only math-verify binary correctness',
                train_seed_formula='42 + 100000*epoch + 1000*step + 8*question_index + sample_index',
                data_order='torch.randperm(full_train_count), seed42+epoch; full canonical train split',
                source=str(source),source_sha256=suite.sha(source),
                train_prompts_sha256=suite.sha(base/'train_prompts.json'),eval_prompts_sha256=suite.sha(base/'eval_prompts.json'),
                verify_source_hashes=True,wandb_project='hidden-bridge-gsm8k-math-grpo',
                wandb_run_name=f'qwen3_4b_to_06b_{task}_{init}_init_hidden_uniform_grpo_g8_bs64_{cap//1024}k_lr1e5_3epoch_eval13_vllm_8gpu')
            suite.immutable(base/'protocol.json',protocol)
            suite.save(base/'initialization_provenance.json',dict(mode=init,expected_state_sha256=expected[init],
                sft_checkpoint=str(checkpoint) if init=='sft' else None,sft_file_sha256=protocol['sft_checkpoint_file_sha256'],
                original_sft_weight_hash_verified=True if init=='sft' else None,new_grpo_optimizer=True))
            jobs.append(dict(name=name,root=str(base),task=task,initialization=init,total_steps=steps*3))
    suite.immutable(root/'plan.json',dict(jobs=jobs,gpu_ids=list(range(8)),epochs=3,
        receiver='Qwen3-0.6B',sender='Qwen3-4B',order=[j['name'] for j in jobs],
        all_smokes_before_formal=True,source_sft_root=str(SFT)))


def source_files():
    names=['qwen3_4b_06b_grpo.py','train_qwen35_bridge_opd_vllm.py','run_qwen35_bridge_opd_vllm.py',
        'qwen35_bridge_opd_core.py','qwen35_bridge_grpo_core.py','qwen35_bridge_opd_vllm_worker.py',
        'qwen35_bridge_opd_score_worker.py','qwen35_bridge_logits_processors.py','qwen35_hybrid_memory_connector.py',
        'prepare_qwen35_bridge_vllm.py','eval_qwen35_35b_4b_math_bridge_8gpu.py','generate_qwen35_train_suite.py',
        'c2c_sft_17b_common.py']
    paths=[HERE/name for name in names]
    paths += [ROOT/'scripts'/name for name in ['direct_residual_chat_bridge.py','shared_subspace_chat_bridge.py',
        'shared_subspace_residual_bridge.py','shared_subspace_bridge.py','hidden_bridge.py','full_vocab_kl.py']]
    paths += [ROOT/'third_party/verl/verl/utils/reward_score'/name for name in ['math_verify.py','math_reward.py']]
    return {str(p.relative_to(ROOT)):suite.sha(p) for p in paths}


def seal(root):
    fingerprints=source_files();suite.immutable(root/'source_sha256.json',fingerprints)
    for job in json.loads((root/'plan.json').read_text())['jobs']:
        base=Path(job['root']);suite.immutable(base/'source_sha256.json',fingerprints)
    for name in fingerprints:
        target=root/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)


def queue(root, smoke_only=True):
    from run_qwen35_bridge_opd_vllm import stop_workers
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((root/'plan.json').read_text());fingerprints=json.loads((root/'source_sha256.json').read_text())
    active=None;completed=[]
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def execute(job,smoke):
        nonlocal active
        assert source_files()==fingerprints
        base=Path(job['root']);protocol=json.loads((base/'protocol.json').read_text())
        assert suite.sha(base/'train_prompts.json')==protocol['train_prompts_sha256']
        assert suite.sha(base/'eval_prompts.json')==protocol['eval_prompts_sha256']
        mode='smoke' if smoke else 'train'
        suite.save(root/'status.json',dict(phase=mode,job=job['name'],completed=completed,time=time.time()))
        command=[str(PYTHON),'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
            str(HERE/'train_qwen35_bridge_opd_vllm.py'),'--root',str(base),'--objective','grpo']
        if smoke:command+=['--smoke','--stress']
        with (base/f'{mode}.log').open('a') as log:
            active=subprocess.Popen(command,cwd=ROOT,env=environment(list(range(8)),compat=False),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            suite.save(root/'active.json',dict(pid=active.pid,command=command,job=job['name'],root=str(base)))
            code=active.wait()
            if code:stop_workers(base,active)
            active=None
        assert code==0,f'{job["name"]} {mode} failed with {code}: {base}/{mode}.log'
        done=json.loads((base/mode/'complete.json').read_text());assert done['complete'] and done['world_size']==8
        assert done['initial_bridge_state_sha256']==protocol['expected_initial_bridge_state_sha256']
        assert done['nonzero_gradient_updates']>0 and done['replica_exact_weights'] and done['frozen_backbones_verified']
        if smoke:
            parity=json.loads((base/'smoke/vllm_handoff_parity/parity.json').read_text());assert parity['passed']
            pressure=json.loads((base/'smoke/worst_case_profile.json').read_text());assert pressure['passed'] and pressure['vllm_resident']
            quality=json.loads((base/'smoke/evaluation/step-0002/summary.json').read_text())
            assert quality['accuracy']>0 and quality['clipping_rate']<1.
            suite.save(base/'smoke_audit.json',dict(passed=True,training=done,parity=parity,worst_case=pressure,quality=quality))
        else:
            assert done['steps']==job['total_steps'] and done['epochs']==3
            assert done['examples_seen']==3*protocol['examples_per_epoch'] and done['evals']==13
    try:
        usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        assert len(usage.splitlines())==8 and all(int(v)<512 for v in usage.splitlines())
        for job in plan['jobs']:
            if not (Path(job['root'])/'smoke_audit.json').exists():execute(job,True)
        suite.save(root/'smokes_complete.json',dict(complete=True,jobs=plan['order'],time=time.time()))
        policy=json.loads((root/'launch_policy.json').read_text()) if (root/'launch_policy.json').exists() else {}
        if smoke_only or policy.get('formal_training_authorized') is False:
            suite.save(root/'status.json',dict(phase='smokes_complete',completed=completed,time=time.time()));return
        for job in plan['jobs']:
            execute(job,False);completed.append(job['name'])
        suite.save(root/'status.json',dict(phase='complete',completed=completed,time=time.time()))
    except BaseException as error:
        suite.save(root/'status.json',dict(phase='stopped' if isinstance(error,SystemExit) else 'failed',
            completed=completed,error=repr(error),traceback=traceback.format_exc(),time=time.time()));raise
    finally:
        if active is not None and active.poll() is None:stop_workers(Path(json.loads((root/'active.json').read_text())['root']),active)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','seal','queue'])
    parser.add_argument('--root',type=Path,default=BASE)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--formal',action='store_true',help='Explicitly request formal training after all four smoke audits')
    args=parser.parse_args();root=args.root.absolute()
    assert not (args.formal and args.smoke_only), 'Choose smoke-only or formal'
    smoke_only=args.smoke_only or not args.formal
    if args.stage=='prepare':prepare(root)
    elif args.stage=='seal':seal(root)
    elif args.detach:
        assert not (root/'controller.json').exists(),'Inspect previous controller before launch'
        with (root/'pipeline.log').open('a') as log:
            process=subprocess.Popen([str(CPU),'-u',str(Path(__file__).absolute()),'queue','--root',str(root),
                *(['--smoke-only'] if smoke_only else ['--formal'])],cwd=ROOT,env=environment([],compat=False),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        suite.save(root/'controller.json',dict(pid=process.pid,started=time.time()))
        print(json.dumps(dict(pid=process.pid,root=str(root))),flush=True)
    else:queue(root,smoke_only)


if __name__=='__main__':main()
