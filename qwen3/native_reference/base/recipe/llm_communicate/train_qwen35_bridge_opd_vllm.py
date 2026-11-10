"""Distributed native vLLM on-policy rollouts and differentiable HF bridge updates."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
import torch.distributed as dist

from qwen35_bridge_opd_core import (ROOT, BASE, TRAIN_SAMPLING, EVAL_SAMPLING, LEGACY_EVAL_SAMPLING, QWEN3_EVAL_SAMPLING, LLAMA31_EVAL_SAMPLING,
    initialize, opd_loss, weight_digest, first_tokens, save)
from prepare_qwen35_bridge_vllm import cache_to_cpu, export_memory
from eval_qwen35_35b_4b_math_bridge_8gpu import prefill as hidden_prefill, prompt_tensors


def rollout_rows(train_rows, selection, rank, world, group_size, epoch, step):
    """One independent seed per reply, identically for GRPO and grouped OPD."""
    return [{**train_rows[i], 'sample_index': j,
             'sampling_seed': 42 + 100000*epoch + 1000*step + group_size*i + j}
            for i in selection[rank::world] for j in range(group_size)]


def validate_sampling(protocol):
    expected=QWEN3_EVAL_SAMPLING if protocol.get('model_family')=='qwen3' else EVAL_SAMPLING
    if protocol.get('model_family')=='llama31':
        expected=LLAMA31_EVAL_SAMPLING
    if protocol.get('eval_sampling_profile')=='historical_qwen35_non_thinking_reasoning':
        assert protocol.get('model_family','qwen35')!='qwen3' and protocol['method']=='grpo'
        expected=LEGACY_EVAL_SAMPLING
    assert protocol['eval_sampling']==expected, 'Evaluation defaults changed; use a fresh protocol/output root'
    assert protocol['train_sampling']==TRAIN_SAMPLING


def prefill(bridge, ids, mask, extra=None):
    if getattr(bridge, 'prefill', None) is not None:
        return bridge.prefill(ids, mask, extra=extra)
    return hidden_prefill(bridge, ids, mask, extra=extra)


class RPC:
    def __init__(self, script, directory, gpu=None, memory_fraction=.12, max_model_len=6144, boxed_only=True,
                 max_num_seqs=32, decode_batch_size=0, model_path=None, generation_config=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        for key in list(env):
            if key in ['RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK',
                       'ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'] or key.startswith('TORCHELASTIC_'):
                env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1',
            HF_DATASETS_OFFLINE='1', MATH_VERIFY_BOXED_ONLY='1' if boxed_only else '0', VLLM_NO_USAGE_STATS='1')
        runtime = 'nemotron-smoke' if gpu is None else 'qwen35-generation'
        runtime_key = 'PLATOUR_SCORING_PYTHON' if gpu is None else 'PLATOUR_GENERATION_PYTHON'
        python = Path(os.environ.get(runtime_key,str(ROOT/'.runtime'/runtime/'bin/python'))).absolute()
        cmd = [str(python), '-u', str(ROOT/'recipe/llm_communicate'/script),
               '--directory', str(self.directory)]
        if gpu is not None:
            cmd += ['--memory-fraction', str(memory_fraction), '--max-model-len', str(max_model_len),
                    '--max-num-seqs', str(max_num_seqs), '--decode-batch-size', str(decode_batch_size)]
            if model_path is not None:
                cmd += ['--model-path', str(model_path)]
            if generation_config is not None:
                cmd += ['--generation-config', generation_config]
        self.log = (self.directory/'worker.log').open('a')
        self.proc = subprocess.Popen(cmd, env=env, cwd=ROOT, stdin=subprocess.PIPE,
            stdout=self.log, stderr=subprocess.STDOUT, text=True)
        self.wait(self.directory/'ready.json', timeout=1200)

    def wait(self, path, timeout=1200):
        started = time.monotonic()
        while not Path(path).exists():
            if self.proc.poll() is not None:
                raise RuntimeError(f'Worker exited {self.proc.returncode}; see {self.directory}/worker.log')
            if time.monotonic()-started > timeout:
                raise TimeoutError(f'Worker timed out waiting for {path}')
            time.sleep(.2)
        value = json.loads(Path(path).read_text())
        if value.get('complete') is False:
            raise RuntimeError(value)
        return value

    def call(self, request):
        target = Path(request['result'])
        assert not target.exists(), f'Refusing stale RPC output {target}'
        self.proc.stdin.write(json.dumps(request)+'\n')
        self.proc.stdin.flush()
        return self.wait(target, timeout=3600)

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('{"command":"stop"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=30)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        self.log.close()


def generate(bridge, engine, rows, digest, sampling, cap, directory, pad, eos, return_logprobs=False):
    if getattr(bridge,'generate_native_actor',None) is not None:
        return bridge.generate_native_actor(engine,rows,digest,sampling,cap,directory,return_logprobs)
    if getattr(bridge, 'native_vllm_first_token', False):
        from qwen3_4b_06b_grpo import generate_native_first
        return generate_native_first(bridge, engine, rows, digest, sampling, cap, directory, pad,
                                     return_logprobs=return_logprobs)
    directory.mkdir(parents=True, exist_ok=True)
    memory = engine.directory/'memory'
    memory.mkdir(exist_ok=True)
    assert not list(memory.glob('*.safetensors')), 'Stale memory from a prior call'
    export_started = time.monotonic()
    exported = []
    with torch.inference_mode():
        for offset in range(0,len(rows),8):
            batch = rows[offset:offset+8]
            ids, mask = prompt_tensors(batch,pad,'cuda')
            output = prefill(bridge,ids,mask)
            first = first_tokens(output.logits[:, -1],batch,cap,sampling)
            if return_logprobs:
                assert sampling==TRAIN_SAMPLING
                logits=output.logits[:,-1].float()
                labels=torch.tensor(first,device=logits.device)
                first_log_probs=(logits.gather(1,labels[:,None]).squeeze(1)-logits.logsumexp(-1)).tolist()
            cache = cache_to_cpu(output.past_key_values)
            for i,row in enumerate(batch):
                transfer = export_memory(cache,row,digest,'full_hidden',memory,i)
                transfer['checkpoint_sha256'] = digest
                extra={'hf_first_logprob':first_log_probs[i]} if return_logprobs else {}
                exported.append(dict(**row,**extra,transfer=transfer,hf_first_token=first[i],handoff_is_eos=first[i] in eos))
            del output, cache, ids, mask
    torch.cuda.synchronize()
    export_seconds = time.monotonic()-export_started
    manifest = directory/'manifest.json'
    save(manifest,dict(rows=exported,tokens=cap,sampling=sampling,bridge_state_sha256=digest,return_logprobs=return_logprobs,
        identity_kind='SHA256 of ordered actual FP32 projection tensor names, shapes and bytes',
        prefill_backend='Original HF bridge',decode_backend='native vLLM 0.17.1',
        export_seconds=export_seconds))
    torch.cuda.empty_cache()
    result = engine.call(dict(manifest=str(manifest), result=str(directory/'responses.json')))
    assert result['bridge_state_sha256'] == digest and len(result['rows']) == len(rows)
    assert all(r['bridge_state_sha256']==digest and r['prompt_token_ids']==expected['prompt_token_ids']
               and r['sampling_seed']==expected['sampling_seed'] for r,expected in zip(result['rows'],rows,strict=True))
    for path in memory.glob('*.safetensors'):
        path.unlink()
    return result, export_seconds


def bridge_checkpoint_state(bridge):
    if hasattr(bridge, 'bridge_checkpoint_state'):
        return bridge.bridge_checkpoint_state()
    return {'P': {k:v.detach().cpu() for k,v in bridge.maps.state_dict().items()}}


def checkpoint(bridge, optimizer, directory, step, initial_sha):
    directory.mkdir(parents=True,exist_ok=True)
    temp = directory/'bridge.pt.tmp'
    torch.save(bridge_checkpoint_state(bridge),temp)
    temp.replace(directory/'bridge.pt')
    config=bridge.metadata()
    if getattr(bridge,'initialize_for_grpo',None) is None:
        config.update(initialization=f'small random uniform +/-1e-4/sqrt({bridge.teacher.config.hidden_size})',constant_gate=1.)
    save(directory/'subspace_config.json',config)
    # One shared projection/optimizer copy; rank RNGs saved separately.
    torch.save(dict(optimizer=optimizer.state_dict(),step=step,initial_sha256=initial_sha),directory/'optimizer.pt')
    save(directory/'checkpoint.json',dict(step=step,bridge_sha256=hashlib.sha256((directory/'bridge.pt').read_bytes()).hexdigest(),
        bridge_state_sha256=weight_digest(bridge),initial_bridge_state_sha256=initial_sha))


def distributed_score(scorer,request,rank):
    """Propagate scorer failures before any rank waits for the next barrier."""
    summary=None
    failure=[None]
    if rank==0:
        try:
            summary=scorer.call(request)
        except Exception as error:
            failure[0]=f'{type(error).__name__}: {error}'
    dist.broadcast_object_list(failure,src=0)
    if failure[0] is not None:
        raise RuntimeError('Distributed scoring failed: '+failure[0])
    return summary


def _evaluate_once(bridge, engine, scorer, eval_rows, run_dir, step, rank, world,
                   tokenizer, eos, wandb_run, smoke=False, gate_mode=None):
    protocol=json.loads((run_dir.parent/'protocol.json').read_text())
    identity=[weight_digest(bridge) if rank==0 else None]
    dist.broadcast_object_list(identity,src=0)
    evaluation=run_dir/'evaluation'/f'step-{step:04d}'
    if gate_mode is not None:evaluation=evaluation/gate_mode
    chosen=eval_rows[:6] if smoke else eval_rows
    samples=2 if smoke else protocol['eval_samples']
    seed_stride=protocol.get('eval_seed_stride',4)
    requests=[dict(**r,sample_index=j,sampling_seed=1234+seed_stride*r['index']+j)
        for r in chosen[rank::world] for j in range(samples)]
    eval_started=time.monotonic()
    generate(bridge,engine,requests,identity[0],protocol['eval_sampling'],
        protocol['max_response_length'],evaluation/f'rank-{rank:02d}',tokenizer.pad_token_id,eos)
    dist.barrier()
    summary=distributed_score(scorer,dict(inputs=[str(evaluation/f'rank-{i:02d}/responses.json') for i in range(world)],
        expected_responses=len(chosen)*samples,result=str(evaluation/'summary.json')),rank)
    if rank==0:
        assert summary['gold_missing']==0 and summary['questions']==len(chosen)
        summary.update(step=step,samples_per_question=samples,wall_seconds=time.monotonic()-eval_started,
            bridge_state_sha256=identity[0],sampling=protocol['eval_sampling'],backend='native vLLM',
            validation_before_training=step==0)
        summary[f'avg_at_{samples}']=summary['accuracy']
        summary[f'pass_at_{samples}']=summary['pass_at_k']
        if gate_mode is not None:
            summary.update(gate_evaluation_mode=gate_mode,
                           gate_statistics=bridge.evaluation_gate_statistics())
        save(evaluation/'summary.json',summary)
        if protocol.get('model_family')=='llama31' and not smoke:
            from llama31_math_grpo import summarize_validation
            summarize_validation(run_dir)
        protocol=json.loads((run_dir.parent/'protocol.json').read_text())
        if protocol.get('save_evaluation_bridges',False):
            temporary=evaluation/'bridge.pt.tmp'
            torch.save(bridge_checkpoint_state(bridge),temporary)
            temporary.replace(evaluation/'bridge.pt')
            save(evaluation/'checkpoint.json',dict(step=step,bridge_state_sha256=identity[0],
                bridge_sha256=hashlib.sha256((evaluation/'bridge.pt').read_bytes()).hexdigest()))
            save(evaluation/'subspace_config.json',bridge.metadata())
        print(json.dumps(dict(evaluation=summary)),flush=True)
        if wandb_run:
            wandb_run.log({f'val/avg@{samples}':summary['accuracy'],f'val/pass@{samples}':summary['pass_at_k'],
                'val/clipping_rate':summary['clipping_rate'],'val/mean_tokens':summary['mean_tokens'],
                'val/seconds':summary['wall_seconds'],'training/global_step':step},step=step)
    dist.barrier()
    return summary if rank==0 else None


def evaluate(bridge, engine, scorer, eval_rows, run_dir, step, rank, world,
             tokenizer, eos, wandb_run, smoke=False):
    if not hasattr(bridge,'evaluation_gates'):
        return _evaluate_once(bridge,engine,scorer,eval_rows,run_dir,step,rank,world,
                              tokenizer,eos,wandb_run,smoke=smoke)
    identity=weight_digest(bridge)
    summaries={}
    for mode in ('soft','hard'):
        with bridge.evaluation_gates(mode):
            summaries[mode]=_evaluate_once(bridge,engine,scorer,eval_rows,run_dir,step,rank,world,
                tokenizer,eos,None,smoke=smoke,gate_mode=mode)
    assert weight_digest(bridge)==identity, 'Dual gate evaluation changed bridge weights'
    assert all(fuser.gate_evaluation_mode=='soft' for fuser in bridge.maps), 'GRPO must resume with soft gates'
    if rank==0:
        # Preserve the ordinary soft summary for existing completion/audit readers.
        summary=dict(summaries['soft'],gate_evaluations=summaries,
                     evaluation_gate_modes=['soft','hard'])
        save(run_dir/'evaluation'/f'step-{step:04d}'/'summary.json',summary)
        if wandb_run:
            metrics={'training/global_step':step}
            for mode,result in summaries.items():
                prefix=f'val/{mode}/'
                metrics.update({prefix+f'avg@{result["samples_per_question"]}':result['accuracy'],
                    prefix+f'pass@{result["samples_per_question"]}':result['pass_at_k'],
                    prefix+'clipping_rate':result['clipping_rate'],prefix+'mean_tokens':result['mean_tokens'],
                    prefix+'seconds':result['wall_seconds']})
            metrics.update({'val/avg@4':summary['accuracy'],'val/pass@4':summary['pass_at_k'],
                            'val/clipping_rate':summary['clipping_rate']})
            wandb_run.log(metrics,step=step)
        return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=BASE)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stress',action='store_true')
    parser.add_argument('--objective',choices=['opd','grpo'],default='opd')
    args=parser.parse_args()
    rank,world,local=(int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK'])
    protocol=json.loads((args.root/'protocol.json').read_text())
    validate_sampling(protocol)
    parallel_size=protocol.get('teacher_model_parallel_size',1)
    paired=parallel_size>1
    assert world==protocol['world_size']
    physical=list(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    assert physical==protocol['gpu_ids']
    assert len(physical)==world*parallel_size
    device_index=local*parallel_size
    teacher_devices=list(range(device_index,device_index+parallel_size))
    if paired:
        groups=protocol.get('teacher_gpu_groups',protocol.get('teacher_gpu_pairs'))
        assert [physical[i] for i in teacher_devices]==groups[local]
        assert physical[device_index]==protocol['decoder_gpu_ids'][local]
    torch.cuda.set_device(device_index)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(minutes=90),device_id=torch.device('cuda',device_index))
    assert protocol.get('method','opd')==args.objective
    is_grpo=args.objective=='grpo'
    if is_grpo:
        from qwen35_bridge_grpo_core import grpo_loss,group_advantages,sampled_log_probs,policy_loss
    run_dir=args.root/('smoke' if args.smoke else 'train')
    run_dir.mkdir(parents=True,exist_ok=True)
    if (run_dir/'metrics.jsonl').exists() or (run_dir/'complete.json').exists():
        raise RuntimeError('Fresh training refuses preexisting metrics/checkpoints')
    from transformers import AutoTokenizer
    if protocol.get('model_family')=='llama31':
        assert is_grpo
        from llama31_math_grpo import load_bridge
        bridge,runtime=load_bridge(args.root,protocol,teacher_devices)
    elif protocol.get('bridge_recipe')=='gsm8k_math_platour_grpo':
        from gsm8k_math_platour_lora_grpo import load_bridge
        bridge,runtime=load_bridge(args.root,protocol,teacher_devices)
    elif protocol.get('bridge_recipe')=='chemistry_dense_gaussian_platour_grpo':
        from chemistry_platour_grpo import load_bridge
        bridge,runtime=load_bridge(args.root,protocol)
    elif protocol.get('bridge_recipe')=='qwen235_17b_chemistry_hidden_grpo':
        from run_chemistry_235b_17b_hidden_grpo import load_bridge
        bridge,runtime=load_bridge(args.root,protocol,teacher_devices)
    elif protocol.get('bridge_recipe')=='qwen35_native_actor':
        assert is_grpo and not paired
        from qwen35_native_actor_grpo import load_bridge
        bridge,runtime=load_bridge(args.root,protocol)
    elif protocol.get('bridge_recipe')=='qwen35_template_embedding_tuning':
        assert is_grpo and not paired
        from qwen35_template_embedding_tuning import load_bridge
        bridge,runtime=load_bridge(args.root,protocol)
    elif protocol.get('bridge_recipe')=='qwen35_virtual_teacher':
        assert is_grpo and not paired
        from qwen35_virtual_teacher_bridge import load_bridge
        bridge,runtime=load_bridge(args.root,protocol)
    elif protocol.get('bridge_recipe')=='qwen35_template_virtual_teacher':
        assert is_grpo and not paired
        from qwen35_template_virtual_teacher_bridge import load_bridge
        bridge,runtime=load_bridge(args.root,protocol)
    elif protocol.get('teacher_family')=='gemma4':
        assert is_grpo, 'Cross-tokenizer Gemma transfer supports GRPO, not full-vocabulary OPD'
        from gemma4_qwen35_math_bridge import load_bridge
        bridge,runtime=load_bridge(args.root,protocol,teacher_devices)
    elif protocol.get('teacher_family')=='glm52':
        assert is_grpo, 'Cross-tokenizer GLM transfer supports GRPO, not full-vocabulary OPD'
        from glm52_qwen35_math_bridge import load_bridge
        bridge,runtime=load_bridge(args.root,protocol,teacher_devices)
    elif protocol.get('bridge_recipe')=='qwen3_4b_06b_hidden':
        assert not paired and is_grpo
        from qwen3_4b_06b_grpo import load_bridge
        bridge,runtime=load_bridge(args.root,protocol)
    elif protocol.get('model_family')=='qwen3':
        assert not paired
        from qwen3_30b_4b_math_grpo import load_bridge
        transfer_mode=protocol.get('transfer_mode','hidden')
        assert transfer_mode=='hidden' or is_grpo, 'KV residual transfer is a GRPO experiment'
        bridge,runtime=load_bridge(transfer_mode=transfer_mode)
    elif protocol.get('teacher')=='Qwen3.5-4B':
        assert not paired and is_grpo and protocol.get('layer_mapping')=='same_index'
        from qwen35_4b_4b_dapo_grpo import load_bridge
        bridge,runtime=load_bridge()
    elif paired:
        from qwen35_122b_4b_math_bridge import load_bridge
        bridge,runtime=load_bridge(ROOT,device_index,device_index+1,
                                  layer_mapping=protocol.get('layer_mapping','last_aligned'),
                                  teacher_precision=protocol.get('teacher_weight_precision','bf16'),
                                  teacher_name=protocol['teacher'],devices=teacher_devices,
                                  layer_split=protocol['teacher_layer_split'])
    else:
        from qwen35_35b_4b_math_bridge import load_bridge
        if protocol.get('layer_mapping')=='mse_per_student_shared_cosine_argmax':
            bridge,runtime=load_bridge(ROOT,layer_mapping=protocol['layer_mapping'],
                mapping=protocol['layer_map'],selection=protocol['mse_selection'])
        else:
            bridge,runtime=load_bridge(ROOT,layer_mapping=protocol.get('layer_mapping','last_aligned'))
    assert bridge.layer_map==protocol['layer_map'], 'Actual train/rollout/eval mapping must match protocol'
    if protocol.get('layer_mapping') in ('equal_width_layers','equal_width_full_to_full'):
        metadata=bridge.metadata()
        assert protocol['cross_attention_mapping_allowed'] and not protocol['group_constraint']
        assert metadata['attention_layer_pairs']==protocol['attention_layer_pairs']
        assert metadata['attention_types_preserved']==protocol['attention_types_preserved']
        assert sum(p.numel() for p in bridge.bridge_parameters())==protocol['trainable_parameters']
        if protocol['layer_mapping']=='equal_width_full_to_full':
            assert protocol['full_layer_constraint'] and metadata['nearest_full_tie_break']=='shallower'
            assert all(p['teacher_attention']=='full_attention' for p in metadata['attention_layer_pairs']
                       if p['student_attention']=='full_attention')
    if protocol.get('transfer_mode') in ('hidden_zero','hidden_mlp','hidden_mlp_w2zero','hidden_factorized_w2zero','hidden_random_standard','hidden_local_kv'):
        assert protocol['projection']==bridge.metadata()['projection']
        assert protocol['trainable_parameters']==sum(p.numel() for p in bridge.bridge_parameters())
    if protocol.get('transfer_mode')=='hidden_local_kv':
        assert protocol['kv_reconstruction']==bridge.metadata()['kv_reconstruction']
        assert protocol['native_prompt_residual_stream'] and not protocol['prefix_propagation']
    if protocol.get('transfer_mode')=='c2c_soft':
        metadata=bridge.metadata()
        assert protocol['fuser']==metadata['fuser']
        assert protocol['gate_mode']==metadata['gate_mode']
        assert protocol['gumbel']==metadata['gumbel']==False
        assert protocol['dropout']==metadata['dropout']==0.
        assert protocol['trainable_parameters']==sum(p.numel() for p in bridge.bridge_parameters())
    bridge.student_activation_checkpointing=protocol.get('student_activation_checkpointing',False)
    runtime['student_activation_checkpointing']=bridge.student_activation_checkpointing
    if protocol.get('teacher_weight_precision')=='fp8':
        # Audit the real quantized encoder before both smoke and formal work.
        # All ranks exercise both halves of the paired model on a real prompt.
        audit_row=json.loads((args.root/'train_prompts.json').read_text())[0]
        if protocol.get('bridge_recipe') in ('qwen235_17b_chemistry_hidden_grpo','gsm8k_math_platour_grpo'):
            ids=torch.tensor([audit_row['prompt_token_ids']],device=bridge.student.device)
            with torch.no_grad():features=bridge.features(ids,torch.ones_like(ids))
            hidden=features
            audit_prompt_tokens=ids.shape[1]
        elif protocol.get('teacher_family')=='glm52':
            # Sender must use its own prepared tokenizer, never receiver IDs.
            features,_=bridge._features(audit_row['prompt_token_ids'])
            hidden=features
            audit_prompt_tokens=len(bridge._alignment(audit_row['prompt_token_ids'])['teacher_prompt_ids'])
        else:
            ids=torch.tensor([audit_row['prompt_token_ids']],device=bridge.student.device)
            with torch.no_grad():
                hidden=bridge.teacher.model(input_ids=ids,attention_mask=torch.ones_like(ids),
                    use_cache=False,output_hidden_states=True).hidden_states
                features=bridge.teacher_features(hidden)
            audit_prompt_tokens=ids.shape[1]
        assert len(features)==bridge.student.config.num_hidden_layers and all(h.dtype==torch.bfloat16 for h in features)
        assert all(h.device==bridge.student.device and torch.isfinite(h).all() for h in features)
        save(run_dir/f'fp8_hidden_audit.rank-{rank:02d}.json',dict(passed=True,
            hidden_dtypes=sorted({str(h.dtype) for h in hidden}),
            transfer_dtype='bfloat16',selected_layers=bridge.layer_map,finite=True,
            prompt_tokens=audit_prompt_tokens,frozen_teacher=True))
        del hidden,features
        if protocol.get('teacher_family')!='glm52':
            del ids
    resume_step=0 if args.smoke else protocol.get('resume_step',0)
    resume_directory=Path(protocol['resume_from_checkpoint']) if resume_step else None
    if resume_step:
        assert protocol['bridge_recipe']=='qwen235_17b_chemistry_hidden_grpo'
        assert resume_step%protocol['steps_per_epoch']==0,'Continuation starts at an epoch boundary'
        state=torch.load(resume_directory/'bridge.pt',map_location='cpu',weights_only=True,mmap=True)
        bridge.maps.load_state_dict(state['P']);del state
        init=dict(kind='Resume actual bridge and AdamW state',resume_step=resume_step,
            checkpoint=str(resume_directory),optimizer_resumed=True,bridge_reinitialized=False)
    else:
        init=initialize(bridge)
    params=bridge.bridge_parameters()
    assert all(p.requires_grad and (p.dtype==torch.float32 or
        (getattr(bridge,'trainable_actor',False) and p.dtype==torch.bfloat16)) for p in params)
    initial_sha=weight_digest(bridge)
    if resume_step:
        assert initial_sha==protocol['resume_bridge_state_sha256'],'Resumed bridge differs from parent final weights'
    if protocol.get('expected_initial_bridge_state_sha256'):
        assert initial_sha==protocol['expected_initial_bridge_state_sha256'], 'Initialization differs from the GRPO baseline'
    initial_hashes=[None]*world
    dist.all_gather_object(initial_hashes,initial_sha)
    assert len(set(initial_hashes))==1
    learning_rate=protocol.get('learning_rate',1e-5)
    optimizer=(bridge.make_optimizer(learning_rate) if hasattr(bridge,'make_optimizer') else
        torch.optim.AdamW(params,lr=learning_rate,weight_decay=0.,betas=(.9,.999)))
    if resume_step:
        saved=torch.load(resume_directory/'optimizer.pt',map_location='cpu',weights_only=True,mmap=True)
        assert saved['step']==resume_step
        optimizer.load_state_dict(saved['optimizer']);del saved
        assert len(optimizer.state)==len(params)
        assert all(int(s['step'])==resume_step for s in optimizer.state.values())
        assert all(g['lr']==learning_rate for g in optimizer.param_groups)
        save(run_dir/f'resume.rank-{rank:02d}.json',dict(passed=True,step=resume_step,
            bridge_state_sha256=initial_sha,optimizer_step=resume_step,
            optimizer_tensors=len(optimizer.state),weights_restored=True,optimizer_restored=True))
    student_path=Path(protocol.get('student_checkpoint',ROOT/'data/models/Qwen3.5-4B'))
    tokenizer=AutoTokenizer.from_pretrained(student_path,local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id=tokenizer.eos_token_id
    eos=bridge.student.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else list(eos)
    train_rows=json.loads((args.root/'train_prompts.json').read_text())
    eval_rows=json.loads((args.root/'eval_prompts.json').read_text())
    examples,epochs=protocol['examples_per_epoch'],protocol['epochs']
    prompt_cap,response_cap=protocol['max_prompt_length'],protocol['max_response_length']
    assert len(train_rows)==examples and [r['index'] for r in train_rows]==list(range(examples))
    assert len(eval_rows)==protocol['eval_questions']
    assert bridge.student.lm_head.weight.shape[0]==protocol.get('vocab_size',248320)
    if (protocol.get('bridge_recipe') in ('qwen35_virtual_teacher','qwen35_template_virtual_teacher','qwen35_template_embedding_tuning','qwen35_native_actor','llama31_lora') or
        (protocol.get('bridge_recipe') in ('chemistry_dense_gaussian_platour_grpo','gsm8k_math_platour_grpo') and not protocol['teacher_model_loaded'])):
        assert not hasattr(bridge,'teacher') and not protocol['teacher_model_loaded']
        assert sum(p.numel() for p in params)==protocol['trainable_parameters']
    elif protocol.get('teacher_family') in ('gemma4','glm52'):
        assert is_grpo and protocol['same_vocabulary']==False
        assert bridge.teacher.lm_head.weight.shape[0]==protocol['teacher_vocab_size']
    else:
        assert bridge.teacher.lm_head.weight.shape[0]==bridge.student.lm_head.weight.shape[0]
    if rank==0:
        checkpoint(bridge,optimizer,run_dir/'initial',resume_step,initial_sha)
        save(run_dir/'initialization.json',dict(init,bridge_state_sha256=initial_sha,
            replica_hashes=initial_hashes,layer_map=bridge.layer_map))
        save(run_dir/'runtime.json',runtime)
    dist.barrier()
    engine=scorer=wandb_run=None
    try:
        torch.cuda.empty_cache()
        engine=RPC(protocol.get('vllm_worker_script','qwen35_bridge_opd_vllm_worker.py'),run_dir/f'workers/rank-{rank:02d}/vllm',
                   gpu=physical[device_index],memory_fraction=protocol['vllm_memory_fraction'],
                   max_model_len=protocol.get('vllm_max_model_len',prompt_cap+response_cap),
                   max_num_seqs=protocol.get('vllm_max_num_seqs',32),
                    decode_batch_size=protocol.get('vllm_decode_batch_size',32),model_path=student_path,
                    generation_config='vllm' if protocol.get('native_vllm_first_token',False) else None)
        if rank==0:
            scorer=RPC(protocol.get('score_worker_script','qwen35_bridge_opd_score_worker.py'),run_dir/'score_worker',
                       boxed_only=protocol.get('math_verify_boxed_only',True))
        dist.barrier()
        if args.smoke and protocol.get('bridge_recipe') in ('qwen35_native_actor','llama31_lora'):
            if rank==0:
                from qwen35_native_actor_grpo import smoke_audit
                smoke_audit(bridge,engine,train_rows,run_dir/'native_actor_audit',tokenizer.pad_token_id,eos)
            dist.barrier()
        if args.smoke and protocol.get('bridge_recipe')=='qwen35_template_embedding_tuning':
            if rank==0:
                from qwen35_template_embedding_tuning import smoke_audit
                smoke_audit(bridge,engine,train_rows,run_dir/'template_embedding_audit',tokenizer.pad_token_id,eos)
            dist.barrier()
        if args.smoke and protocol.get('bridge_recipe')=='qwen35_virtual_teacher':
            if rank==0:
                from qwen35_virtual_teacher_bridge import smoke_audit
                smoke_audit(bridge,engine,train_rows,run_dir/'virtual_teacher_audit',tokenizer.pad_token_id,eos)
            dist.barrier()
        if args.smoke and protocol.get('bridge_recipe')=='qwen35_template_virtual_teacher':
            if rank==0:
                from qwen35_template_virtual_teacher_bridge import smoke_audit
                smoke_audit(bridge,engine,train_rows,run_dir/'template_virtual_teacher_audit',tokenizer.pad_token_id,eos)
            dist.barrier()
        if args.smoke and protocol.get('bridge_recipe')=='llama31_hidden':
            if rank==0:
                from qwen3_30b_4b_math_grpo import verify_vllm_handoff
                verify_vllm_handoff(bridge,engine,train_rows,run_dir/'vllm_handoff_parity',tokenizer.pad_token_id,eos)
            dist.barrier()
        if args.smoke and protocol.get('bridge_recipe') in ('chemistry_dense_gaussian_platour_grpo','gsm8k_math_platour_grpo'):
            if rank==0:
                from chemistry_platour_grpo import smoke_audit
                smoke_audit(bridge,engine,train_rows,run_dir/'platour_audit',tokenizer.pad_token_id)
            dist.barrier()
        if args.smoke and protocol.get('bridge_recipe')=='qwen235_17b_chemistry_hidden_grpo':
            if rank==0:
                from run_chemistry_235b_17b_hidden_grpo import verify_handoff
                verify_handoff(bridge,engine,train_rows,run_dir/'vllm_handoff_parity',tokenizer.pad_token_id)
            dist.barrier()
        if args.smoke and protocol.get('model_family')=='qwen3' and protocol.get('bridge_recipe') not in ('qwen35_native_actor','qwen235_17b_chemistry_hidden_grpo','chemistry_dense_gaussian_platour_grpo','gsm8k_math_platour_grpo'):
            if rank==0:
                if protocol.get('bridge_recipe')=='qwen3_4b_06b_hidden':
                    from qwen3_4b_06b_grpo import verify_vllm_handoff
                    verify_vllm_handoff(bridge,engine,train_rows,run_dir/'vllm_handoff_parity',tokenizer.pad_token_id,eos)
                elif protocol.get('transfer_mode')=='hidden_local_kv':
                    from qwen3_30b_4b_math_grpo import verify_local_hidden_isolation
                    verify_local_hidden_isolation(bridge,train_rows,run_dir,tokenizer.pad_token_id)
                if protocol.get('bridge_recipe')!='qwen3_4b_06b_hidden':
                    from qwen3_30b_4b_math_grpo import verify_vllm_handoff
                    verify_vllm_handoff(bridge,engine,train_rows,run_dir/'vllm_handoff_parity',tokenizer.pad_token_id,eos)
            dist.barrier()
        if args.stress:
            if hasattr(bridge,'prepare_optimizer_for_stress'):
                bridge.prepare_optimizer_for_stress(optimizer)
            # Exact maximum active lengths; no padding, with the resident vLLM
            # allocation present. Do not update W using this synthetic batch.
            if rank==0:
                source=max(train_rows,key=lambda r:len(r['prompt_token_ids']))['prompt_token_ids']
                prompt=(source*((prompt_cap+len(source)-1)//len(source)))[:prompt_cap]
                if hasattr(bridge,'build_stress_prompt'):
                    prompt=bridge.build_stress_prompt(source,prompt_cap)
                response=(source*((response_cap+len(source)-1)//len(source)))[:response_cap]
                teacher_stress_tokens=None
                if getattr(bridge,'register_stress_prompt',None) is not None:
                    teacher_stress_tokens=bridge.register_stress_prompt(prompt)
                torch.cuda.reset_peak_memory_stats()
                before=time.monotonic()
                if is_grpo:
                    log_probs=sampled_log_probs(bridge,prompt,response)
                    value,_=policy_loss(log_probs,1.,log_probs.detach())
                else:
                    value=opd_loss(bridge,prompt,response,chunk_tokens=protocol.get('kl_chunk_tokens',32))
                value.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                assert any(p.grad is not None and p.grad.norm()>0 for p in params)
                save(run_dir/'worst_case_profile.json',dict(passed=True,prompt_tokens=prompt_cap,response_tokens=response_cap,
                    objective=args.objective,
                    teacher_prompt_tokens=teacher_stress_tokens,
                    active_input_tokens=prompt_cap+response_cap-1,padding_tokens=0,micro_batch=1,loss=float(value),grad_norm=float(norm),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    gpu_total_used_gib=(torch.cuda.get_device_properties(device_index).total_memory-torch.cuda.mem_get_info()[0])/2**30,
                    gpu_pair_used_gib={str(i):(torch.cuda.get_device_properties(i).total_memory-torch.cuda.mem_get_info(i)[0])/2**30
                        for i in teacher_devices},
                    seconds=time.monotonic()-before,vllm_resident=True,
                    optimizer_state_resident=bool(optimizer.state)))
                optimizer.zero_grad(set_to_none=True)
                del value
                torch.cuda.empty_cache()
            dist.barrier()
        if not args.smoke and rank==0:
            import wandb
            wandb_run=wandb.init(project=protocol.get('wandb_project','hidden-bridge-math-grpo' if is_grpo else 'hidden-bridge-math-opd'),
                name=protocol.get('wandb_run_name',
                    f'qwen35_35b_to_4b_constant_gate1_grpo_g8_bs64_4k_lr1e5_3epoch_eval13_vllm_{len(physical)}gpu' if is_grpo else
                    f'qwen35_35b_to_4b_constant_gate1_opd_g1_bs128_4k_lr1e5_3epoch_eval12_vllm_{len(physical)}gpu'),
                config=protocol,dir=str(run_dir),mode=os.environ.get('WANDB_MODE','online'))
            assert wandb_run.offline==(os.environ.get('WANDB_MODE','online')=='offline')
            save(run_dir/'wandb.json',dict(id=wandb_run.id,url=wandb_run.url,entity=wandb_run.entity,project=wandb_run.project))
            wandb_run.log({'bridge/initial_weight_norm':math.sqrt(sum(float(p.detach().square().sum()) for p in params))},step=resume_step,commit=False)
        dist.barrier()
        if not args.smoke and protocol['validation_before_training']:
            if rank==0:
                save(run_dir/'status.json',dict(phase='initial_eval',step=0,time=time.time()))
            cpu_rng,cuda_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
            evaluate(bridge,engine,scorer,eval_rows,run_dir,0,rank,world,tokenizer,eos,wandb_run)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            assert weight_digest(bridge)==initial_sha and not optimizer.state
            assert all(p.grad is None for p in params)
            if rank==0:
                save(run_dir/'initial_eval_audit.json',dict(passed=True,step=0,
                    initial_bridge_state_sha256=initial_sha,weights_unchanged=True,
                    optimizer_updates=0,rng_restored=True,questions=len(eval_rows),samples_per_question=protocol['eval_samples']))
                save(run_dir/'status.json',dict(phase='train',step=0,time=time.time()))
            dist.barrier()
        if resume_step:
            rng=torch.load(resume_directory/f'rng.rank-{rank:02d}.pt',map_location='cpu',weights_only=True)
            torch.set_rng_state(rng['cpu_rng']);torch.cuda.set_rng_state(rng['cuda_rng'])
            assert torch.equal(torch.get_rng_state(),rng['cpu_rng'])
            assert torch.equal(torch.cuda.get_rng_state(),rng['cuda_rng'])
            save(run_dir/f'resume_rng.rank-{rank:02d}.json',dict(passed=True,step=resume_step,
                cpu_rng_restored=True,cuda_rng_restored=True,first_update=resume_step+1,
                first_epoch=resume_step//protocol['steps_per_epoch']+1))
            if rank==0:save(run_dir/'status.json',dict(phase='train',step=resume_step,time=time.time()))
        started=time.monotonic()
        steps=2 if args.smoke else protocol['total_steps']
        batch_size,group_size,epoch_steps=(protocol[k] for k in ['batch_size','group_size','steps_per_epoch'])
        if not args.smoke and protocol.get('max_train_steps') is not None:
            assert steps==protocol['max_train_steps'] and 1<=steps<=epoch_steps*epochs
        orders=[torch.randperm(examples,generator=torch.Generator().manual_seed(42+epoch)).tolist() for epoch in range(epochs)]
        visited=[]
        smoke_selection=sorted(range(examples),key=lambda i:len(train_rows[i]['prompt_token_ids']))[30:46]
        if is_grpo:
            ordered=sorted(range(examples),key=lambda i:len(train_rows[i]['prompt_token_ids']))
            smoke_selection=[ordered[(j+1)*len(ordered)//25] for j in range(24)]
        if args.smoke and protocol.get('smoke_full_batch',False):
            ordered=sorted(range(examples),key=lambda i:len(train_rows[i]['prompt_token_ids']))
            smoke_selection=[ordered[(j+1)*len(ordered)//(batch_size+1)] for j in range(batch_size)]
        nonzero_updates=0
        for step in range(resume_step+1,steps+1):
            tick=time.monotonic()
            epoch=(step-1)//epoch_steps
            offset=((step-1)%epoch_steps)*batch_size
            selection=smoke_selection if args.smoke else orders[epoch][offset:offset+batch_size]
            if args.smoke and step==2:
                selection=selection[:1]  # Uneven singleton tail: all other ranks are empty.
            rows=rollout_rows(train_rows,selection,rank,world,group_size,epoch,step)
            identity=[weight_digest(bridge) if rank==0 else None]
            dist.broadcast_object_list(identity,src=0)
            digest=identity[0]
            path=run_dir/'rollouts'/f'step-{step:04d}'/f'rank-{rank:02d}'
            rollout,export_seconds=generate(bridge,engine,rows,digest,TRAIN_SAMPLING,
                response_cap,path,tokenizer.pad_token_id,eos,return_logprobs=is_grpo)
            if is_grpo:
                dist.barrier()
                directory=run_dir/'rollouts'/f'step-{step:04d}'
                summary=distributed_score(scorer,dict(inputs=[str(directory/f'rank-{i:02d}/responses.json') for i in range(world)],
                    expected_responses=len(selection)*group_size,result=str(directory/'score_summary.json'),
                    format_penalty=bool(protocol.get('format_penalty',False)),
                    reward_answer_format=protocol.get('reward_answer_format','boxed')),rank)
                dist.barrier()
                scored=[json.loads(line) for line in (directory/'scored.jsonl').open()]
                if protocol.get('train_reward_policy')=='zero_for_length_limit':
                    from llama31_math_grpo import training_rewards
                    scored=training_rewards(scored)
                advantages,group_stats=group_advantages(scored,group_size)
                if rank==0:
                    save(directory/'advantages.json',dict(group_stats=group_stats,rows=[dict(index=r['index'],
                        sample_index=r['sample_index'],reward=r.get('reward',r['acc']),answer_acc=r['acc'],
                        advantage=advantages[(r['index'],r['sample_index'])]) for r in scored]))
            count=torch.tensor(sum(r['tokens'] for r in rollout['rows']),device='cuda',dtype=torch.long)
            dist.all_reduce(count)
            total_tokens=int(count)
            assert total_tokens>0
            optimizer.zero_grad(set_to_none=True)
            loss_sum=torch.zeros((),device='cuda',dtype=torch.float64)
            pg_metrics=torch.zeros(5,device='cuda',dtype=torch.float64)
            update_started=time.monotonic()
            for row in rollout['rows']:
                if is_grpo:
                    advantage=advantages[(row['index'],row['sample_index'])]
                    # Homogeneous or missing-gold groups have identically zero
                    # policy gradient; skip their costly model pass exactly.
                    if advantage==0.:
                        continue
                    value,metrics=grpo_loss(bridge,row['prompt_token_ids'],row['response_token_ids'],
                        advantage,row['rollout_log_probs'])
                else:
                    value=opd_loss(bridge,row['prompt_token_ids'],row['response_token_ids'],
                                   chunk_tokens=protocol.get('kl_chunk_tokens',32))
                assert torch.isfinite(value)
                if is_grpo:
                    for i,key in enumerate(['pg_clip_fraction','ppo_kl','rollout_is_mean',
                                           'rollout_is_clipped_fraction','actor_rollout_logprob_abs_error']):
                        pg_metrics[i]+=metrics[key].double()*row['tokens']
                    loss_sum+=value.detach().double()
                    (value/(len(selection)*group_size)).backward()
                else:
                    loss_sum+=value.detach().double()*row['tokens']
                    (value*(row['tokens']/total_tokens)).backward()
                del value
            for p in params:
                if p.grad is None:
                    p.grad=torch.zeros_like(p)
                dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_sum)
            if is_grpo:
                dist.all_reduce(pg_metrics)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            if hasattr(bridge,'assert_trainable_parameters'):
                bridge.assert_trainable_parameters()
            else:
                frozen_models=([bridge.teacher,bridge.student] if hasattr(bridge,'teacher') else [bridge.student])
                assert all(p.grad is None and not p.requires_grad for model in frozen_models for p in model.parameters())
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            assert norm>=0 if is_grpo else norm>0
            nonzero_updates+=int(norm>0)
            optimizer.step()
            if args.smoke:
                # Verify actual tensors, not only aggregate weight norms.
                for p in params:
                    expected=p.detach().clone()
                    dist.broadcast(expected,src=0)
                    assert torch.equal(p,expected)
                if norm>0:
                    assert weight_digest(bridge)!=digest
            timing=torch.tensor([export_seconds,rollout['seconds'],time.monotonic()-update_started,
                torch.cuda.max_memory_allocated()/2**30],device='cuda',dtype=torch.float64)
            dist.all_reduce(timing,op=dist.ReduceOp.MAX)
            dist.barrier()
            if rank==0:
                directory=run_dir/'rollouts'/f'step-{step:04d}'
                if not is_grpo:
                    summary=scorer.call(dict(inputs=[str(directory/f'rank-{i:02d}/responses.json') for i in range(world)],
                        expected_responses=len(selection)*group_size,result=str(directory/'score_summary.json')))
                record=dict(step=step,epoch=epoch+1,total_steps=steps,examples=len(selection),reply_tokens=total_tokens,
                    learning_rate=learning_rate,grad_norm=float(norm),
                    train_accuracy=summary['accuracy'],clipping_rate=summary['clipping_rate'],mean_tokens=summary['mean_tokens'],
                    hf_export_seconds=float(timing[0]),vllm_decode_seconds=float(timing[1]),update_seconds=float(timing[2]),
                    step_seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started,
                    peak_allocated_gib=float(timing[3]),world_size=world,
                    rollout_bridge_state_sha256=digest,training_generation='native vLLM')
                if hasattr(bridge,'training_statistics'):
                    record.update(bridge.training_statistics())
                if is_grpo and protocol.get('format_penalty',False):
                    record.update(train_accuracy_unpenalized=summary['accuracy_unpenalized'],
                        format_violation_rate=summary['format_violation_rate'],format_zeroed_rate=summary['format_zeroed_rate'])
                if is_grpo:
                    if protocol.get('train_reward_policy')=='zero_for_length_limit':
                        eligible=[r for r in scored if not r['gold_missing']]
                        record['train_reward_mean']=sum(r['reward'] for r in eligible)/len(eligible)
                    active_tokens=sum(r['tokens'] for r in scored if advantages[(r['index'],r['sample_index'])]!=0.)
                    record.update(policy_loss=float(loss_sum/(len(selection)*group_size)),responses=len(selection)*group_size,
                        group_size=group_size,**group_stats,active_reply_tokens=active_tokens,
                        **{key:float(pg_metrics[i]/max(active_tokens,1)) for i,key in enumerate(
                            ['pg_clip_fraction','ppo_kl','rollout_is_mean','rollout_is_clipped_fraction','actor_rollout_logprob_abs_error'])})
                else:
                    record.update(reverse_kl=float(loss_sum/total_tokens),
                        responses=len(selection)*group_size,group_size=group_size)
                with (run_dir/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
                save(run_dir/'live.json',record)
                print(json.dumps(record),flush=True)
                if wandb_run:
                    objective_metrics=({'train/policy_loss':record['policy_loss'],
                        'train/mixed_group_fraction':group_stats['mixed_groups']/len(selection),
                        **{f'actor/{key}':record[key] for key in ['pg_clip_fraction','ppo_kl','rollout_is_mean',
                            'rollout_is_clipped_fraction','actor_rollout_logprob_abs_error']}}
                        if is_grpo else {'train/reverse_kl':record['reverse_kl']})
                    if is_grpo and protocol.get('format_penalty',False):
                        objective_metrics.update({'critic/score_unpenalized/mean':summary['accuracy_unpenalized'],
                            'reward/format_violation_rate':summary['format_violation_rate'],
                            'reward/format_zeroed_rate':summary['format_zeroed_rate']})
                    wandb_run.log({**objective_metrics,'actor/lr':learning_rate,
                        'actor/grad_norm':record['grad_norm'],'critic/score/mean':summary['accuracy'],
                        'response_length/clip_ratio':summary['clipping_rate'],'response_length/mean':summary['mean_tokens'],
                        'timing/step_seconds':record['step_seconds'],'timing/vllm_decode_seconds':float(timing[1]),
                        'timing/update_seconds':float(timing[2]),'training/global_step':step,
                         'training/epoch':epoch+1,'bridge/peak_allocated_gib':float(timing[3]),
                         **{f'bridge/{k}':v for k,v in record.items()
                            if k.startswith(('virtual_hidden_','projection_','effective_delta_','template_embedding_'))}},step=step)
            visited.extend(selection)
            dist.barrier()
            eval_now=step==steps if args.smoke else step in protocol['eval_steps']
            if eval_now:
                evaluate(bridge,engine,scorer,eval_rows,run_dir,step,rank,world,
                    tokenizer,eos,wandb_run,smoke=args.smoke)
            checkpoint_steps=protocol.get('checkpoint_steps')
            save_now=(step in checkpoint_steps if checkpoint_steps is not None else step%epoch_steps==0)
            recovery_every=protocol.get('recovery_checkpoint_every',0)
            save_now=save_now or bool(recovery_every and step%recovery_every==0)
            if (not args.smoke and save_now) or (args.smoke and step==steps):
                destination=run_dir/'checkpoints'/f'step-{step:04d}' if checkpoint_steps is not None else run_dir/f'epoch-{epoch+1}'
                if rank==0:
                    checkpoint(bridge,optimizer,destination,step,initial_sha)
                dist.barrier()
                torch.save(dict(cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),
                    destination/f'rng.rank-{rank:02d}.pt')
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        if not args.smoke:
            expected_visited=[i for step_index in range(resume_step,steps)
                for i in orders[step_index//epoch_steps][(step_index%epoch_steps)*batch_size:
                                                          (step_index%epoch_steps+1)*batch_size]]
            assert visited==expected_visited
            if protocol.get('max_train_steps') is None:
                assert len(visited)==examples*(epochs-resume_step//epoch_steps)
        hashes=[None]*world
        dist.all_gather_object(hashes,weight_digest(bridge))
        assert len(set(hashes))==1
        if is_grpo:
            assert nonzero_updates>0 and hashes[0]!=initial_sha
        if rank==0:
            evals=list((run_dir/'evaluation').glob('step-*/summary.json'))
            assert len(evals)==(1 if args.smoke else protocol['total_evaluations'])
            if not args.smoke and protocol.get('checkpoint_steps') is not None:
                assert all((run_dir/'checkpoints'/f'step-{s:04d}'/'optimizer.pt').exists()
                           and all((run_dir/'checkpoints'/f'step-{s:04d}'/f'rng.rank-{r:02d}.pt').exists()
                                   for r in range(world)) for s in protocol['checkpoint_steps'])
            save(run_dir/'complete.json',dict(complete=True,smoke=args.smoke,steps=steps,
                objective=args.objective,group_size=group_size,nonzero_gradient_updates=nonzero_updates,
                examples_seen=len(visited),epochs=1 if args.smoke else epochs-resume_step//epoch_steps,evals=len(evals),
                resumed_from_step=resume_step,updates_this_run=steps-resume_step,total_training_epochs=epochs,
                six_rank_exact_weights=world==6,eight_rank_exact_weights=world==8,
                replica_exact_weights=True,world_size=world,
                gpu_ids=physical,teacher=protocol['teacher'],final_bridge_state_sha256=hashes[0],
                initial_bridge_state_sha256=initial_sha,seconds=time.monotonic()-started,
                rollout_and_eval_backend='native vLLM',frozen_backbones_verified=not getattr(bridge,'trainable_actor',False),
                trainable_actor=getattr(bridge,'trainable_actor',False),
                singleton_tail_verified=args.smoke))
            if wandb_run:
                wandb_run.finish()
        dist.barrier()
    except BaseException as error:
        save(run_dir/f'failure.rank-{rank:02d}.json',dict(error=str(error),traceback=traceback.format_exc()))
        if wandb_run:
            wandb_run.finish(exit_code=1)
        raise
    finally:
        if engine:
            engine.close()
        if scorer:
            scorer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__=='__main__':
    main()
