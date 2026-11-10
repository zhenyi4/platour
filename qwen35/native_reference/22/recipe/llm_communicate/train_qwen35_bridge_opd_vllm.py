"""Distributed native vLLM on-policy rollouts and differentiable HF bridge updates."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import torch
import torch.distributed as dist

from qwen35_bridge_opd_core import (ROOT, BASE, TRAIN_SAMPLING, EVAL_SAMPLING,
    opd_loss, weight_digest, first_tokens, save, early_stop_update)
from random_config import initialize_zero as initialize, rank_layout, verify_sources
from prepare_qwen35_bridge_vllm import cache_to_cpu, export_memory
from eval_qwen35_35b_4b_math_bridge_8gpu import prefill, prompt_tensors


class LocalRPC:
    def __init__(self, script, directory, gpu=None, memory_fraction=.12, max_model_len=6144, boxed_only=True,
                 max_num_seqs=32, decode_batch_size=0):
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
        # FlashInfer JIT (vLLM GDN prefill on sm90) invokes the venv's ninja by name.
        env['PATH'] = str(python.parent)+os.pathsep+env.get('PATH','')
        cmd = [str(python), '-u', str(ROOT/'recipe/llm_communicate'/script),
               '--directory', str(self.directory)]
        if gpu is not None:
            cmd += ['--memory-fraction', str(memory_fraction), '--max-model-len', str(max_model_len),
                    '--max-num-seqs', str(max_num_seqs), '--decode-batch-size', str(decode_batch_size)]
        self.log = (self.directory/'worker.log').open('a')
        self.proc = subprocess.Popen(cmd, env=env, cwd=ROOT, stdin=subprocess.PIPE,
            stdout=self.log, stderr=subprocess.STDOUT, text=True)
        self.wait(self.directory/'ready.json', timeout=3600)

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


def RPC(script, directory, gpu=None, **kwargs):
    if gpu is None and script=='lcb_score_worker.py':
        from shared_code_judge_client import SharedCodeScorer
        return SharedCodeScorer(directory)
    return LocalRPC(script,directory,gpu=gpu,**kwargs)


def generate(bridge, engine, rows, digest, sampling, cap, directory, pad, eos, return_logprobs=False):
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


def checkpoint(bridge, optimizer, directory, step, initial_sha):
    directory.mkdir(parents=True,exist_ok=True)
    state = bridge.bridge_checkpoint_state()
    temp = directory/'bridge.pt.tmp'
    torch.save(state,temp)
    temp.replace(directory/'bridge.pt')
    save(directory/'subspace_config.json',dict(bridge.metadata(),
        checkpoint_initialization='native4B train-only Gaussian V / zero W',constant_gate=1.))
    # One shared projection/optimizer copy; rank RNGs saved separately.
    torch.save(dict(optimizer=optimizer.state_dict(),step=step,initial_sha256=initial_sha),directory/'optimizer.pt')
    save(directory/'checkpoint.json',dict(step=step,bridge_sha256=hashlib.sha256((directory/'bridge.pt').read_bytes()).hexdigest(),
        bridge_state_sha256=weight_digest(bridge),initial_bridge_state_sha256=initial_sha))


GRPO_ACTOR_KEYS=['pg_clip_fraction','ppo_kl','rollout_is_mean','rollout_is_clipped_fraction','actor_rollout_logprob_abs_error']


def train_payload(record, protocol, is_grpo):
    """W&B metrics of one training step, built only from its metrics.jsonl record (also used to replay history)."""
    payload={'actor/lr':record['learning_rate'],'actor/grad_norm':record['grad_norm'],
        'critic/score/mean':record['train_accuracy'],'response_length/clip_ratio':record['clipping_rate'],
        'response_length/mean':record['mean_tokens'],'timing/step_seconds':record['step_seconds'],
        'timing/vllm_decode_seconds':record['vllm_decode_seconds'],'timing/update_seconds':record['update_seconds'],
        'training/global_step':record['step'],'training/epoch':record['epoch'],
        'bridge/peak_allocated_gib':record['peak_allocated_gib']}
    if is_grpo:
        payload.update({'train/policy_loss':record['policy_loss'],
            'train/mixed_group_fraction':record['mixed_groups']/record['examples'],
            **{f'actor/{key}':record[key] for key in GRPO_ACTOR_KEYS}})
    else:
        payload['train/reverse_kl']=record['reverse_kl']
    if protocol.get('format_penalty') and 'accuracy_unpenalized' in record:
        payload.update({'critic/score_unpenalized/mean':record['accuracy_unpenalized'],
            'reward/format_violation_rate':record['format_violation_rate'],
            'reward/format_zeroed_rate':record['format_zeroed_rate']})
    return payload


def val_payload(summary, step):
    """W&B metrics of one validation. With val_metric='answer', val/avg@4 is the corrected metric."""
    samples=summary.get('samples_per_question',4)
    payload={f'val/avg@{samples}':summary['accuracy'],f'val/pass@{samples}':summary['pass_at_k'],
        'val/clipping_rate':summary['clipping_rate'],'val/mean_tokens':summary['mean_tokens'],
        'training/global_step':step}
    if 'wall_seconds' in summary:
        payload['val/seconds']=summary['wall_seconds']
    if summary.get('metric')=='answer':
        payload.update({'val/avg@4_full':summary['accuracy_full'],'val/pass@4_full':summary['pass_at_k_full'],
            'val/avg@4_strict':summary['accuracy_strict'],'val/pass@4_strict':summary['pass_at_k_strict'],
            'val/format_violation_rate':summary['format_violation_rate']})
    elif summary.get('answer_metrics'):
        payload.update({'val/avg@4_answer':summary['accuracy_answer'],'val/pass@4_answer':summary['pass_at_k_answer'],
            'val/avg@4_strict':summary['accuracy_strict'],'val/pass@4_strict':summary['pass_at_k_strict']})
    return payload


def early_payload(state):
    return {'early_stop/fails':state['fails'],'early_stop/best_step':state['best_step'],
        f'early_stop/best_avg@{state.get("samples_per_question",4)}':state['best_accuracy']}


def checkpoint_complete(directory, world):
    return all((directory/name).exists() for name in ('checkpoint.json','bridge.pt','optimizer.pt')) and \
        all((directory/f'rng.rank-{r:02d}.pt').exists() for r in range(world))


def evaluation_summaries(run_dir, upto):
    found={}
    for path in (run_dir/'evaluation').glob('step-*/summary.json'):
        step=int(path.parent.name.split('-')[1])
        if step<=upto:
            found[step]=json.loads(path.read_text())
    return dict(sorted(found.items()))


def replay_early_stop(run_dir, protocol, upto):
    """Recompute the early-stop state from the saved validations up to `upto` (pure; same rule as training)."""
    early=protocol.get('early_stop')
    state=reason=None
    snapshots={}
    if not early:
        return state,reason,snapshots
    for step,summary in evaluation_summaries(run_dir,upto).items():
        if step==0:
            state,_=early_stop_update(None,0,summary['accuracy'],summary['clipping_rate'],
                dict(early,patience=10**9,max_steps=None))
        else:
            state,reason=early_stop_update(state,step,summary['accuracy'],summary['clipping_rate'],early)
            snapshots[step]=state
    return state,reason,snapshots


def resume_housekeeping(run_dir, step, world):
    """Rank 0, before any worker starts: move everything written after checkpoint `step` into an archive."""
    directory=run_dir/'checkpoints'/f'step-{step:04d}'
    assert checkpoint_complete(directory,world), f'Incomplete checkpoint: {directory}'
    assert not (run_dir/'complete.json').exists(), 'Training already complete; nothing to resume'
    archive=run_dir/'resume_archive'/f'from-step-{step:04d}-{time.strftime("%Y%m%d_%H%M%S")}'
    archive.mkdir(parents=True)
    moved=[]
    for kind in ('rollouts','evaluation','checkpoints'):
        for path in sorted((run_dir/kind).glob('step-*')):
            if int(path.name.split('-')[1])>step:
                (archive/kind).mkdir(exist_ok=True)
                path.rename(archive/kind/path.name)
                moved.append(f'{kind}/{path.name}')
    for name in ['workers','score_worker','live.json','status.json','early_stop.json','runtime.json',
                 *sorted(p.name for p in run_dir.glob('failure.rank-*.json'))]:
        if (run_dir/name).exists():
            (run_dir/name).rename(archive/name)
            moved.append(name)
    rows=[json.loads(line) for line in (run_dir/'metrics.jsonl').open()]
    shutil.copyfile(run_dir/'metrics.jsonl',archive/'metrics.jsonl')
    kept=[r for r in rows if r['step']<=step]
    assert [r['step'] for r in kept]==list(range(1,step+1)), 'metrics.jsonl must hold every step up to the checkpoint'
    temporary=run_dir/'metrics.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(r)+'\n' for r in kept))
    temporary.replace(run_dir/'metrics.jsonl')
    assert set(evaluation_summaries(run_dir,step))==set(evaluation_summaries(run_dir,10**9))
    return dict(from_step=step,archive=str(archive),moved=moved,dropped_metric_rows=len(rows)-len(kept),time=time.time())


def load_checkpoint(bridge, optimizer, run_dir, step, initial_sha, learning_rate):
    """Restore projector weights and the AdamW state written at `step`; verifies every stored hash."""
    directory=run_dir/'checkpoints'/f'step-{step:04d}'
    meta=json.loads((directory/'checkpoint.json').read_text())
    assert meta['step']==step and meta['initial_bridge_state_sha256']==initial_sha
    assert hashlib.sha256((directory/'bridge.pt').read_bytes()).hexdigest()==meta['bridge_sha256']
    bridge.load_bridge_checkpoint(directory/'bridge.pt')
    assert weight_digest(bridge)==meta['bridge_state_sha256']
    saved=torch.load(directory/'optimizer.pt',map_location='cpu',weights_only=True)
    assert saved['step']==step and saved['initial_sha256']==initial_sha
    optimizer.load_state_dict(saved['optimizer'])
    assert len(optimizer.param_groups)==1 and optimizer.param_groups[0]['lr']==learning_rate
    assert len(optimizer.state)==len(optimizer.param_groups[0]['params'])
    assert all(float(v['step'])==step for v in optimizer.state.values()), 'AdamW step count != checkpoint step'
    return meta


def wandb_replay(run, run_dir, protocol, is_grpo, first, last):
    """Re-log steps first..last from local files (metrics.jsonl, evaluation summaries, early-stop replay)."""
    records={r['step']:r for r in map(json.loads,(run_dir/'metrics.jsonl').open())}
    evals=evaluation_summaries(run_dir,last)
    _,_,snapshots=replay_early_stop(run_dir,protocol,last)
    logged=[]
    for step in range(max(first,0),last+1):
        payload={}
        if step in records:
            payload.update(train_payload(records[step],protocol,is_grpo))
        if step in evals:
            payload.update(val_payload(evals[step],step))
        if step in snapshots:
            payload.update(early_payload(snapshots[step]))
        if payload:
            run.log(payload,step=step)
            logged.append(step)
    return logged


def wandb_resume(protocol, run_dir, step, is_grpo):
    """Continue the W&B run. Same run id when its remote history ends at or before `step` (missing steps are
    re-logged from local files); otherwise a new run `<name>_resume<step>` carrying the replayed history."""
    import wandb
    info=json.loads((run_dir/'wandb.json').read_text())
    mode=os.environ.get('WANDB_MODE','online')
    remote_last=None
    if mode=='online':
        try:
            remote_last=wandb.Api(timeout=60).run(f"{info['entity']}/{info['project']}/{info['id']}").lastHistoryStep
        except Exception as error:
            print(json.dumps(dict(wandb_remote_query_failed=repr(error))),flush=True)
    if remote_last is not None and remote_last<=step:
        run=wandb.init(entity=info['entity'],project=info['project'],id=info['id'],resume='must',
            dir=str(run_dir),mode=mode)
        how,first=('same_run',remote_last+1)
    else:
        tags=[t for t in os.environ.get('WANDB_TAGS','').split(',') if t]+['resumed']
        run=wandb.init(entity=info.get('entity'),project=protocol.get('wandb_project'),
            name=protocol['wandb_run_name']+f'_resume{step}',config=protocol,dir=str(run_dir),mode=mode,tags=tags)
        how,first=('new_run',0)
    logged=wandb_replay(run,run_dir,protocol,is_grpo,first,step)
    history=info.get('resumes',[])+[dict(from_step=step,how=how,previous_id=info['id'],remote_last_step=remote_last,
        replayed_steps=[logged[0],logged[-1]] if logged else [],new_id=run.id,time=time.time())]
    save(run_dir/'wandb.json',dict(id=run.id,url=run.url,entity=run.entity,project=run.project,resumes=history))
    return run


def evaluate(bridge, engine, scorer, eval_rows, run_dir, step, rank, world,
             tokenizer, eos, wandb_run, smoke=False):
    protocol=json.loads((run_dir.parent/'protocol.json').read_text())
    identity=[weight_digest(bridge) if rank==0 else None]
    dist.broadcast_object_list(identity,src=0)
    evaluation=run_dir/'evaluation'/f'step-{step:04d}'
    quick=smoke and not protocol.get('diagnostic_full_validation')
    chosen=eval_rows[:6] if quick else eval_rows
    samples=2 if quick else protocol['eval_samples']
    requests=[dict(**r,sample_index=j,sampling_seed=1234+protocol.get('eval_seed_stride',4)*r['index']+j)
        for r in chosen[rank::world] for j in range(samples)]
    eval_started=time.monotonic()
    generate(bridge,engine,requests,identity[0],EVAL_SAMPLING,
        protocol['max_response_length'],evaluation/f'rank-{rank:02d}',tokenizer.pad_token_id,eos)
    dist.barrier()
    summary=None  # only rank 0 scores; the other ranks still return a value
    if rank==0:
        request=dict(inputs=[str(evaluation/f'rank-{i:02d}/responses.json') for i in range(world)],
            expected_responses=len(chosen)*samples,result=str(evaluation/'summary.json'))
        dapo_answer=protocol.get('dataset')=='dapo' and not protocol.get('math_verify_boxed_only',True)
        if protocol.get('val_metric','full')=='answer':
            # Corrected metric: last `Answer:` match, length-capped = 0. Legacy full parse -> accuracy_full.
            assert dapo_answer
            request['metric']='answer'
        elif dapo_answer:
            request['answer_metrics']=True  # older protocols: legacy val/avg@4 plus the corrected fields
        summary=scorer.call(request)
        assert summary['gold_missing']==0 and summary['questions']==len(chosen)
        summary.update(step=step,samples_per_question=samples,wall_seconds=time.monotonic()-eval_started,
            bridge_state_sha256=identity[0],sampling=EVAL_SAMPLING,backend='native vLLM',
            validation_before_training=step==0)
        summary[f'avg_at_{samples}']=summary['accuracy']
        summary[f'pass_at_{samples}']=summary['pass_at_k']
        save(evaluation/'summary.json',summary)
        protocol=json.loads((run_dir.parent/'protocol.json').read_text())
        if protocol.get('save_evaluation_bridges',False):
            state=bridge.bridge_checkpoint_state()
            temporary=evaluation/'bridge.pt.tmp'
            torch.save(state,temporary)
            temporary.replace(evaluation/'bridge.pt')
            save(evaluation/'checkpoint.json',dict(step=step,bridge_state_sha256=identity[0],
                bridge_sha256=hashlib.sha256((evaluation/'bridge.pt').read_bytes()).hexdigest()))
            save(evaluation/'subspace_config.json',bridge.metadata())
        print(json.dumps(dict(evaluation=summary)),flush=True)
        if wandb_run:
            wandb_run.log(val_payload(summary,step),step=step)
    dist.barrier()
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=BASE)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stress',action='store_true')
    parser.add_argument('--objective',choices=['opd','grpo'],default='opd')
    parser.add_argument('--resume-step',type=int,help='Continue a formal run from its complete checkpoint at this step')
    args=parser.parse_args()
    resume=args.resume_step
    assert resume is None or (not args.smoke and not args.stress and resume>0)
    protocol=json.loads((args.root/'protocol.json').read_text())
    startup_checks=bool(protocol.get('formal_startup_checks',False) and resume is None and not args.smoke)
    assert protocol['source_manifest_sha256']==verify_sources()
    assert protocol['initialization']=='zero'
    layout=rank_layout(protocol)
    rank,world,local=layout['rank'],layout['world'],layout['local_rank']
    cache=Path('/tmp/platour_random4b_train_v1')/f'{args.root.name}-node-{layout["node_rank"]}'/f'rank-{rank:02d}'
    assert os.statvfs('/tmp').f_bavail * os.statvfs('/tmp').f_frsize > 16 * 2**30
    for key,name in [('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),('VLLM_CACHE_ROOT','vllm')]:
        directory=cache/name
        directory.mkdir(parents=True,exist_ok=True)
        os.environ[key]=str(directory)
    assert protocol['eval_sampling']==EVAL_SAMPLING, 'Evaluation defaults changed; use a fresh protocol/output root'
    assert protocol['train_sampling']==TRAIN_SAMPLING
    parallel_size=protocol.get('teacher_model_parallel_size',1)
    paired=parallel_size>1
    assert world==protocol['world_size']
    physical=list(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    assert physical==protocol['gpu_ids']
    assert not paired and world==8 and len(physical)==8 and protocol['random_template_formal']
    device_index=layout['primary']
    teacher_devices=layout['devices']
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
    if resume is None:
        if (run_dir/'metrics.jsonl').exists() or (run_dir/'complete.json').exists():
            raise RuntimeError('Fresh training refuses preexisting metrics/checkpoints')
    else:
        assert resume<protocol['total_steps'] and resume in (protocol.get('checkpoint_steps') or [])
        holder=[resume_housekeeping(run_dir,resume,world) if rank==0 else None]
        dist.broadcast_object_list(holder,src=0)
        resume_record=holder[0]
    from transformers import AutoTokenizer
    from random_template_bridge import load_bridge
    bridge,runtime=load_bridge(ROOT,protocol)
    assert not hasattr(bridge,'teacher') and protocol['teacher_model_loaded'] is False
    assert bridge.layer_map==protocol['layer_map'], 'Actual train/rollout/eval mapping must match protocol'
    bridge.student_activation_checkpointing=protocol.get('student_activation_checkpointing',False)
    runtime['student_activation_checkpointing']=bridge.student_activation_checkpointing
    runtime.update(node_rank=layout['node_rank'],global_rank=rank,world_size=world,
                   local_world_size=layout['local_world'],local_devices=teacher_devices)
    save(run_dir/f'placement.rank-{rank:02d}.json',runtime)
    if protocol.get('teacher_weight_precision')=='fp8':
        # Audit the real quantized encoder before both smoke and formal work.
        # All ranks exercise both halves of the paired model on a real prompt.
        audit_row=json.loads((args.root/'train_prompts.json').read_text())[0]
        ids=torch.tensor([audit_row['prompt_token_ids']],device=bridge.student.device)
        with torch.no_grad():
            hidden=bridge.teacher.model(input_ids=ids,attention_mask=torch.ones_like(ids),
                use_cache=False,output_hidden_states=True).hidden_states
            features=bridge.teacher_features(hidden)
        assert len(features)==32 and all(h.dtype==torch.bfloat16 for h in features)
        assert all(h.device==bridge.student.device and torch.isfinite(h).all() for h in features)
        save(run_dir/f'fp8_hidden_audit.rank-{rank:02d}.json',dict(passed=True,
            hidden_dtypes=sorted({str(h.dtype) for h in hidden}),
            transfer_dtype='bfloat16',selected_layers=bridge.layer_map,finite=True,
            prompt_tokens=ids.shape[1],frozen_teacher=True))
        del hidden,features,ids
    init=initialize(bridge)
    params=bridge.bridge_parameters()
    assert all(p.requires_grad and p.dtype==torch.float32 for p in params)
    initial_sha=weight_digest(bridge)
    initial_hashes=[None]*world
    dist.all_gather_object(initial_hashes,initial_sha)
    assert len(set(initial_hashes))==1
    learning_rate=float(protocol['learning_rate'])
    assert protocol.get('scheduler','constant')=='constant' and protocol.get('warmup_steps',0)==0
    optimizer=torch.optim.AdamW(params,lr=learning_rate,weight_decay=0.,betas=(.9,.999))
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'data/models/Qwen3.5-4B',local_files_only=True)
    eos=bridge.student.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else list(eos)
    train_rows=json.loads((args.root/'train_prompts.json').read_text())
    eval_rows=json.loads((args.root/'eval_prompts.json').read_text())
    examples,epochs=protocol['examples_per_epoch'],protocol['epochs']
    prompt_cap,response_cap=protocol['max_prompt_length'],protocol['max_response_length']
    assert len(train_rows)==examples and [r['index'] for r in train_rows]==list(range(examples))
    assert len(eval_rows)==protocol['eval_questions']
    assert bridge.student.lm_head.weight.shape[0]==248320 and len(tokenizer)==248077
    if rank==0:
        if resume is None:
            checkpoint(bridge,optimizer,run_dir/'initial',0,initial_sha)
            save(run_dir/'initialization.json',dict(init,bridge_state_sha256=initial_sha,
                replica_hashes=initial_hashes,layer_map=bridge.layer_map))
        else:
            # Deterministic initialisation: the resumed process must rebuild the original starting point.
            assert json.loads((run_dir/'initialization.json').read_text())['bridge_state_sha256']==initial_sha
        save(run_dir/'runtime.json',runtime)
    if resume is not None:
        resumed_meta=load_checkpoint(bridge,optimizer,run_dir,resume,initial_sha,learning_rate)
        resumed_hashes=[None]*world
        dist.all_gather_object(resumed_hashes,weight_digest(bridge))
        assert len(set(resumed_hashes))==1 and resumed_hashes[0]==resumed_meta['bridge_state_sha256']
    dist.barrier()
    engine=scorer=wandb_run=None
    try:
        torch.cuda.empty_cache()
        engine=RPC('qwen35_bridge_opd_vllm_worker.py',run_dir/f'workers/rank-{rank:02d}/vllm',
                   gpu=physical[device_index],memory_fraction=protocol['vllm_memory_fraction'],
                   max_model_len=protocol.get('vllm_max_model_len',prompt_cap+response_cap),
                   max_num_seqs=protocol.get('vllm_max_num_seqs',32),
                   decode_batch_size=protocol.get('vllm_decode_batch_size',32))
        if rank==0:
            # Code tasks (LCB) use their own container-judging scorer with the same RPC contract.
            scorer=RPC(protocol.get('score_worker','qwen35_bridge_opd_score_worker.py'),run_dir/'score_worker',
                       boxed_only=protocol.get('math_verify_boxed_only',True))
        dist.barrier()
        if args.stress or startup_checks:
            # Exact maximum active lengths; no padding, with the resident vLLM
            # allocation present. Do not update W using this synthetic batch.
            if protocol.get('stress_all_ranks',False) or rank==0:
                from random_config import stress_prompt
                original=max(train_rows,key=lambda r:len(r['prompt_token_ids']))
                source=original['prompt_token_ids']
                prompt=stress_prompt(original,protocol)
                response=(source*((response_cap+len(source)-1)//len(source)))[:response_cap]
                torch.cuda.reset_peak_memory_stats()
                before=time.monotonic()
                if is_grpo:
                    log_probs=sampled_log_probs(bridge,prompt,response)
                    value,_=policy_loss(log_probs,1.,log_probs.detach())
                else:
                    value=opd_loss(bridge,prompt,response)
                value.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                assert any(p.grad is not None and p.grad.norm()>0 for p in params)
                save(run_dir/f'worst_case_profile.rank-{rank:02d}.json',dict(passed=True,prompt_tokens=prompt_cap,response_tokens=response_cap,
                    objective=args.objective,
                    active_input_tokens=prompt_cap+response_cap-1,padding_tokens=0,micro_batch=1,loss=float(value),grad_norm=float(norm),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    gpu_total_used_gib=(torch.cuda.get_device_properties(device_index).total_memory-torch.cuda.mem_get_info()[0])/2**30,
                    gpu_pair_used_gib={str(i):(torch.cuda.get_device_properties(i).total_memory-torch.cuda.mem_get_info(i)[0])/2**30
                        for i in teacher_devices},
                    seconds=time.monotonic()-before,vllm_resident=True))
                optimizer.zero_grad(set_to_none=True)
                del value
                torch.cuda.empty_cache()
            dist.barrier()
        if resume is not None and rank==0:
            wandb_run=wandb_resume(protocol,run_dir,resume,is_grpo)
            if os.environ.get('WANDB_MODE','online')=='online':
                assert not wandb_run.offline
        elif not args.smoke and rank==0:
            import wandb
            wandb_run=wandb.init(project=protocol.get('wandb_project','hidden-bridge-math-grpo' if is_grpo else 'hidden-bridge-math-opd'),
                name=protocol.get('wandb_run_name',
                    f'qwen35_35b_to_4b_constant_gate1_grpo_g8_bs64_4k_lr1e5_3epoch_eval13_vllm_{len(physical)}gpu' if is_grpo else
                    f'qwen35_35b_to_4b_constant_gate1_opd_g1_bs128_4k_lr1e5_3epoch_eval12_vllm_{len(physical)}gpu'),
                config=protocol,dir=str(run_dir),mode=os.environ.get('WANDB_MODE','online'),
                **(dict(id=os.environ['WANDB_RUN_ID'],resume='must') if os.environ.get('WANDB_RUN_ID') else {}))
            if os.environ.get('WANDB_RUN_ID'):
                wandb_run.summary.update(dict(queue_status='qualifying' if startup_checks else 'running',
                                               training_started=not startup_checks))
            assert os.environ.get('WANDB_MODE','online')!='online' or not wandb_run.offline
            save(run_dir/'wandb.json',dict(id=wandb_run.id,url=wandb_run.url,entity=wandb_run.entity,project=wandb_run.project))
            wandb_run.log({'bridge/initial_weight_norm':math.sqrt(sum(float(p.detach().square().sum()) for p in params))},step=0,commit=False)
        dist.barrier()
        early=protocol.get('early_stop') if not args.smoke else None
        early_state=None
        stop_reason=None
        evaluated_steps=[]
        if resume is None and not args.smoke and not protocol['validation_before_training']:
            # Step-0 validation explicitly skipped (protocol records it); weights are untouched until step 1.
            assert protocol.get('initial_eval_step') is None
            if rank==0:
                save(run_dir/'status.json',dict(phase='train',step=0,initial_eval_skipped=True,time=time.time()))
            dist.barrier()
        if resume is None and not args.smoke and protocol['validation_before_training']:
            if rank==0:
                save(run_dir/'status.json',dict(phase='initial_eval',step=0,time=time.time()))
            cpu_rng,cuda_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
            initial_summary=evaluate(bridge,engine,scorer,eval_rows,run_dir,0,rank,world,tokenizer,eos,wandb_run)
            evaluated_steps.append(0)
            if early:
                # Step 0 is the baseline: later evaluations must beat it (it can never trigger a stop itself).
                early_state,_=early_stop_update(None,0,*(([initial_summary['accuracy'],initial_summary['clipping_rate']]
                    if rank==0 else [0.,0.])),dict(early,patience=10**9,max_steps=None))
                holder=[early_state]
                dist.broadcast_object_list(holder,src=0)
                early_state=holder[0]
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
        if startup_checks:
            from formal_qualification import verify_live
            check=verify_live(run_dir,protocol,rank,initial_sha)
            save(run_dir/f'qualification.rank-{rank:02d}.json',check)
            dist.barrier()
            if rank==0:
                save(args.root/'qualification_audit.json',dict(passed=True,
                    source_manifest_sha256=protocol['source_manifest_sha256'],protocol_sha256=hashlib.sha256((args.root/'protocol.json').read_bytes()).hexdigest(),
                    checks=[json.loads((run_dir/f'qualification.rank-{r:02d}.json').read_text()) for r in range(world)],
                    time=time.time(),actual_model='397B-FP8->4B',updates_before_qualification=0))
                if wandb_run:wandb_run.summary.update(dict(queue_status='running',startup_qualification_passed=True,
                                                           training_started=True))
            dist.barrier()
        started=time.monotonic()
        steps=2 if args.smoke else protocol['total_steps']
        batch_size,group_size,epoch_steps=(protocol[k] for k in ['batch_size','group_size','steps_per_epoch'])
        orders=[torch.randperm(examples,generator=torch.Generator().manual_seed(42+epoch)).tolist() for epoch in range(epochs)]
        visited=[]
        smoke_selection=sorted(range(examples),key=lambda i:len(train_rows[i]['prompt_token_ids']))[30:46]
        if is_grpo:
            ordered=sorted(range(examples),key=lambda i:len(train_rows[i]['prompt_token_ids']))
            smoke_selection=[ordered[(j+1)*len(ordered)//(batch_size+1)] for j in range(batch_size)]
        nonzero_updates=0
        def selection_for(step):
            epoch=(step-1)//epoch_steps
            offset=((step-1)%epoch_steps)*batch_size
            return epoch,(smoke_selection if args.smoke else orders[epoch][offset:offset+batch_size])
        first_step=1
        if resume is not None:
            # Fast-forward the bookkeeping exactly as the uninterrupted run would have it after `resume`.
            records=[json.loads(line) for line in (run_dir/'metrics.jsonl').open()]
            assert [r['step'] for r in records]==list(range(1,resume+1))
            assert all(r['learning_rate']==learning_rate for r in records)
            for s in range(1,resume+1):
                visited.extend(selection_for(s)[1])
            nonzero_updates=sum(r['grad_norm']>0 for r in records)
            evaluated_steps=list(evaluation_summaries(run_dir,resume))
            early_state,stop_reason,_=replay_early_stop(run_dir,protocol,resume)
            assert stop_reason is None, f'Run had already stopped at step {resume}: {stop_reason}'
            started=time.monotonic()-records[-1]['elapsed_seconds']
            first_step=resume+1
            rng=torch.load(run_dir/'checkpoints'/f'step-{resume:04d}'/f'rng.rank-{rank:02d}.pt',
                map_location='cpu',weights_only=True)
            torch.set_rng_state(rng['cpu_rng'])
            torch.cuda.set_rng_state(rng['cuda_rng'])
            if rank==0:
                history=json.loads((run_dir/'resume.json').read_text())['resumes'] if (run_dir/'resume.json').exists() else []
                save(run_dir/'resume.json',dict(resumes=history+[dict(resume_record,
                    nonzero_updates=nonzero_updates,evaluated_steps=evaluated_steps,
                    bridge_state_sha256=resumed_meta['bridge_state_sha256'],
                    wandb=json.loads((run_dir/'wandb.json').read_text()).get('resumes',[])[-1:])]))
                save(run_dir/'status.json',dict(phase='train',step=resume,resumed_from=resume,time=time.time()))
            dist.barrier()
        for step in range(first_step,steps+1):
            tick=time.monotonic()
            epoch,selection=selection_for(step)
            if args.smoke and step==2:
                selection=selection[:1]  # Uneven singleton tail: all other ranks are empty.
            rows=[dict(**train_rows[i],sample_index=j,
                sampling_seed=42+100000*epoch+1000*step+(group_size*i+j if is_grpo else i))
                for i in selection[rank::world] for j in range(group_size)]
            identity=[weight_digest(bridge) if rank==0 else None]
            dist.broadcast_object_list(identity,src=0)
            digest=identity[0]
            path=run_dir/'rollouts'/f'step-{step:04d}'/f'rank-{rank:02d}'
            rollout,export_seconds=generate(bridge,engine,rows,digest,TRAIN_SAMPLING,
                response_cap,path,tokenizer.pad_token_id,eos,return_logprobs=is_grpo)
            if is_grpo:
                dist.barrier()
                directory=run_dir/'rollouts'/f'step-{step:04d}'
                if rank==0:
                    summary=scorer.call(dict(inputs=[str(directory/f'rank-{i:02d}/responses.json') for i in range(world)],
                        expected_responses=len(selection)*group_size,result=str(directory/'score_summary.json'),
                        format_penalty=bool(protocol.get('format_penalty',False))))
                dist.barrier()
                scored=[json.loads(line) for line in (directory/'scored.jsonl').open()]
                advantages,group_stats=group_advantages(scored,group_size)
                if rank==0:
                    save(directory/'advantages.json',dict(group_stats=group_stats,rows=[dict(index=r['index'],
                        sample_index=r['sample_index'],reward=r['acc'],
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
                    value=opd_loss(bridge,row['prompt_token_ids'],row['response_token_ids'])
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
            assert not hasattr(bridge,'teacher') and all(p.grad is None and not p.requires_grad for p in bridge.student.parameters())
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
                if step==1:
                    before_probe=weight_digest(bridge)
                    optimizer.zero_grad(set_to_none=True)
                    probe=rollout['rows'][0]
                    probe_loss=-sampled_log_probs(bridge,probe['prompt_token_ids'],probe['response_token_ids'][:4]).mean()
                    probe_loss.backward()
                    vector_norm=float(bridge.virtual_hidden.grad.norm())
                    projector_norm=math.sqrt(sum(float(p.grad.square().sum()) for p in bridge.maps.parameters()))
                    assert math.isfinite(vector_norm) and vector_norm>0 and math.isfinite(projector_norm) and projector_norm>0
                    save(run_dir/f'virtual_gradient_probe.rank-{rank:02d}.json',dict(passed=True,
                        V_grad_norm=vector_norm,W_grad_norm=projector_norm,optimizer_updates=0,
                        original_parameters_unchanged=weight_digest(bridge)==before_probe))
                    optimizer.zero_grad(set_to_none=True)
                    del probe_loss
            timing=torch.tensor([export_seconds,rollout['seconds'],time.monotonic()-update_started,
                torch.cuda.max_memory_allocated()/2**30],device='cuda',dtype=torch.float64)
            dist.all_reduce(timing,op=dist.ReduceOp.MAX)
            dist.barrier()
            if rank==0:
                directory=run_dir/'rollouts'/f'step-{step:04d}'
                if not is_grpo:
                    summary=scorer.call(dict(inputs=[str(directory/f'rank-{i:02d}/responses.json') for i in range(world)],
                        expected_responses=len(selection),result=str(directory/'score_summary.json')))
                record=dict(step=step,epoch=epoch+1,total_steps=steps,examples=len(selection),reply_tokens=total_tokens,
                    learning_rate=learning_rate,grad_norm=float(norm),
                    train_accuracy=summary['accuracy'],clipping_rate=summary['clipping_rate'],mean_tokens=summary['mean_tokens'],
                    hf_export_seconds=float(timing[0]),vllm_decode_seconds=float(timing[1]),update_seconds=float(timing[2]),
                    step_seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started,
                    peak_allocated_gib=float(timing[3]),world_size=world,
                    rollout_bridge_state_sha256=digest,training_generation='native vLLM')
                if is_grpo:
                    active_tokens=sum(r['tokens'] for r in scored if advantages[(r['index'],r['sample_index'])]!=0.)
                    record.update(policy_loss=float(loss_sum/(len(selection)*group_size)),responses=len(selection)*group_size,
                        group_size=group_size,**group_stats,active_reply_tokens=active_tokens,
                        **{key:float(pg_metrics[i]/max(active_tokens,1)) for i,key in enumerate(
                            ['pg_clip_fraction','ppo_kl','rollout_is_mean','rollout_is_clipped_fraction','actor_rollout_logprob_abs_error'])})
                    if protocol.get('format_penalty'):
                        # Only the penalised (GRPO) scorer reports these; the smoke non-GRPO summary does not.
                        record.update({k:summary[k] for k in
                            ['accuracy_unpenalized','format_violation_rate','format_zeroed_rate'] if k in summary})
                else:
                    record['reverse_kl']=float(loss_sum/total_tokens)
                with (run_dir/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
                save(run_dir/'live.json',record)
                print(json.dumps(record),flush=True)
                if wandb_run:
                    wandb_run.log(train_payload(record,protocol,is_grpo),step=step)
            visited.extend(selection)
            dist.barrier()
            eval_now=step==steps if args.smoke else step in protocol['eval_steps']
            if eval_now:
                eval_summary=evaluate(bridge,engine,scorer,eval_rows,run_dir,step,rank,world,
                    tokenizer,eos,wandb_run,smoke=args.smoke)
                if not args.smoke:
                    evaluated_steps.append(step)
                if early:
                    # Decide on rank 0 only, then broadcast: every rank must take the same branch.
                    if rank==0:
                        early_state,stop_reason=early_stop_update(early_state,step,eval_summary['accuracy'],
                            eval_summary['clipping_rate'],early)
                        save(run_dir/'early_stop.json',dict(config=early,state=early_state,stop_reason=stop_reason,
                            stopped=stop_reason is not None,step=step,time=time.time()))
                        if wandb_run:
                            wandb_run.log(early_payload(early_state),step=step)
                    holder=[(early_state,stop_reason) if rank==0 else None]
                    dist.broadcast_object_list(holder,src=0)
                    early_state,stop_reason=holder[0]
            checkpoint_steps=protocol.get('checkpoint_steps')
            save_now=(step in checkpoint_steps if checkpoint_steps is not None else step%epoch_steps==0)
            if (not args.smoke and save_now) or (args.smoke and step==steps):
                destination=run_dir/'checkpoints'/f'step-{step:04d}' if checkpoint_steps is not None else run_dir/f'epoch-{epoch+1}'
                if rank==0:
                    checkpoint(bridge,optimizer,destination,step,initial_sha)
                dist.barrier()
                torch.save(dict(cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),
                    destination/f'rng.rank-{rank:02d}.pt')
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if stop_reason is not None:
                # The evaluation checkpoint of this step has just been written above; leave the loop normally
                # so every rank, W&B and the vLLM/scoring workers shut down through the usual path.
                break
        early_stopped=stop_reason is not None
        last_step=step
        if not args.smoke and not early_stopped:
            assert len(visited)==examples*epochs
            for epoch in range(epochs):
                assert sorted(visited[epoch*examples:(epoch+1)*examples])==list(range(examples))
        hashes=[None]*world
        dist.all_gather_object(hashes,weight_digest(bridge))
        assert len(set(hashes))==1
        if is_grpo:
            assert nonzero_updates>0 and hashes[0]!=initial_sha
        if rank==0:
            evals=list((run_dir/'evaluation').glob('step-*/summary.json'))
            if args.smoke:
                assert len(evals)==1
            else:
                assert len(evals)==len(evaluated_steps)
                if not early_stopped:
                    assert len(evals)==protocol['total_evaluations']
            if not args.smoke and protocol.get('checkpoint_steps') is not None:
                assert all((run_dir/'checkpoints'/f'step-{s:04d}'/'optimizer.pt').exists()
                           and all((run_dir/'checkpoints'/f'step-{s:04d}'/f'rng.rank-{r:02d}.pt').exists()
                                   for r in range(world)) for s in protocol['checkpoint_steps'] if s<=last_step)
            if early:
                best=early_state['best_step']
                save(run_dir/'early_stop.json',dict(config=early,state=early_state,stop_reason=stop_reason,
                    stopped=early_stopped,last_step=last_step,best_step=best,
                    best_checkpoint=str(run_dir/'initial' if best==0 else run_dir/'checkpoints'/f'step-{best:04d}'),
                    time=time.time()))
            save(run_dir/'complete.json',dict(complete=True,smoke=args.smoke,steps=steps,steps_run=last_step,
                early_stopped=early_stopped,stop_reason=stop_reason,evaluated_steps=evaluated_steps,
                objective=args.objective,group_size=group_size,nonzero_gradient_updates=nonzero_updates,
                examples_seen=len(visited),epochs=1 if args.smoke else epochs,evals=len(evals),
                six_rank_exact_weights=world==6,eight_rank_exact_weights=world==8,
                replica_exact_weights=True,world_size=world,
                gpu_ids=physical,teacher=protocol['teacher'],final_bridge_state_sha256=hashes[0],
                initial_bridge_state_sha256=initial_sha,seconds=time.monotonic()-started,
                rollout_and_eval_backend='native vLLM',frozen_backbones_verified=True,
                singleton_tail_verified=args.smoke,learning_rate=learning_rate,
                resumed_from_steps=[r['from_step'] for r in json.loads((run_dir/'resume.json').read_text())['resumes']]
                    if (run_dir/'resume.json').exists() else []))
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
