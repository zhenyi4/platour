"""Eight-GPU full-parameter 4B GRPO, native rollouts, complete model and sharded FP32 optimizer resume."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

import torch
import torch.distributed as dist
from safetensors.torch import save_file, load_file
from qwen35_student import load_student
from qwen35_direct_math_bridge import accelerated_models
from qwen35_bridge_opd_core import ROOT, save, TRAIN_SAMPLING, EVAL_SAMPLING
from qwen35_bridge_grpo_core import group_advantages, policy_loss
from train_qwen35_bridge_opd_vllm import RPC as ExistingRPC
from full4b_core import model_digest, log_probs, ShardedMasterAdam


class Decoder:
    def __init__(self,root,rank,model,max_model_len=18432):
        self.directory=root/'workers'/f'rank-{rank:02d}'/'decoder'
        self.directory.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ)
        for key in list(env):
            if key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK',
                       'ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT') or key.startswith('TORCHELASTIC_'):
                env.pop(key,None)
        env['CUDA_VISIBLE_DEVICES']=str(rank)
        python=Path(env['PLATOUR_GENERATION_PYTHON'])
        env['PATH']=str(python.parent)+os.pathsep+env.get('PATH','')
        self.log=(self.directory/'worker.log').open('a')
        self.proc=subprocess.Popen([str(python),'-u',str(Path(__file__).with_name('full4b_generation.py')),
            '--model',str(model),'--directory',str(self.directory),'--max-model-len',str(max_model_len)],env=env,cwd=ROOT,
            stdin=subprocess.PIPE,stdout=self.log,stderr=subprocess.STDOUT,text=True)
        self.wait(self.directory/'ready.json',1200)
        self.counter=0

    def wait(self,path,timeout):
        start=time.monotonic()
        while not path.exists():
            if self.proc.poll() is not None:raise RuntimeError(f'Decoder exited: {self.directory}/worker.log')
            if time.monotonic()-start>timeout:raise TimeoutError(str(path))
            time.sleep(.2)
        result=json.loads(path.read_text())
        if result.get('complete') is False:raise RuntimeError(result)
        return result

    def call(self,request,path=None):
        self.counter+=1
        target=path or self.directory/f'rpc-{self.counter:04d}.json'
        assert not target.exists()
        self.proc.stdin.write(json.dumps(dict(request,result=str(target)))+'\n');self.proc.stdin.flush()
        return self.wait(target,7200)

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('{"command":"stop"}\n');self.proc.stdin.flush();self.proc.wait(timeout=30)
            except (BrokenPipeError,subprocess.TimeoutExpired):
                self.proc.terminate()
                try:self.proc.wait(timeout=15)
                except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=15)
        self.log.close()


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def status(root,rank,**value):
    save(root/f'status.rank-{rank:02d}.json',dict(value,time=time.time()))
    if rank==0:save(root/'status.json',dict(value,time=time.time()))


def synchronize_model(model,decoder,root,step,rank,world):
    identity=model_digest(model)
    identities=[None]*world
    dist.all_gather_object(identities,identity)
    assert len(set(identities))==1,'Full model replicas differ after optimizer update'
    directory=root/'weights'/f'step-{step:04d}'
    if rank==0:
        directory.mkdir(parents=True,exist_ok=True)
        destination=directory/'model.safetensors'
        cpu={key:value.detach().cpu().contiguous().clone() for key,value in model.state_dict().items()}
        save_file(cpu,str(destination)+'.tmp');Path(str(destination)+'.tmp').replace(destination)
        del cpu
        save(directory/'identity.json',dict(state_sha256=identity,file_sha256=sha(destination),step=step))
    dist.barrier()
    info=read(directory/'identity.json')
    result=decoder.call(dict(command='sync',path=str(directory/'model.safetensors'),file_sha256=info['file_sha256'],
                             state_sha256=identity))
    assert result['complete'] and result['identity']==identity
    engine_hashes=[None]*world
    dist.all_gather_object(engine_hashes,result['reports'][0]['actual_parameter_sha256'])
    assert len(set(engine_hashes))==1,'Native vLLM weights differ across replicas'
    save(directory/f'sync.rank-{rank:02d}.json',result)
    return identity,directory


def sample(decoder,rows,identity,sampling,cap,path,logprobs=False):
    if not rows:
        save(path,dict(complete=True,rows=[],state_sha256=identity,seconds=0.))
        return []
    result=decoder.call(dict(command='decode',rows=rows,state_sha256=identity,sampling=sampling,cap=cap,
                              logprobs=logprobs),path)
    assert result['complete'] and result['state_sha256']==identity and len(result['rows'])==len(rows)
    for actual,expected in zip(result['rows'],rows,strict=True):
        assert all(actual[key]==expected[key] for key in ('index','sample_index','sampling_seed','prompt_token_ids'))
        assert actual['full_model_state_sha256']==identity
    return result['rows']


def checkpoint(model,optimizer,root,step,weights,rank,world):
    directory=root/'checkpoints'/f'step-{step:04d}';directory.mkdir(parents=True,exist_ok=True)
    state=optimizer.state_dict()
    temporary=directory/f'optimizer.rank-{rank:02d}.pt.tmp';torch.save(state,temporary);temporary.replace(directory/f'optimizer.rank-{rank:02d}.pt')
    del state
    torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),step=step),directory/f'rng.rank-{rank:02d}.pt')
    dist.barrier()
    if rank==0:
        save(directory/'checkpoint.json',dict(complete=True,step=step,weights=str(weights/'model.safetensors'),
             weights_sha256=sha(weights/'model.safetensors'),state_sha256=read(weights/'identity.json')['state_sha256'],
             world_size=world,optimizer_shards=world,full_parameter_training=True))
    dist.barrier()


def valid_backend_logprobs(hf_logprob,vllm_logprob,ratio_cap=2.):
    assert ratio_cap>1
    return (math.isfinite(hf_logprob) and math.isfinite(vllm_logprob) and
            hf_logprob<=1e-6 and vllm_logprob<=1e-6 and
            abs(hf_logprob-vllm_logprob)<math.log(ratio_cap))


def audit_decoder(model,decoder,identity,row,root,step):
    prompt=row['prompt_token_ids']
    request=dict(row,sample_index=0,sampling_seed=42)
    decoded=sample(decoder,[request],identity,dict(EVAL_SAMPLING,temperature=0.),1,
        root/f'decoder_audit/step-{step:04d}.json',True)[0]
    token=decoded['response_token_ids'][0]
    with torch.no_grad():
        probability=float(log_probs(model,prompt,[token],checkpointing=False)[0])
    actual=decoded['rollout_log_probs'][0]
    difference=abs(probability-actual)
    passed=valid_backend_logprobs(probability,actual)
    save(root/f'decoder_audit/step-{step:04d}-comparison.json',dict(passed=passed,token=token,
        hf_logprob=probability,vllm_logprob=actual,absolute_error=difference,state_sha256=identity,
        audit_kind='backend_logprob_sanity_after_exact_full_weight_reload',logprob_ratio_cap=2.,
        maximum_logprob_gap=math.log(2.)))
    assert passed,f'HF/current vLLM probability sanity check exceeded existing IS bound: {difference}'


def validate(model,decoder,scorer,rows,identity,root,step,rank,world,protocol,wandb_run,smoke=False):
    chosen=rows[:8] if smoke else rows
    samples=2 if smoke else protocol['eval_samples']
    requests=[dict(row,sample_index=j,sampling_seed=1234+samples*row['index']+j)
              for row in chosen[rank::world] for j in range(samples)]
    directory=root/'evaluation'/f'step-{step:04d}'
    tick=time.monotonic();status(root,rank,phase='validation',step=step)
    sample(decoder,requests,identity,EVAL_SAMPLING,protocol['max_response_length'],directory/f'rank-{rank:02d}/responses.json')
    dist.barrier()
    summary=None
    if rank==0:
        request=dict(inputs=[str(directory/f'rank-{r:02d}/responses.json') for r in range(world)],
                     expected_responses=len(chosen)*samples,result=str(directory/'summary.json'))
        if protocol['dataset']=='dapo':request['metric']='answer'
        summary=scorer.call(request)
        summary.update(step=step,samples_per_question=samples,wall_seconds=time.monotonic()-tick,
                       full_model_state_sha256=identity)
        save(directory/'summary.json',summary)
        if wandb_run:wandb_run.log({f'val/avg@{samples}':summary['accuracy'],f'val/pass@{samples}':summary['pass_at_k'],
            'val/clipping_rate':summary['clipping_rate'],'val/mean_tokens':summary['mean_tokens'],
            'val/seconds':summary['wall_seconds'],'training/global_step':step},step=step)
    holder=[summary];dist.broadcast_object_list(holder,src=0)
    return holder[0]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume-step',type=int)
    args=parser.parse_args();protocol=read(args.root/'protocol.json')
    for relative,digest in read(args.root.parent/'source_manifest.json').items():
        assert sha(args.root.parent/relative)==digest,relative
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    assert rank==local and world==8
    torch.cuda.set_device(local);torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',device_id=torch.device('cuda',local),timeout=timedelta(minutes=90))
    run=args.root/('smoke' if args.smoke else 'train');run.mkdir(parents=True,exist_ok=True)
    decoder=scorer=wandb_run=None
    try:
        status(run,rank,phase='load_full_model',step=0)
        runtime=accelerated_models()
        model=load_student(ROOT/'data/models/Qwen3.5-4B').eval().requires_grad_(True)
        assert all(p.requires_grad for p in model.parameters())
        assert model.lm_head.weight is model.model.embed_tokens.weight,'Preserve original tied embedding/head'
        parameter_count=sum(p.numel() for p in model.parameters())
        optimizer=ShardedMasterAdam(model,rank,world,protocol['learning_rate'])
        save(run/f'parameters.rank-{rank:02d}.json',dict(full_model_parameters=parameter_count,
            local_master_parameters=sum(p.numel() for p in optimizer.masters),model_dtype='bfloat16',master_dtype='float32',
            tied_embedding_lm_head=True,all_text_parameters_trainable=True,teacher=False,projector=False))
        training=read(args.root/'train_prompts.json');validation=read(args.root/'eval_prompts.json')
        step0=0
        if args.resume_step is not None:
            assert not args.smoke
            step0=args.resume_step;directory=run/'checkpoints'/f'step-{step0:04d}';metadata=read(directory/'checkpoint.json')
            assert metadata['world_size']==world and sha(metadata['weights'])==metadata['weights_sha256']
            model.load_state_dict(load_file(metadata['weights']),strict=True)
            optimizer.load_state_dict(torch.load(directory/f'optimizer.rank-{rank:02d}.pt',map_location='cpu',weights_only=True))
        decoder=Decoder(run,rank,ROOT/'data/models/Qwen3.5-4B')
        if rank==0:scorer=ExistingRPC('lcb_score_worker.py' if protocol['dataset']=='lcb' else 'qwen35_bridge_opd_score_worker.py',run/'scorer',boxed_only=False)
        identity,weights=synchronize_model(model,decoder,run,step0,rank,world)
        audit_decoder(model,decoder,identity,training[rank],run/f'audit.rank-{rank:02d}',step0)
        if args.smoke:
            status(run,rank,phase='stress_full_model_16k',step=0)
            torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
            prompt=training[0]['prompt_token_ids'];eos=model.generation_config.eos_token_id
            response=[7]*16383+[eos if isinstance(eos,int) else eos[0]]
            probability=log_probs(model,prompt,response)
            (-probability.mean()).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            save(run/f'stress.rank-{rank:02d}.json',dict(passed=True,response_tokens=16384,seconds=time.monotonic()-tick,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,trainable_parameters=parameter_count,
                all_gradient_tensors_present=True,full_parameter_training=True))
            optimizer.zero_grad();del probability;torch.cuda.empty_cache();dist.barrier()
        if not args.smoke and rank==0:
            import wandb
            wandb_run=wandb.init(project='native-full4b-dapo',config=protocol,dir=str(run),mode='disabled')
            wandb_run.summary.update(dict(training_started=True,queue_status='running',trainable_parameters=parameter_count))
            save(run/'wandb.json',dict(id=wandb_run.id,project='native-full4b-dapo',url=None))
        state=None
        if step0==0:
            baseline=validate(model,decoder,scorer,validation,identity,run,0,rank,world,protocol,wandb_run,smoke=args.smoke)
            state=dict(best_step=0,best_accuracy=baseline['accuracy'],fails=0)
        else:
            state=read(run/'early_stop.json')['state']
            rng=torch.load(run/'checkpoints'/f'step-{step0:04d}'/f'rng.rank-{rank:02d}.pt',weights_only=True,map_location='cpu')
            torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'])
        checkpoint(model,optimizer,run,step0,weights,rank,world)
        total=2 if args.smoke else protocol['total_steps'];start=time.monotonic();reason=None
        for step in range(step0+1,total+1):
            tick=time.monotonic();status(run,rank,phase='rollout',step=step)
            epoch=(step-1)//protocol['steps_per_epoch'];offset=(step-1)%protocol['steps_per_epoch']*64
            order=torch.randperm(len(training),generator=torch.Generator().manual_seed(42+epoch)).tolist()
            selected=[training[i] for i in (order[:64] if args.smoke and step==1 else order[:1] if args.smoke else order[offset:offset+64])]
            requests=[dict(row,sample_index=j,sampling_seed=1_000_000+step*512+i*8+j)
                      for i,row in enumerate(selected) if i%world==rank for j in range(8)]
            directory=run/'rollouts'/f'step-{step:04d}'
            sample(decoder,requests,identity,TRAIN_SAMPLING,protocol['max_response_length'],directory/f'rank-{rank:02d}/responses.json',True)
            dist.barrier();scored=None
            if rank==0:
                scored=scorer.call(dict(inputs=[str(directory/f'rank-{r:02d}/responses.json') for r in range(world)],
                    expected_responses=len(selected)*8,format_penalty=True,result=str(directory/'summary.json')))
            holder=[scored];dist.broadcast_object_list(holder,src=0);scored=holder[0]
            with (directory/'scored.jsonl').open() as stream:all_rows=[json.loads(line) for line in stream]
            advantages,counts=group_advantages(all_rows,8)
            local_rows=[r for r in all_rows if (next(i for i,p in enumerate(selected) if p['index']==r['index']))%world==rank]
            optimizer.zero_grad();status(run,rank,phase='full_parameter_backward',step=step)
            for row in local_rows:
                advantage=advantages[(row['index'],row['sample_index'])]
                if advantage==0:continue
                lp=log_probs(model,row['prompt_token_ids'],row['response_token_ids'])
                loss,_=policy_loss(lp,advantage,row['rollout_log_probs'])
                (loss/(len(selected)*8)).backward()
                del loss,lp
            update=optimizer.step();torch.cuda.empty_cache()
            previous_identity=identity
            identity,weights=synchronize_model(model,decoder,run,step,rank,world)
            assert (identity!=previous_identity)==update['optimizer_updated']
            audit_decoder(model,decoder,identity,training[rank],run/f'audit.rank-{rank:02d}',step)
            record=dict(step=step,epoch=epoch+1,examples=len(selected),responses=len(selected)*8,
                train_accuracy=scored['accuracy'],accuracy_unpenalized=scored['accuracy_unpenalized'],
                clipping_rate=scored['clipping_rate'],mean_tokens=scored['mean_tokens'],
                grad_norm=update['grad_norm'],optimizer_updates=optimizer.updates,
                full_model_state_sha256=identity,rollout_full_model_state_sha256=previous_identity,
                step_seconds=time.monotonic()-tick,trainable_parameters=parameter_count,mixed_groups=counts['mixed_groups'])
            if rank==0:
                with (run/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                save(run/'live.json',record);print(json.dumps(record),flush=True)
                if wandb_run:wandb_run.log({'training/global_step':step,'training/epoch':epoch+1,
                    'critic/score/mean':record['train_accuracy'],'critic/score_unpenalized/mean':record['accuracy_unpenalized'],
                    'response_length/clip_ratio':record['clipping_rate'],'response_length/mean':record['mean_tokens'],
                    'actor/lr':protocol['learning_rate'],'actor/grad_norm':record['grad_norm'],
                    'timing/step_seconds':record['step_seconds'],'full_model/optimizer_updates':optimizer.updates},step=step)
            if args.smoke and step==2 or not args.smoke and step in protocol['eval_steps']:
                result=validate(model,decoder,scorer,validation,identity,run,step,rank,world,protocol,wandb_run,smoke=args.smoke)
                improved=result['accuracy']>state['best_accuracy']
                state=dict(best_step=step if improved else state['best_step'],
                           best_accuracy=result['accuracy'] if improved else state['best_accuracy'],
                           fails=0 if improved else state['fails']+1)
                if not args.smoke and step>=protocol['minimum_steps'] and state['fails']>=protocol['patience']:reason='patience'
                if not args.smoke and step==total:reason='max_steps'
                if rank==0:save(run/'early_stop.json',dict(state=state,last_step=step,reason=reason))
                checkpoint(model,optimizer,run,step,weights,rank,world)
            if reason:break
        assert optimizer.updates>0,'Need at least one effective full-model update'
        if rank==0:
            save(run/'complete.json',dict(complete=True,steps_run=step,optimizer_updates=optimizer.updates,
                best_step=state['best_step'],best_accuracy=state['best_accuracy'],reason=reason,
                all_parameters_trainable=True,model_parameters=parameter_count,full_model_state_sha256=identity,
                world_size=world,elapsed_seconds=time.monotonic()-start,smoke=args.smoke))
        dist.barrier()
    except BaseException as error:
        save(run/f'failure.rank-{rank:02d}.json',dict(error=repr(error),traceback=traceback.format_exc(),time=time.time()))
        raise
    finally:
        if decoder is not None:decoder.close()
        if scorer is not None:scorer.close()
        if wandb_run is not None:wandb_run.finish()
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
