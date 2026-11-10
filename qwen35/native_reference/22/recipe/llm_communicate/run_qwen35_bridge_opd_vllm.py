"""Fresh distributed worst-case smoke -> on-policy training, with fail-fast handover."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
from qwen35_bridge_opd_core import ROOT,BASE,GRPO_BASE,GRPO_122_BASE,GRPO_122_RELATIVE_BASE,GRPO_122_FP8_RELATIVE_BASE,GRPO_397_FP8_RELATIVE_BASE,GRPO_35_ARGMAX_BASE,GRPO_35_RELATIVE_BASE,MSE_ARGMAX_MAPPING,prepare,save

PYTHON=Path(os.environ.get('PLATOUR_BRIDGE_PYTHON',str(ROOT/'.runtime/qwen35-bridge/bin/python'))).absolute()
SCRIPT=Path(__file__).resolve()


def stop_workers(base,child):
    """Torchrun ranks own process groups; stop this job's full process tree."""
    table={}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            stat=(directory/'stat').read_text().rsplit(')',1)[1].split()
            cmd=(directory/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            table[int(directory.name)]=(int(stat[1]),cmd,stat[19])
        except (OSError,IndexError):
            continue
    selected={child.pid}
    selected.update(pid for pid,(_,cmd,_) in table.items() if pid!=os.getpid() and
        str(base) in cmd and ('train_qwen35_bridge_opd_vllm.py' in cmd or
                             'qwen35_bridge_opd_vllm_worker.py' in cmd or
                             'qwen35_bridge_opd_score_worker.py' in cmd))
    while True:
        descendants={pid for pid,(parent,_,_) in table.items() if parent in selected}
        if descendants<=selected:
            break
        selected.update(descendants)
    selected.intersection_update(table)
    def alive(pid):
        try:
            stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            return stat[0]!='Z' and stat[19]==table[pid][2]
        except OSError:
            return False
    for pid in sorted(selected):
        if alive(pid):
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
    deadline=time.monotonic()+20
    while any(alive(pid) for pid in selected) and time.monotonic()<deadline:
        time.sleep(.5)
    for pid in selected:
        if alive(pid):
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass


def latest_checkpoint(base, world):
    """Highest training step whose checkpoint (weights, AdamW state, every rank RNG) is complete."""
    found=[]
    for directory in (base/'train'/'checkpoints').glob('step-*'):
        names=['checkpoint.json','bridge.pt','optimizer.pt']+[f'rng.rank-{r:02d}.pt' for r in range(world)]
        if all((directory/n).exists() for n in names):
            found.append(int(directory.name.split('-')[1]))
    return max(found) if found else None


def queue(base,objective='opd',smoke_only=False,resume_step=None):
    lock=(base/'pipeline.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[]
    def stopped(signum,frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stopped)
    signal.signal(signal.SIGINT,stopped)
    try:
        protocol=json.loads((base/'protocol.json').read_text())
        gpus,world=protocol['gpu_ids'],protocol['world_size']
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),PYTHONPATH=str(ROOT/'scripts')+os.pathsep+str(SCRIPT.parent),
            OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
            WANDB_MODE=os.environ.get('PLATOUR_WANDB_MODE','online'),MATH_VERIFY_BOXED_ONLY='1' if protocol.get('math_verify_boxed_only',True) else '0')
        if protocol.get('teacher_weight_precision')=='fp8':
            env.update(HF_HOME=protocol['fp8_kernel_cache'],TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR='1')
        usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        used={int(x.split(',')[0]):int(x.split(',')[1]) for x in usage.splitlines()}
        assert all(used[i]<512 for i in gpus), f'GPUs {gpus} must be idle before a fresh run'
        if resume_step is not None:
            # The smoke of this root already passed; only the formal run continues from its checkpoint.
            assert not smoke_only and json.loads((base/'smoke_audit.json').read_text())['passed']
        modes=['train'] if resume_step is not None else (['smoke'] if smoke_only else ['smoke','train'])
        for mode in modes:
            save(base/'phase.json',dict(phase=mode,gpu_ids=gpus,teacher=protocol['teacher'],time=time.time()))
            # Single-node static rendezvous on loopback; this host's FQDN resolves only to IPv6.
            with socket.socket() as probe:
                probe.bind(('127.0.0.1',0))
                port=probe.getsockname()[1]
            command=[str(PYTHON),'-u','-m','torch.distributed.run','--nnodes=1','--master-addr=127.0.0.1',
                f'--master-port={port}',f'--nproc_per_node={world}',
                str(SCRIPT.with_name('train_qwen35_bridge_opd_vllm.py')),'--root',str(base),'--objective',objective]
            if mode=='smoke':command+=['--smoke','--stress']
            if resume_step is not None:command+=['--resume-step',str(resume_step)]
            with (base/f'{mode}.log').open('a') as log:
                child=subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True)
                jobs.append(child)
                save(base/f'{mode}.controller.json',dict(pid=child.pid,command=command,time=time.time(),resume_step=resume_step))
                code=child.wait()
                if code:raise RuntimeError(f'{mode} failed with exit {code}; see {base}/{mode}.log')
                jobs.remove(child)
            done=json.loads((base/mode/'complete.json').read_text())
            assert done['complete'] and done['rollout_and_eval_backend']=='native vLLM'
            if objective=='grpo':
                assert done['objective']=='grpo' and done['group_size']==8 and done['nonzero_gradient_updates']>0
            if mode=='smoke':
                assert done['replica_exact_weights'] and done['world_size']==world and done['singleton_tail_verified']
                profile=json.loads((base/'smoke/worst_case_profile.json').read_text())
                assert profile['passed'] and profile['vllm_resident']
                quality=json.loads((base/'smoke/evaluation/step-0002/summary.json').read_text())
                assert quality['samples_per_question']==2
                assert quality['accuracy']>0 and quality['clipping_rate']<1., 'Smoke generation quality collapsed; formal run blocked'
                save(base/'smoke_audit.json',dict(passed=True,structural=done,worst_case=profile,generation=quality))
        save(base/'phase.json',dict(phase='complete',smoke_only=smoke_only,time=time.time()))
    except BaseException as error:
        save(base/'phase.json',dict(phase='interrupted' if isinstance(error,SystemExit) else 'failed',
            error=str(error),traceback=traceback.format_exc(),time=time.time()))
        raise
    finally:
        for child in jobs:
            stop_workers(base,child)
        lock.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path)
    parser.add_argument('--objective',choices=['opd','grpo'],default='opd')
    parser.add_argument('--teacher',choices=['35b','122b','397b'],default='35b')
    parser.add_argument('--teacher-precision',choices=['bf16','fp8'],default='bf16')
    parser.add_argument('--dataset',choices=['math','dapo'],default='math')
    parser.add_argument('--max-response-length',type=int,default=4096)
    parser.add_argument('--epochs',type=int,help='DAPO epochs; default 3')
    parser.add_argument('--evaluations-per-epoch',type=int,help='DAPO validation/checkpoint frequency; default 4')
    parser.add_argument('--prepare-only',action='store_true',help='Prepare and validate data/config without GPU execution')
    parser.add_argument('--smoke-only',action='store_true',help='Run distributed rollout/update/eval and worst-case checks, then stop without formal training')
    parser.add_argument('--gpus',type=int,choices=[6,8],default=8)
    parser.add_argument('--layer-mapping',choices=['last_aligned','relative_depth_groups4',MSE_ARGMAX_MAPPING],default='last_aligned')
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--queue',action='store_true')
    early=parser.add_argument_group('validation-based early stop (DAPO only; off by default = reference behaviour)')
    early.add_argument('--early-stop',action='store_true',help='Stop after --early-stop-patience consecutive validations not better than the best')
    early.add_argument('--early-stop-patience',type=int,default=2)
    early.add_argument('--early-stop-max-steps',type=int,help='Hard cap; must be a multiple of the validation interval')
    early.add_argument('--early-stop-clip-abs',type=float,default=.05,help='Validation clipping rate above this counts as not better')
    early.add_argument('--early-stop-clip-ratio',type=float,default=2.,help='Clipping rate >= ratio x best (once best > 1%%) counts as not better')
    tune=parser.add_argument_group('run tuning (recorded in protocol.json; an existing protocol cannot be changed)')
    tune.add_argument('--skip-initial-eval',action='store_true',help='Skip the step-0 validation')
    tune.add_argument('--format-penalty',action='store_true',help='DAPO only: training reward is 0 for truncated responses or when the last non-empty line is not "Answer: X" (validation unchanged; off by default = reference behaviour)')
    tune.add_argument('--vllm-memory-fraction',type=float)
    tune.add_argument('--vllm-max-num-seqs',type=int)
    tune.add_argument('--vllm-decode-batch-size',type=int)
    tune.add_argument('--wandb-run-suffix',default='',help='Appended to the W&B run name, e.g. _run2')
    tune.add_argument('--learning-rate',type=float,help='Constant AdamW learning rate (default 1e-5); relabels lr1e5 in the W&B run name')
    resume=parser.add_argument_group('resume an interrupted formal run (no smoke; protocol.json of --root is reused as is)')
    resume.add_argument('--resume',action='store_true',help='Continue from the latest complete checkpoint under ROOT/train/checkpoints')
    resume.add_argument('--resume-step',type=int,help='Continue from this checkpoint step instead of the latest')
    args=parser.parse_args()
    if args.resume_step is not None:
        args.resume=True
    if args.teacher=='122b':assert args.objective=='grpo' and args.gpus==8
    if args.teacher=='397b':assert args.teacher_precision=='fp8' and args.gpus==8
    if args.teacher_precision=='fp8':
        assert args.teacher in ('122b','397b') and args.objective=='grpo' and args.gpus==8
        assert args.layer_mapping=='relative_depth_groups4'
    if args.layer_mapping=='relative_depth_groups4':assert args.objective=='grpo'
    if args.layer_mapping==MSE_ARGMAX_MAPPING:assert args.teacher=='35b' and args.objective=='grpo'
    teacher={'35b':'Qwen3.5-35B-A3B','122b':'Qwen3.5-122B-A10B','397b':'Qwen3.5-397B-A17B'}[args.teacher]
    default=(GRPO_35_ARGMAX_BASE if args.layer_mapping==MSE_ARGMAX_MAPPING else
             (GRPO_122_RELATIVE_BASE if args.teacher=='122b' else GRPO_35_RELATIVE_BASE)
             if args.layer_mapping=='relative_depth_groups4' else
             GRPO_122_BASE if args.teacher=='122b' else GRPO_BASE if args.objective=='grpo' else BASE)
    if args.gpus==8:
        default=default.with_name(default.name.replace('_6gpu_', '_8gpu_'))
    if args.teacher_precision=='fp8':
        default=GRPO_397_FP8_RELATIVE_BASE if args.teacher=='397b' else GRPO_122_FP8_RELATIVE_BASE
    if args.dataset=='dapo':
        assert args.objective=='grpo' and args.gpus==8 and args.layer_mapping=='relative_depth_groups4'
        precision='_fp8' if args.teacher_precision=='fp8' else ''
        default=ROOT/'outputs/llm_communicate'/f'qwen35_{args.teacher}{precision}_to_4b_dapo_grpo_relative_depth_groups4_constant1_lr1e5_bs64_g8_cap{args.max_response_length}_8gpu_vllm'
    # Preserve all historical protocols/results when the evaluation default changes.
    default=default.with_name(default.name+'_evalgeneral'+('_gc' if args.dataset=='dapo' else ''))
    if args.epochs is not None or args.evaluations_per_epoch is not None:
        assert args.dataset=='dapo'
        default=default.with_name(default.name+f'_ep{args.epochs if args.epochs is not None else 3}_eval{args.evaluations_per_epoch if args.evaluations_per_epoch is not None else 4}')
    base=(args.root or default).resolve()
    if args.resume:
        assert not args.smoke_only and not args.prepare_only and (base/'protocol.json').exists()
        protocol=json.loads((base/'protocol.json').read_text())
        assert protocol.get('method','opd')==args.objective
        if args.resume_step is None:
            args.resume_step=latest_checkpoint(base,protocol['world_size'])
            assert args.resume_step is not None, f'No complete checkpoint under {base}/train/checkpoints'
        assert not (base/'train'/'complete.json').exists(), 'Training already complete'
    elif not args.queue:
        tuning=dict(vllm_memory_fraction=args.vllm_memory_fraction,vllm_max_num_seqs=args.vllm_max_num_seqs,
            vllm_decode_batch_size=args.vllm_decode_batch_size,wandb_run_suffix=args.wandb_run_suffix,
            skip_initial_eval=args.skip_initial_eval,format_penalty=args.format_penalty,
            learning_rate=args.learning_rate)
        if args.early_stop:
            assert args.dataset=='dapo' and args.evaluations_per_epoch, 'Early stop needs --dataset dapo and --evaluations-per-epoch'
            # The validation interval is taken from the protocol's own eval_steps inside prepare().
            tuning['early_stop']=dict(patience=args.early_stop_patience,max_steps=args.early_stop_max_steps,
                clip_abs=args.early_stop_clip_abs,clip_ratio=args.early_stop_clip_ratio)
        else:
            assert args.early_stop_max_steps is None, '--early-stop-max-steps requires --early-stop'
        # Keep explicit zeros: vllm_decode_batch_size=0 means "queue every request" and is not "unset".
        tuning={k:v for k,v in tuning.items() if v is not None and v is not False and v!=''}
        protocol=prepare(base,args.objective,teacher,args.layer_mapping,gpu_count=args.gpus,teacher_precision=args.teacher_precision,
            dataset=args.dataset,max_response_length=args.max_response_length,epochs=args.epochs,evaluations_per_epoch=args.evaluations_per_epoch,
            tuning=tuning)
    if args.prepare_only:
        assert not args.queue and not args.detach
        print(json.dumps(dict(prepared=True,root=str(base),protocol=protocol)),flush=True)
        return
    if args.detach:
        with (base/'controller.log').open('a') as log:
            proc=subprocess.Popen([str(PYTHON),'-u',str(SCRIPT),'--queue','--root',str(base),'--objective',args.objective,'--teacher',args.teacher,
                '--layer-mapping',args.layer_mapping,'--gpus',str(args.gpus),'--teacher-precision',args.teacher_precision,
                '--dataset',args.dataset,'--max-response-length',str(args.max_response_length)]+(['--smoke-only'] if args.smoke_only else [])
                +(['--resume-step',str(args.resume_step)] if args.resume else []),
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(base/'controller.json',dict(pid=proc.pid,started=time.time(),gpu_ids=json.loads((base/'protocol.json').read_text())['gpu_ids'],
            resume_step=args.resume_step if args.resume else None))
        print(json.dumps(dict(pid=proc.pid,root=str(base),resume_step=args.resume_step if args.resume else None)),flush=True)
    else:
        queue(base,args.objective,smoke_only=args.smoke_only,resume_step=args.resume_step if args.resume else None)


if __name__=='__main__':main()
