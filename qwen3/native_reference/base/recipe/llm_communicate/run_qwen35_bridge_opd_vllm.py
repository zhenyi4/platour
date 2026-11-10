"""Fresh distributed worst-case smoke -> on-policy training, with fail-fast handover."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
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


def validate_completed_smoke(base,protocol):
    """Reuse prior full-batch smoke only with the same adapter/data/settings."""
    source=Path(protocol['reuse_completed_smoke_root'])
    old=json.loads((source/'protocol.json').read_text())
    done=json.loads((source/'smoke/complete.json').read_text())
    profile=json.loads((source/'smoke/worst_case_profile.json').read_text())
    quality=json.loads((source/'smoke/evaluation/step-0002/summary.json').read_text())
    assert done['complete'] and done['smoke'] and done['objective']=='grpo'
    assert done['world_size']==protocol['world_size'] and done['replica_exact_weights']
    assert done['group_size']==8 and done['nonzero_gradient_updates']>0 and done['singleton_tail_verified']
    assert profile['passed'] and profile['vllm_resident']
    assert quality['complete'] and quality['accuracy']>0 and quality['clipping_rate']<1.
    required=['bridge_recipe','student_checkpoint','trainable_parameters','world_size','batch_size',
        'group_size','max_prompt_length','max_response_length','train_sampling','eval_sampling',
        'train_prompts_sha256','eval_prompts_sha256','layer_map']
    optional=['virtual_hidden_shape','virtual_hidden_std','virtual_hidden_statistics_sha256',
        'template_prefix_ids','template_suffix_ids','actor_training_mode','lora_rank','lora_alpha','lora_target_modules',
        'gaussian_statistics_sha256','teacher_checkpoint','variant','projection_initialization','initialization']
    for key in required+[k for k in optional if k in protocol]:
        assert protocol.get(key)==old.get(key),f'Reused smoke configuration differs: {key}'
    assert protocol.get('learning_rate',1e-5)==old.get('learning_rate',1e-5)
    recipes={
        'gsm8k_math_platour_grpo':['gsm8k_math_platour_lora_grpo.py','chemistry_platour_grpo.py','run_chemistry_235b_17b_hidden_grpo.py','qwen3_235b_trivia_hidden_sft.py','qwen3_4b_06b_grpo.py'],
        'chemistry_dense_gaussian_platour_grpo':['chemistry_platour_grpo.py','qwen3_4b_06b_grpo.py'],
        'qwen35_virtual_teacher':['qwen35_virtual_teacher_bridge.py'],
        'qwen35_template_virtual_teacher':['qwen35_template_virtual_teacher_bridge.py','qwen35_virtual_teacher_bridge.py','qwen35_template_embedding_tuning.py'],
        'qwen35_template_embedding_tuning':['qwen35_template_embedding_tuning.py'],
        'qwen35_native_actor':['qwen35_native_actor_grpo.py','qwen35_native_actor_vllm_worker.py'],
    }
    recipe=protocol.get('bridge_recipe')
    if recipe is None:
        assert protocol['teacher']=='Qwen3.5-122B-A10B' and protocol['teacher_weight_precision']=='fp8'
        implementation=['qwen35_122b_4b_math_bridge.py','qwen35_last_aligned_math_bridge.py',
            'train_qwen35_bridge_opd_vllm.py','qwen35_bridge_opd_vllm_worker.py',
            'qwen35_bridge_opd_score_worker.py','generate_qwen35_train_suite.py','safe_math_powers.py']
    else:
        implementation=recipes[recipe]
    paths=implementation+['qwen35_student.py','qwen35_direct_math_bridge.py']
    compared=[]
    for name in paths:
        rel=Path('recipe/llm_communicate')/name
        prior=(source/'source_snapshot'/rel).read_text()
        current=(ROOT/rel).read_text()
        if name=='qwen35_bridge_opd_vllm_worker.py':
            cache_block="""    if os.environ.get('PLATOUR_ISOLATE_COMPILE_CACHES') == '1':
        for key in ['TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'VLLM_CACHE_ROOT']:
            path = Path(os.environ[key]) / ('gpu-' + os.environ['CUDA_VISIBLE_DEVICES'].replace(',', '-'))
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
"""
            prior,current=prior.replace(cache_block,''),current.replace(cache_block,'')
        if name=='chemistry_platour_grpo.py' and prior!=current and protocol.get('variant') in ('8b','17b'):
            # Gaussian-only maximum-length stress fixture was added after
            # the real-teacher smoke. It is never called by dense bridges.
            import ast
            def without_gaussian_stress(text):
                tree=ast.parse(text)
                for node in tree.body:
                    if isinstance(node,ast.ClassDef) and node.name=='GaussianBridge':
                        node.body=[n for n in node.body if not (isinstance(n,ast.FunctionDef) and n.name=='build_stress_prompt')]
                return ast.dump(tree,include_attributes=False)
            assert without_gaussian_stress(prior)==without_gaussian_stress(current)
            compared.append(str(rel)+' (all AST identical except unused Gaussian stress fixture)')
            continue
        if name=='qwen35_native_actor_vllm_worker.py':
            # Scalar-compatible byte hashing preserves all tensor bytes; the
            # added checkpoint unpacking is exclusive to the full actor.
            old='value.detach().cpu().contiguous().view(torch.uint8)'
            new='value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)'
            prior=prior.replace(old,new)
            block="    if 'actor' in state:\n        assert state['mode']=='full'\n        state=state['actor']\n"
            prior,current=prior.replace(block,''),current.replace(block,'')
        if name=='qwen35_native_actor_grpo.py':
            # Opt-in transient file retention does not change actor math.
            line="    actor.full_actor_snapshot_retention=protocol.get('full_actor_snapshot_retention')\n"
            prior,current=prior.replace(line,''),current.replace(line,'')
            if current!=prior:
                import ast
                def actor_math(text):
                    tree=ast.parse(text)
                    selected=[]
                    for node in tree.body:
                        if isinstance(node,ast.ClassDef) and node.name in ('LoRALinear','MasterAdamW','NativeActor'):
                            if node.name=='NativeActor':
                                node.body=[m for m in node.body if not isinstance(m,ast.FunctionDef)
                                           or m.name not in ('generate_native_actor','metadata')]
                            selected.append(ast.dump(node,include_attributes=False))
                        elif isinstance(node,ast.FunctionDef) and node.name=='load_bridge':
                            selected.append(ast.dump(node,include_attributes=False))
                    assert len(selected)==4
                    return selected
                # Native LoRA's older snapshot predates the full-only shared
                # file sync and revised diagnostics. Its actual actor,
                # optimizer, response forward, and loading math must match.
                assert actor_math(current)==actor_math(prior),f'Actor math changed since smoke: {name}'
                compared.append(str(rel)+' (actor/optimizer/loader AST; I/O and diagnostics excluded)')
                continue
        assert current==prior,f'Actor implementation changed since smoke: {name}'
        compared.append(str(rel))
    result=dict(passed=True,source=str(source),full_batch_and_tail_verified=True,
        settings_identical=True,actor_math_sources_identical=compared,
        smoke=done,worst_case=profile,small_evaluation=quality,
        reused_smoke_optimizer_or_weights=False,formal_initialization='fresh')
    save(base/'reused_smoke_audit.json',result)
    return result


def queue(base,objective='opd',smoke_only=False):
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
            WANDB_MODE=os.environ.get('WANDB_MODE','online'),MATH_VERIFY_BOXED_ONLY='1' if protocol.get('math_verify_boxed_only',True) else '0')
        if protocol.get('teacher_weight_precision')=='fp8':
            env.update(HF_HOME=protocol['fp8_kernel_cache'],TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR='1')
        usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        used={int(x.split(',')[0]):int(x.split(',')[1]) for x in usage.splitlines()}
        assert all(used[i]<512 for i in gpus), f'GPUs {gpus} must be idle before a fresh run'
        modes=['smoke'] if smoke_only else ['smoke','train']
        if protocol.get('resume_from_checkpoint'):
            assert not smoke_only and protocol['bridge_recipe']=='qwen235_17b_chemistry_hidden_grpo'
            from run_chemistry_235b_17b_hidden_grpo import validate_continuation
            validate_continuation(base,protocol)
            modes=['train']
        if not smoke_only and protocol.get('reuse_completed_smoke_root'):
            assert objective=='grpo'
            validate_completed_smoke(base,protocol)
            modes=['train']
        for mode in modes:
            if protocol.get('verify_source_hashes',False):
                fingerprints=json.loads((base/'source_sha256.json').read_text())
                assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
                           for path,digest in fingerprints.items()), 'Source changed since preparation; use a fresh run'
            save(base/'phase.json',dict(phase=mode,gpu_ids=gpus,teacher=protocol['teacher'],time=time.time()))
            command=[str(PYTHON),'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
                str(SCRIPT.with_name('train_qwen35_bridge_opd_vllm.py')),'--root',str(base),'--objective',objective]
            if mode=='smoke':command+=['--smoke','--stress']
            with (base/f'{mode}.log').open('a') as log:
                child=subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True)
                jobs.append(child)
                save(base/f'{mode}.controller.json',dict(pid=child.pid,command=command,time=time.time()))
                code=child.wait()
                if code:raise RuntimeError(f'{mode} failed with exit {code}; see {base}/{mode}.log')
                jobs.remove(child)
            done=json.loads((base/mode/'complete.json').read_text())
            assert done['complete'] and done['rollout_and_eval_backend']=='native vLLM'
            if objective=='grpo':
                assert done['objective']=='grpo' and done['group_size']==8 and done['nonzero_gradient_updates']>0
            else:
                assert done['objective']=='opd' and done['group_size']==protocol['group_size']
                assert done['nonzero_gradient_updates']>0
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
    args=parser.parse_args()
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
    if not args.queue:
        protocol=prepare(base,args.objective,teacher,args.layer_mapping,gpu_count=args.gpus,teacher_precision=args.teacher_precision,
            dataset=args.dataset,max_response_length=args.max_response_length,epochs=args.epochs,evaluations_per_epoch=args.evaluations_per_epoch)
    if args.prepare_only:
        assert not args.queue and not args.detach
        print(json.dumps(dict(prepared=True,root=str(base),protocol=protocol)),flush=True)
        return
    if args.detach:
        with (base/'controller.log').open('a') as log:
            proc=subprocess.Popen([str(PYTHON),'-u',str(SCRIPT),'--queue','--root',str(base),'--objective',args.objective,'--teacher',args.teacher,
                '--layer-mapping',args.layer_mapping,'--gpus',str(args.gpus),'--teacher-precision',args.teacher_precision,
                '--dataset',args.dataset,'--max-response-length',str(args.max_response_length)]+(['--smoke-only'] if args.smoke_only else []),
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(base/'controller.json',dict(pid=proc.pid,started=time.time(),gpu_ids=json.loads((base/'protocol.json').read_text())['gpu_ids']))
        print(json.dumps(dict(pid=proc.pid,root=str(base))),flush=True)
    else:
        queue(base,args.objective,smoke_only=args.smoke_only)


if __name__=='__main__':main()
