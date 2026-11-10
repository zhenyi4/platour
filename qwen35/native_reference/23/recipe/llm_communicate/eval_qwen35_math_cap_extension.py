"""Re-generate only 16K-truncated HMMT/IMO responses at 32K; preserve other samples and provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BENCHMARKS = ('hmmt', 'imo')
INPUT_FIELDS = ('response','response_token_ids','tokens','finish_reason','stop_reason','acc','score','pred',
    'gold_missing','scoring_timed_out','acc_answer','acc_strict','acc_full','score_full','pred_full','pred_answer',
    'answer_format_ok','answer_metric_timeout','answer_scoring_version','answer_parse_failed','gold_parse_failed',
    'verification_mode','normalized_answer','bridge_state_sha256')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def identity(row):
    return row['index'], row['sample_index']


def select_truncated(original, old_cap=16384):
    assert len({identity(row) for row in original})==len(original)
    selected = []
    for row in original:
        if row['finish_reason']=='length':
            assert row['tokens']==old_cap==len(row['response_token_ids'])
            selected.append({key:value for key,value in row.items() if key not in INPUT_FIELDS})
    return selected


def merge_results(original, regenerated, samples, cap=32768):
    selected = {identity(row) for row in original if row['finish_reason']=='length'}
    updated = {identity(row):row for row in regenerated}
    assert len(updated)==len(regenerated) and set(updated)==selected, 'Re-run exactly the truncated sample IDs'
    combined = []
    for old in original:
        current = updated.get(identity(old),old)
        if identity(old) in selected:
            for key in ('gold','prompt_token_ids','sampling_seed'):
                assert current[key]==old[key], f'Changed {key} for selected sample'
            assert 0<current['tokens']<=cap and current['tokens']==len(current['response_token_ids'])
            assert current['finish_reason']!='length' or current['tokens']==cap
            current = dict(current,cap_extension_source_tokens=old['tokens'],cap_extension_regenerated=True,
                           cap_extension_old_acc=old['acc'])
        combined.append(current)
    groups = {}
    for row in combined:
        groups.setdefault(row['index'],[]).append(row['acc'])
    assert all(len(values)==samples for values in groups.values())
    count = len(combined)
    result = dict(complete=True,responses=count,questions=len(groups),samples_per_question=samples,
        accuracy=sum(row['acc'] for row in combined)/count,
        pass_at_k=sum(any(values) for values in groups.values())/len(groups),
        clipping_rate=sum(row['finish_reason']=='length' for row in combined)/count,
        clipped=sum(row['finish_reason']=='length' for row in combined),mean_tokens=sum(row['tokens'] for row in combined)/count,
        max_new_tokens=cap,regenerated_responses=len(selected),reused_stopped_responses=count-len(selected),
        evaluation_method='selective_32k_rerun_of_16k_truncated_responses',
        full_32k_regeneration=False,scoring_version='answer_math_v2',
        **{f'avg_at_{samples}':sum(row['acc'] for row in combined)/count,
           f'pass_at_{samples}':sum(any(values) for values in groups.values())/len(groups)},
        newly_correct=sum(row['acc']==1 for key,row in updated.items()),
        old_accuracy=sum(row['acc'] for row in original)/count,
        old_clipping_rate=sum(row['finish_reason']=='length' for row in original)/count)
    for field in ('competition','category'):
        if not all(field in row for row in combined):continue
        result['per_'+field] = {}
        for value in sorted({row[field] for row in combined}):
            part=[row for row in combined if row[field]==value]
            ids={row['index'] for row in part}
            result['per_'+field][value]=dict(accuracy=sum(row['acc'] for row in part)/len(part),
                clipping_rate=sum(row['finish_reason']=='length' for row in part)/len(part),responses=len(part),questions=len(ids))
    return combined,result


def configure(plan):
    workspace = Path(plan['workspace'])
    sys.path[:0] = [str(workspace/'recipe/llm_communicate'),str(workspace/'scripts')]
    for relative,digest in plan['source_files'].items():
        assert sha(workspace/relative)==digest, f'Execution source changed: {relative}'
    assert sha(Path(__file__))==plan['driver_sha256']
    return workspace


def prepare(args):
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    assert not (root/'plan.json').exists(),'Extension already prepared; use a new output directory'
    source_files = {}
    manifest = read(args.workspace.parent/'source_manifest.json')
    for relative,digest in manifest.items():
        path=args.workspace.parent/relative
        if path.is_relative_to(args.workspace):source_files[str(path.relative_to(args.workspace))]=digest
    paths={key:os.environ.get(variable,str(args.workspace/'.runtime/multinode-portable'/name/'bin/python'))
           for key,variable,name in [('bridge','PLATOUR_BRIDGE_PYTHON','qwen35-bridge'),
               ('generation','PLATOUR_GENERATION_PYTHON','qwen35-generation'),('scoring','PLATOUR_SCORING_PYTHON','nemotron-smoke')]}
    plan=dict(workspace=str(args.workspace),source_files=source_files,driver_sha256=sha(Path(__file__)),
        python_paths=paths,wandb_key_file=str(args.wandb_key_file.resolve()),
        max_new_tokens=args.max_new_tokens,models={})
    for label,source in [('native4b',args.native_4b),('native35b',args.native_35b),('bridge4b',args.bridge)]:
        source=source.resolve();model={}
        for key in BENCHMARKS:
            directory=root/label/key
            protocol=read(source/key/'protocol.json')
            original=rows(source/key/'scored.jsonl')
            assert len(original)==protocol['responses']
            assert protocol['max_new_tokens']==16384 and protocol['prompt_format']=='answer'
            summary=read(source/key/'summary.json')
            assert summary.get('scoring_version',summary.get('answer_scoring_version'))=='answer_math_v2'
            selected=select_truncated(original)
            assert len(selected)==sum(row['finish_reason']=='length' for row in original)
            request=dict(protocol,max_new_tokens=args.max_new_tokens,max_model_len=args.max_new_tokens+2048,
                max_num_seqs=4,decode_batch_size=4,cap_extension=True)
            save(directory/'protocol.json',request)
            save(directory/'requests.json',selected)
            model[key]=dict(source=str(source/key),source_scored_sha256=sha(source/key/'scored.jsonl'),
                source_protocol_sha256=sha(source/key/'protocol.json'),selected=len(selected),
                selected_ids=[list(identity(row)) for row in selected],responses=protocol['responses'])
        plan['models'][label]=model
    configure(plan)
    save(root/'plan.json',plan)
    save(root/'status.json',dict(stage='prepared',counts={model:{k:v['selected'] for k,v in value.items()} for model,value in plan['models'].items()}))
    return plan


def worker(args,plan):
    label=args.model;rank=args.rank;root=args.output;configure(plan)
    requests={key:[row for row in read(root/label/key/'requests.json') if row['index']%8==rank] for key in BENCHMARKS}
    if not any(requests.values()):return
    protocol=read(root/label/'hmmt/protocol.json')
    native=not protocol.get('checkpoint')
    os.environ['CUDA_VISIBLE_DEVICES']=str(rank) if native else ','.join(map(str,range(8)))
    import torch
    import eval_qwen35_bridge_aime135 as suite
    from train_qwen35_bridge_opd_vllm import RPC, generate
    from qwen35_bridge_opd_core import weight_digest
    from transformers import AutoTokenizer
    torch.cuda.set_device(0 if native else rank)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    engine=llm=None
    started=time.monotonic()
    runtime_dir=root/label/'generation'/f'rank-{rank:02d}'
    runtime_dir.mkdir(parents=True,exist_ok=True)
    try:
        if native:
            from vllm import LLM, SamplingParams
            for name,digest in protocol['model_file_sha256'].items():
                assert sha(Path(protocol['model_path'])/name)==digest
            llm=LLM(model=protocol['model_path'],dtype='bfloat16',tensor_parallel_size=1,
                max_model_len=protocol['max_model_len'],max_num_seqs=4,max_num_batched_tokens=4096,
                gpu_memory_utilization=protocol['vllm_memory_fraction'],enable_prefix_caching=False,
                seed=42,disable_log_stats=True,language_model_only=True,
                limit_mm_per_prompt={'image':0,'video':0},attention_config={'backend':'FLASH_ATTN'})
        else:
            from qwen35_35b_4b_math_bridge import load_bridge
            bridge,model_runtime=load_bridge(Path(plan['workspace']),layer_mapping=protocol['layer_mapping'])
            assert bridge.layer_map==protocol['layer_map']
            assert sha(protocol['checkpoint'])==protocol['checkpoint_sha256']
            state=torch.load(protocol['checkpoint'],map_location='cpu',weights_only=True)
            bridge.maps.load_state_dict(state['P'],strict=True)
            del state
            bridge.eval().requires_grad_(False)
            assert weight_digest(bridge)==protocol['bridge_state_sha256']
            tokenizer=AutoTokenizer.from_pretrained(Path(plan['workspace'])/'data/models/Qwen3.5-4B',local_files_only=True)
            eos=bridge.student.generation_config.eos_token_id
            eos=[eos] if isinstance(eos,int) else list(eos)
            engine=RPC('qwen35_bridge_opd_vllm_worker.py',runtime_dir/'vllm',gpu=rank,
                memory_fraction=protocol['vllm_memory_fraction'],max_model_len=protocol['max_model_len'],
                max_num_seqs=4,decode_batch_size=4)
        save(runtime_dir/'runtime.json',dict(load_seconds=time.monotonic()-started,
             model=label,rank=rank,max_model_len=protocol['max_model_len'],inference_only=True))
        for key,batch in requests.items():
            if not batch:continue
            directory=root/label/key/'generation'/f'rank-{rank:02d}'
            result=directory/'responses.json'
            current=read(root/label/key/'protocol.json')
            if not result.exists():
                if native:
                    parameters=[SamplingParams(n=1,max_tokens=current['max_new_tokens'],seed=row['sampling_seed'],
                                                **current['sampling']) for row in batch]
                    outputs=llm.generate([dict(prompt_token_ids=row['prompt_token_ids']) for row in batch],parameters,use_tqdm=False)
                    generated=[]
                    for row,item in zip(batch,outputs,strict=True):
                        answer=item.outputs[0]
                        generated.append(dict(row,response=answer.text,response_token_ids=list(answer.token_ids),
                            tokens=len(answer.token_ids),finish_reason=answer.finish_reason,stop_reason=answer.stop_reason,
                            bridge_state_sha256=None))
                    save(result,dict(complete=True,rows=generated))
                else:
                    generate(bridge,engine,batch,current['bridge_state_sha256'],current['sampling'],
                             current['max_new_tokens'],directory,tokenizer.pad_token_id,eos)
            payload=read(result)
            assert payload['complete']
            suite.CAP=current['max_new_tokens']
            suite.validate_rows(payload['rows'],batch,current)
        save(runtime_dir/'complete.json',dict(complete=True,seconds=time.monotonic()-started))
    finally:
        if engine is not None:engine.close()
        if llm is not None:llm.llm_engine.engine_core.shutdown()


def score_selected(directory,protocol,inputs):
    from train_qwen35_bridge_opd_vllm import RPC
    scorer=RPC('qwen35_bridge_opd_score_worker.py',directory/'scoring'/f'worker-{os.getpid()}',boxed_only=False)
    result=directory/'scoring'/f'attempt-{os.getpid()}'/'summary.json'
    result.parent.mkdir(parents=True,exist_ok=True)
    try:
        summary=scorer.call(dict(inputs=[str(p) for p in inputs],expected_responses=protocol['selected_count'],
                                result=str(result),metric='answer'))
    finally:scorer.close()
    assert summary['complete'] and summary['scoring_timeouts']==0 and summary['answer_metric_timeouts']==0
    return rows(result.with_name('scored.jsonl'))


def publish(root,label,results):
    import wandb
    plan=read(root/'plan.json')
    checkpoint=read(root/'bridge4b/hmmt/protocol.json')
    info=read(Path(checkpoint['training_root'])/'train/wandb.json')
    key=Path(plan['wandb_key_file'])
    os.environ['WANDB_API_KEY']=key.read_text().strip()
    run=wandb.Api(timeout=60).run(f"{info['entity']}/{info['project']}/{info['id']}")
    prefix='test' if label=='bridge4b' else f'test_baseline/native_{"4b" if label=="native4b" else "35b"}'
    backup=root/'wandb_backups'/f'{label}.json'
    if not backup.exists():
        before={k:v for k,v in dict(run.summary).items() if k.startswith(prefix+'/') and any(f'/{b}/' in k for b in BENCHMARKS)}
        save(backup,dict(before=before,run_id=run.id,created=time.time()))
    previous=read(backup)['before']
    metrics={f'{prefix}_cap16k/{k[len(prefix)+1:]}':v for k,v in previous.items()}
    for benchmark,result in results.items():
        samples=result['samples_per_question']
        for key in (f'avg_at_{samples}',f'pass_at_{samples}','clipping_rate','mean_tokens','max_new_tokens',
                    'evaluation_method','full_32k_regeneration','regenerated_responses','reused_stopped_responses'):
            metrics[f'{prefix}/{benchmark}/{key}']=result[key]
        metrics[f'{prefix}/{benchmark}/scoring_version']='answer_math_v2'
    run.summary.update(metrics);run.update()


def run(args,plan):
    import fcntl
    import eval_qwen35_bridge_aime135 as suite
    root=args.output
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        all_results={}
        for label in plan['models']:
            configure(plan)
            complete=root/label/'complete.json'
            if complete.exists():
                all_results[label]=read(complete)['results']
                continue
            while True:
                usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
                if all(int(x)<512 for x in usage.split()):break
                save(root/'status.json',dict(stage='waiting_idle_gpu',model=label,time=time.time()))
                time.sleep(15)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,range(8))),
                PYTHONPATH=os.pathsep.join([str(Path(plan['workspace'])/'recipe/llm_communicate'),str(Path(plan['workspace'])/'scripts')]),
                PATH=str(Path(plan['python_paths']['generation']).parent)+os.pathsep+os.environ['PATH'],
                PLATOUR_GENERATION_PYTHON=plan['python_paths']['generation'],PLATOUR_SCORING_PYTHON=plan['python_paths']['scoring'],
                HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',MATH_VERIFY_BOXED_ONLY='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',
                OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
            jobs=[]
            try:
                all_requests={key:read(root/label/key/'requests.json') for key in BENCHMARKS}
                python=plan['python_paths']['bridge'] if label=='bridge4b' else plan['python_paths']['generation']
                for rank in range(8):
                    if not any(row['index']%8==rank for batch in all_requests.values() for row in batch):continue
                    directory=root/label/'generation'/f'rank-{rank:02d}'
                    if (directory/'complete.json').exists():continue
                    directory.mkdir(parents=True,exist_ok=True)
                    output=(directory/'worker.log').open('a')
                    process=subprocess.Popen([python,'-u',str(Path(__file__).resolve()),'worker','--output',str(root),
                        '--model',label,'--rank',str(rank)],cwd=plan['workspace'],env=env,
                        stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                    jobs.append((process,output))
                save(root/label/'controller.json',dict(worker_pids=[p.pid for p,_ in jobs],started=time.time()))
                save(root/'status.json',dict(stage='regenerating_truncated_32k',model=label,
                     counts={key:len(batch) for key,batch in all_requests.items()},time=time.time()))
                while any(p.poll() is None for p,_ in jobs):
                    for process,_ in jobs:
                        if process.poll() not in (None,0):raise RuntimeError(f'{label} worker exited {process.returncode}')
                    time.sleep(2)
                assert all(p.returncode==0 for p,_ in jobs)
                results={}
                for key in BENCHMARKS:
                    directory=root/label/key;metadata=plan['models'][label][key];source=Path(metadata['source'])
                    assert sha(source/'scored.jsonl')==metadata['source_scored_sha256']
                    assert sha(source/'protocol.json')==metadata['source_protocol_sha256']
                    original=rows(source/'scored.jsonl')
                    protocol=read(directory/'protocol.json')
                    selected=all_requests[key]
                    regenerated=[]
                    if selected:
                        inputs=[]
                        for rank in range(8):
                            batch=[row for row in selected if row['index']%8==rank]
                            if not batch:continue
                            path=directory/'generation'/f'rank-{rank:02d}'/'responses.json'
                            payload=read(path);assert payload['complete']
                            suite.CAP=32768;suite.validate_rows(payload['rows'],batch,protocol)
                            inputs.append(path)
                        regenerated=score_selected(directory,dict(protocol,selected_count=len(selected)),inputs)
                    combined,summary=merge_results(original,regenerated,protocol['samples_per_question'])
                    assert summary['responses']==metadata['responses']
                    summary.update(source=metadata['source'],source_scored_sha256=metadata['source_scored_sha256'],
                                   protocol=protocol,extension_scoring_version='answer_math_v2')
                    temporary=directory/'scored.jsonl.tmp'
                    temporary.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in combined))
                    temporary.replace(directory/'scored.jsonl')
                    save(directory/'summary.json',summary)
                    save(directory/'complete.json',dict(complete=True,scored_sha256=sha(directory/'scored.jsonl'),
                         protocol_sha256=sha(directory/'protocol.json'),finished=time.time()))
                    results[key]=summary
                save(root/label/'results.json',dict(model=label,results=results))
                save(complete,dict(complete=True,results=results,finished=time.time()))
                all_results[label]=results
                try:publish(root,label,results)
                except Exception as error:save(root/f'publish_failed-{label}.json',dict(error=repr(error)))
                save(root/'status.json',dict(stage='model_complete',model=label,
                    results={b:{k:r[k] for k in ('accuracy','clipping_rate','newly_correct')} for b,r in results.items()},time=time.time()))
            finally:suite.terminate(jobs)
        save(root/'complete.json',dict(complete=True,results=all_results,finished=time.time()))
        save(root/'status.json',dict(stage='complete',time=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','run','worker'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workspace',type=Path)
    parser.add_argument('--native-4b',type=Path)
    parser.add_argument('--native-35b',type=Path)
    parser.add_argument('--bridge',type=Path)
    parser.add_argument('--max-new-tokens',type=int,choices=(32768,),default=32768)
    parser.add_argument('--model',choices=('native4b','native35b','bridge4b'))
    parser.add_argument('--rank',type=int)
    parser.add_argument('--wandb-key-file',type=Path)
    args=parser.parse_args();args.output=args.output.resolve()
    if args.stage=='prepare':
        assert args.workspace and args.native_4b and args.native_35b and args.bridge and args.wandb_key_file
        args.workspace=args.workspace.resolve()
        plan=prepare(args)
        print(json.dumps(dict(prepared=True,counts={model:{b:d['selected'] for b,d in value.items()}
              for model,value in plan['models'].items()})),flush=True)
    else:
        plan=read(args.output/'plan.json');configure(plan)
        if args.stage=='worker':worker(args,plan)
        else:run(args,plan)


if __name__=='__main__':
    def stopped(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stopped)
    try:main()
    except BaseException as error:
        import traceback
        parser_root=Path(sys.argv[sys.argv.index('--output')+1]) if '--output' in sys.argv else None
        if parser_root is not None:
            if 'worker' in sys.argv:
                rank=sys.argv[sys.argv.index('--rank')+1]
                model=sys.argv[sys.argv.index('--model')+1]
                save(parser_root/model/'generation'/f'rank-{rank}'/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
            else:save(parser_root/'status.json',dict(stage='failed',error=repr(error),traceback=traceback.format_exc()))
        raise
