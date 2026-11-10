"""Qwen3.5 non-thinking GSM8K/MATH train generation; real smoke benchmarks first."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
CPU=ROOT/'.runtime/nemotron-smoke/bin/python'
INFER=Path(os.environ.get('PLATOUR_GENERATION_PYTHON',str(ROOT/'.runtime/qwen35-generation/bin/python'))).absolute()
BASE=ROOT/'outputs/llm_communicate/qwen35_train_generation'
NAMES=['Qwen3.5-9B','Qwen3.5-27B','Qwen3.5-35B-A3B','Qwen3.5-122B-A10B']
TP={name:4 if '122B' in name else 1 for name in NAMES}
COUNTS={'gsm8k':7473,'math':7500}
CAPS={'gsm8k':2048,'math':4096}
SUFFIX='\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
SMOKE=64


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def read(path):
    if not path.exists():return []
    with path.open() as handle:return [json.loads(line) for line in handle if line.strip()]


def jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows));temp.replace(path)


def immutable(path,value):
    if path.exists():assert json.loads(path.read_text())==value,f'Manifest changed: {path}'
    else:save(path,value)


def prepare():
    from transformers import AutoTokenizer
    gsm=ROOT/'outputs/llm_communicate/single_gpu_regression/teacher_data/responses.jsonl'
    assert sha(gsm)=='63f2805adc8ad342ac918a9b8dc7a57c8ca2a4030e2cec8721deea734c582fc7'
    archived={'gsm8k':read(gsm)}
    math_source=ROOT/'data/hidden_bridge_math/responses.jsonl.gz'
    with gzip.open(math_source,'rb') as handle:raw=handle.read()
    assert hashlib.sha256(raw).hexdigest()=='850eff2fb530883684d8069ab34eceaae0afb1dd93ca011a92730b04c0c158b6'
    archived['math']=[json.loads(line) for line in raw.splitlines() if line.strip()]
    sources={'gsm8k':dict(path=str(gsm),sha256=sha(gsm),repo='openai/gsm8k',config='main',split='train',
                revision='740312add88f781978c0658806c59bc2815b9866'),
             'math':dict(path=str(math_source),sha256=sha(math_source),repo='DigitalLearningGmbH/MATH-lighteval',
                split='train',revision='0530c78699ea5e8eb5530600900e1f328b48acad')}
    canonical={}
    for task,rows in archived.items():
        rows.sort(key=lambda row:row['index'])
        assert [row['index'] for row in rows]==list(range(COUNTS[task]))
        records=[]
        for row in rows:
            text=row['input_text']
            suffix=next((s for s in (SUFFIX," Let's think step by step and output the final answer within \\boxed{}.")
                         if text.endswith(s)),None)
            assert suffix is not None
            question=text[:-len(suffix)].strip();assert question
            gold=str(row.get('gold',row.get('gts',row.get('gold_answer_text',''))))
            records.append(dict(index=row['index'],id=f'{task}_train_{row["index"]:06d}',task=task,
                question=question,input_text=question+SUFFIX,gold=gold,
                gold_answer_text=row.get('gold_answer_text',''),sample_index=0,sampling_seed=42+row['index']))
        canonical[task]=records
        target=BASE/'inputs'/f'{task}.jsonl'
        if target.exists():assert read(target)==records
        else:jsonl(target,records)
    downloads=json.loads((ROOT/'outputs/model_downloads/qwen35_suite/manifest.json').read_text())
    for name in NAMES:
        spec=next(item for item in downloads['models'] if item['name']==name)
        model=ROOT/'data/models'/name
        assert all((model/file).is_file() and (model/file).stat().st_size==size for file,size in spec['files'].items())
        tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
        config=json.loads((model/'config.json').read_text())
        for task,records in canonical.items():
            prompts=[]
            for row in records:
                ids=tokenizer.apply_chat_template([dict(role='user',content=row['input_text'])],tokenize=True,
                    add_generation_prompt=True,enable_thinking=False,return_dict=False)
                rendered=tokenizer.decode(ids)
                assert rendered.endswith('<think>\n\n</think>\n\n')
                assert rendered.count('<|im_start|>user\n')==1
                prompts.append(dict(**row,prompt_token_ids=ids))
            longest=max(range(len(prompts)),key=lambda i:len(prompts[i]['prompt_token_ids']))
            sampled=random.Random(42).sample([i for i in range(len(prompts)) if i!=longest],SMOKE-1)
            selected=sorted([*sampled,longest])
            maximum=max(len(row['prompt_token_ids']) for row in prompts)
            path=BASE/'inputs'/name/f'{task}.jsonl'
            if path.exists():assert read(path)==prompts
            else:jsonl(path,prompts)
            protocol=dict(model=spec['repo'],revision=spec['revision'],path=str(model),task=task,
                source=sources[task],prompts_sha256=sha(path),questions=COUNTS[task],samples_per_question=1,
                enable_thinking=False,native_chat_template=True,prompt_instruction=SUFFIX,
                max_new_tokens=CAPS[task],maximum_prompt_tokens=maximum,
                max_model_len=4096+math.ceil(maximum/128)*128,
                temperature=.7,top_p=.8,top_k=20,min_p=0.,presence_penalty=0.,frequency_penalty=0.,
                repetition_penalty=1.,sampling_seed='42 + train index',dtype='bfloat16',quantization=None,
                tensor_parallel_size=TP[name],max_num_seqs=32,request_batch_size=64,
                gpu_memory_utilization=.65,enforce_eager=False,enable_prefix_caching=False,
                attention_backend='FLASH_ATTN',filter='None; keep every question and generated response',
                gold_missing=sum(not row['gold'].strip() for row in prompts),
                smoke_indices=selected,smoke_selection='63 uniform random questions (seed 42) plus longest prompt',
                model_config_sha256=sha(model/'config.json'),tokenizer_config_sha256=sha(model/'tokenizer_config.json'))
            assert protocol['max_model_len']<=config.get('text_config',config)['max_position_embeddings']
            immutable(BASE/'inputs'/name/f'{task}.protocol.json',protocol)
    save(BASE/'prepared.json',dict(prepared=True,models=NAMES,counts=COUNTS,responses_per_model=sum(COUNTS.values()),
        total_responses=len(NAMES)*sum(COUNTS.values()),samples_per_question=1,sources=sources,time=time.time()))
    print(json.dumps(json.loads((BASE/'prepared.json').read_text()),indent=2),flush=True)


def paths(mode,name,task,rank):
    directory=BASE/mode/name/task
    return directory/f'rank-{rank:02d}.jsonl',directory/f'rank-{rank:02d}.complete.json'


def selected_records(name,task,mode,rank,world):
    protocol=json.loads((BASE/'inputs'/name/f'{task}.protocol.json').read_text())
    path=BASE/'inputs'/name/f'{task}.jsonl';assert sha(path)==protocol['prompts_sha256']
    records=read(path)
    if mode=='smoke':records=[records[i] for i in protocol['smoke_indices']]
    return records[rank::world],protocol


def worker(args):
    from vllm import LLM,SamplingParams
    import torch
    torch.set_num_threads(4)
    tasks={task:selected_records(args.model,task,args.mode,args.rank,args.world) for task in COUNTS}
    pending={}
    for task,(records,protocol) in tasks.items():
        path,marker=paths(args.mode,args.model,task,args.rank)
        if marker.exists():
            done=json.loads(marker.read_text());assert done['responses']==len(records) and done['output_sha256']==sha(path)
            assert done['protocol_sha256']==sha(BASE/'inputs'/args.model/f'{task}.protocol.json')
            continue
        existing=read(path)
        assert [row['index'] for row in existing]==[row['index'] for row in records[:len(existing)]]
        for saved,expected in zip(existing,records,strict=False):
            assert all(saved[key]==expected[key] for key in expected)
            assert saved['max_new_tokens']==protocol['max_new_tokens']
        pending[task]=(records,protocol,existing)
    if not pending:return
    started=time.monotonic()
    llm=LLM(model=str(ROOT/'data/models'/args.model),dtype='bfloat16',tensor_parallel_size=TP[args.model],
        max_model_len=max(p['max_model_len'] for _,p in tasks.values()),max_num_seqs=32,gpu_memory_utilization=.65,
        enforce_eager=False,enable_prefix_caching=False,seed=42,disable_log_stats=True,
        language_model_only=True,limit_mm_per_prompt={'image':0,'video':0},attention_config={'backend':'FLASH_ATTN'})
    loaded=time.monotonic()-started
    tokenizer=llm.get_tokenizer()
    warm=next(iter(pending.values()))[0][:4]
    warm_start=time.monotonic()
    llm.generate([dict(prompt_token_ids=row['prompt_token_ids']) for row in warm],
        SamplingParams(n=1,temperature=.7,top_p=.8,top_k=20,min_p=0.,max_tokens=32,seed=42),use_tqdm=False)
    warm_seconds=time.monotonic()-warm_start
    configuration=llm.llm_engine.vllm_config
    runtime=dict(vllm=importlib.metadata.version('vllm'),torch=torch.__version__,
        transformers=importlib.metadata.version('transformers'),dtype=str(configuration.model_config.dtype),
        quantization=configuration.model_config.quantization,tensor_parallel_size=TP[args.model],
        enforce_eager=configuration.model_config.enforce_eager,
        max_model_len=configuration.model_config.max_model_len,
        compilation_config=str(configuration.compilation_config),
        cudagraph_mode=str(configuration.compilation_config.cudagraph_mode),
        attention_config=str(configuration.attention_config),
        load_seconds=loaded,warmup_seconds=warm_seconds,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    assert runtime['dtype']=='torch.bfloat16' and runtime['quantization'] is None
    assert runtime['enforce_eager'] is False
    assert 'NONE' not in runtime['cudagraph_mode'],'CUDA graph acceleration is required'
    save(BASE/args.mode/args.model/f'runtime.rank-{args.rank:02d}.json',runtime)
    opening=tokenizer.convert_tokens_to_ids('<think>')
    for task,(records,protocol,existing) in pending.items():
        path,marker=paths(args.mode,args.model,task,args.rank)
        batches=read(path.with_suffix('.timings.jsonl'));generation_start=time.monotonic()
        for offset in range(len(existing),len(records),64):
            batch=records[offset:offset+64]
            parameters=[SamplingParams(n=1,temperature=protocol['temperature'],top_p=protocol['top_p'],
                top_k=protocol['top_k'],min_p=protocol['min_p'],max_tokens=protocol['max_new_tokens'],
                seed=row['sampling_seed'],repetition_penalty=1.,presence_penalty=0.,frequency_penalty=0.) for row in batch]
            batch_start=time.monotonic()
            outputs=llm.generate([dict(prompt_token_ids=row['prompt_token_ids']) for row in batch],parameters,use_tqdm=False)
            seconds=time.monotonic()-batch_start
            generated=0
            for row,result in zip(batch,outputs,strict=True):
                assert len(result.outputs)==1
                response=result.outputs[0];tokens=list(response.token_ids)
                assert 0<len(tokens)<=protocol['max_new_tokens'] and response.text.strip()
                assert opening not in tokens,'Non-thinking generation unexpectedly emitted <think>'
                if response.finish_reason=='length':assert len(tokens)==protocol['max_new_tokens']
                generated+=len(tokens)
                existing.append(dict(**row,response=response.text,model_response=response.text,model_reasoning='',
                    response_token_ids=tokens,tokens=len(tokens),max_new_tokens=protocol['max_new_tokens'],
                    finish_reason=response.finish_reason,stop_reason=response.stop_reason,
                    is_finished=response.finish_reason=='stop'))
            jsonl(path,existing)
            info=dict(task=task,model=args.model,rank=args.rank,responses=len(batch),seconds=seconds,
                generated_tokens=generated,output_tokens_per_second=generated/seconds)
            batches.append(info)
            jsonl(path.with_suffix('.timings.jsonl'),batches)
            save(path.with_suffix('.live.json'),dict(**info,completed=len(existing),total=len(records),
                generation_seconds=time.monotonic()-generation_start))
            print(json.dumps(info),flush=True)
        assert len(existing)==len(records)
        save(marker,dict(complete=True,responses=len(existing),output_sha256=sha(path),
            protocol_sha256=sha(BASE/'inputs'/args.model/f'{task}.protocol.json'),runtime=runtime,
            generation_seconds=sum(batch['seconds'] for batch in batches),batches=batches))


def score(row):
    from verl.utils.reward_score.math_verify import compute_score_dict
    if not row['gold'].strip():return dict(**row,score=None,acc=None,pred=None,gold_missing=True,scoring_timed_out=False)
    if os.environ.get('MATH_VERIFY_BOXED_ONLY','1') not in ('0','false','False'):
        from llama31_math_grpo import last_complete_boxed
        from safe_math_powers import compare_integer_expressions
        box=last_complete_boxed(row['response'])
        if box is not None:
            content=box[box.index('{')+1:-1].strip()
            matched=compare_integer_expressions(row['gold'],content)
            if matched is not None:
                return dict(**row,score=int(matched),acc=int(matched),pred=content,
                    gold_missing=False,scoring_timed_out=False,
                    verification_mode='exact_bounded_integer_expression')
    result=compute_score_dict(row['response'],row['gold'],timeout_score=-1)
    if result['score']<0:result=compute_score_dict(row['response'],row['gold'],timeout_score=-1)
    if result['score']<0:return dict(**row,score=None,acc=None,pred=None,gold_missing=False,scoring_timed_out=True)
    return dict(**row,**result,gold_missing=False,scoring_timed_out=False)


def merge_model(name,mode,world):
    results={}
    for task in COUNTS:
        rows=[];seconds=0.;loads=[];runtimes=[]
        for rank in range(world):
            path,marker=paths(mode,name,task,rank)
            done=json.loads(marker.read_text());assert done['complete'] and done['output_sha256']==sha(path)
            assert done['protocol_sha256']==sha(BASE/'inputs'/name/f'{task}.protocol.json')
            shard=read(path);expected,protocol=selected_records(name,task,mode,rank,world)
            assert [r['index'] for r in shard]==[r['index'] for r in expected]
            for row,prompt in zip(shard,expected,strict=True):
                assert all(row[key]==prompt[key] for key in prompt)
                assert row['tokens']==len(row['response_token_ids']) and 0<row['tokens']<=CAPS[task]
            rows+=shard;seconds=max(seconds,done['generation_seconds']);loads.append(done['runtime']['load_seconds'])
            runtimes.append(done['runtime'])
        rows.sort(key=lambda row:row['index'])
        assert len(rows)==(SMOKE if mode=='smoke' else COUNTS[task]) and len({r['index'] for r in rows})==len(rows)
        with ProcessPoolExecutor(max_workers=8) as pool:scored=list(pool.map(score,rows,chunksize=8))
        directory=BASE/mode/name/task
        jsonl(directory/'responses.jsonl',rows);jsonl(directory/'scored.jsonl',scored)
        lengths=sorted(row['tokens'] for row in rows)
        judged=[row for row in scored if row['acc'] is not None]
        summary=dict(model=name,task=task,mode=mode,passed=True,responses=len(rows),unique_questions=len(rows),
            expected_questions=SMOKE if mode=='smoke' else COUNTS[task],max_new_tokens=CAPS[task],
            clipped=sum(row['finish_reason']=='length' for row in rows),
            clipping_rate=sum(row['finish_reason']=='length' for row in rows)/len(rows),
            generated_tokens=sum(lengths),mean_tokens=sum(lengths)/len(rows),
            median_tokens=(lengths[(len(lengths)-1)//2]+lengths[len(lengths)//2])/2,
            maximum_tokens=max(lengths),generation_seconds=seconds,load_seconds=max(loads),
            responses_per_second=len(rows)/seconds,output_tokens_per_second=sum(lengths)/seconds,
            accuracy=sum(row['acc'] for row in judged)/len(judged) if judged else None,
            correct=sum(row['acc'] for row in judged),gold_missing=sum(row['gold_missing'] for row in scored),
            scoring_timeouts=sum(row['scoring_timed_out'] for row in scored),
            world_size=world,tensor_parallel_size=TP[name],dtype='bfloat16',enable_thinking=False,
            responses_sha256=sha(directory/'responses.jsonl'),protocol=protocol,runtimes=runtimes,
            actual_max_model_len=runtimes[0]['max_model_len'],
            engine_context_policy='One shared engine for both tasks; context is the maximum of per-task requirements')
        if mode=='smoke':assert summary['scoring_timeouts']==0
        save(directory/'summary.json',summary);results[task]=summary
        if mode=='full':
            from datasets import Dataset
            target=directory/'dataset_finished'
            if not target.exists():
                Dataset.from_dict({key:[row[key] for row in rows] for key in
                    ['id','index','task','question','input_text','gold','model_response','model_reasoning','is_finished']}).save_to_disk(str(target))
    return results


def environment(gpus):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),
        PYTHONPATH=f'{ROOT}/recipe/llm_communicate:{ROOT}/scripts:{ROOT}/third_party/verl',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',
        TRITON_CACHE_DIR=str(ROOT/'.runtime/qwen35-generation-triton'),
        TORCHINDUCTOR_CACHE_DIR=str(ROOT/'.runtime/qwen35-generation-inductor'),
        XDG_CACHE_HOME=str(ROOT/'.runtime/qwen35-generation-cache'),
        MATH_VERIFY_BOXED_ONLY='1',MATH_VERIFY_TIMEOUT_S='10')


def launch(name,mode,rank,world,gpus):
    directory=BASE/mode/name;directory.mkdir(parents=True,exist_ok=True)
    log=(directory/f'rank-{rank:02d}.log').open('a')
    command=[str(INFER),'-u',str(Path(__file__).resolve()),'worker','--model',name,'--mode',mode,
             '--rank',str(rank),'--world',str(world)]
    job=subprocess.Popen(command,cwd=ROOT,env=environment(gpus),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return job,log


def stop_job(job):
    try:os.killpg(job.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:job.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:os.killpg(job.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        job.wait()


def wait_jobs(jobs):
    try:
        while any(job.poll() is None for job,_ in jobs):
            failures=[job.returncode for job,_ in jobs if job.poll() not in (None,0)]
            if failures:raise RuntimeError(f'Generation worker failed: {failures}')
            time.sleep(2)
        assert all(job.returncode==0 for job,_ in jobs)
    finally:
        for job,log in jobs:
            if job.poll() is None or job.returncode!=0:stop_job(job)
            job.wait();log.close()


def smoke_suite():
    started=time.time()
    save(BASE/'smoke.phase.json',dict(phase='generating',started=started))
    allocations={NAMES[0]:[0],NAMES[1]:[1],NAMES[2]:[2],NAMES[3]:[4,5,6,7]}
    jobs=[]
    try:
        for name,gpus in allocations.items():jobs.append(launch(name,'smoke',0,1,gpus))
        wait_jobs(jobs)
    finally:
        for job,log in jobs:
            if job.poll() is None:stop_job(job)
            if not log.closed:log.close()
    save(BASE/'smoke.phase.json',dict(phase='scoring',started=started))
    summaries={name:merge_model(name,'smoke',1) for name in NAMES}
    estimates={}
    for name,tasks in summaries.items():
        replicas=8//TP[name]
        times={task:COUNTS[task]/(summary['responses_per_second']*replicas) for task,summary in tasks.items()}
        cold=max(summary['load_seconds'] for summary in tasks.values())
        estimates[name]=dict(replicas=replicas,total_gpus=8,tensor_parallel_size=TP[name],
            task_seconds=times,load_seconds=cold,central_seconds=sum(times.values())+cold,
            lower_seconds=.8*sum(times.values())+cold,upper_seconds=1.8*sum(times.values())+cold+120)
    estimate=dict(complete=True,basis='64 real train questions per task; measured production batch size 64, max_num_seqs 32; cold load excluded from throughput',
        schedule='All 8 GPUs per model, models sequential; BF16, TP=1 for 9B/27B/35B, TP=4 for 122B',
        total_responses=59892,models=estimates,central_seconds=sum(v['central_seconds'] for v in estimates.values()),
        lower_seconds=sum(v['lower_seconds'] for v in estimates.values()),
        upper_seconds=sum(v['upper_seconds'] for v in estimates.values()),
        uncertainty='Small sample and batch tail; eight-replica scaling estimated rather than directly measured')
    save(BASE/'estimate.json',estimate)
    save(BASE/'smoke.complete.json',dict(complete=True,passed=8,total=8,responses=8*SMOKE,
        finished=time.time(),wall_seconds=time.time()-started,summaries=summaries,estimate=estimate))
    save(BASE/'smoke.phase.json',dict(phase='complete',finished=time.time()))
    print(json.dumps(estimate,indent=2),flush=True)


def full_suite():
    assert json.loads((BASE/'smoke.complete.json').read_text())['passed']==8
    for name in NAMES:
        replicas=8//TP[name]
        save(BASE/'full.phase.json',dict(phase='generating',model=name,replicas=replicas,started=time.time()))
        jobs=[]
        try:
            for rank in range(replicas):
                jobs.append(launch(name,'full',rank,replicas,list(range(rank*TP[name],(rank+1)*TP[name]))))
            wait_jobs(jobs)
        finally:
            for job,log in jobs:
                if job.poll() is None:stop_job(job)
                if not log.closed:log.close()
        save(BASE/'full.phase.json',dict(phase='scoring',model=name,started=time.time()))
        summaries=merge_model(name,'full',replicas)
        save(BASE/'full'/name/'complete.json',dict(complete=True,finished=time.time(),summaries=summaries))
    save(BASE/'full.complete.json',dict(complete=True,models=NAMES,responses=59892,finished=time.time()))
    save(BASE/'full.phase.json',dict(phase='complete',finished=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','smoke','full','worker','status'])
    parser.add_argument('--model',choices=NAMES)
    parser.add_argument('--mode',choices=['smoke','full'],default='smoke')
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world',type=int,default=1)
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args();BASE.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'third_party/verl'))
    os.environ['MATH_VERIFY_BOXED_ONLY']='1';os.environ['MATH_VERIFY_TIMEOUT_S']='10'
    if args.stage=='prepare':prepare();return
    if args.stage=='worker':
        assert args.model and 0<=args.rank<args.world
        worker(args);return
    if args.stage=='status':
        output={}
        for mode in ['smoke','full']:
            phase=BASE/f'{mode}.phase.json'
            output[mode]=dict(phase=json.loads(phase.read_text()) if phase.exists() else None)
            output[mode]['tasks']={}
            for name in NAMES:
                for task in COUNTS:
                    directory=BASE/mode/name/task;summary=directory/'summary.json'
                    live=[json.loads(path.read_text()) for path in directory.glob('*.live.json')]
                    output[mode]['tasks'][f'{name}/{task}']=dict(completed=sum(row['completed'] for row in live),
                        expected=SMOKE if mode=='smoke' else COUNTS[task],
                        summary={k:v for k,v in json.loads(summary.read_text()).items() if k!='protocol'} if summary.exists() else None)
        estimate=BASE/'estimate.json'
        if estimate.exists():output['estimate']=json.loads(estimate.read_text())
        print(json.dumps(output,indent=2),flush=True);return
    assert (BASE/'prepared.json').exists(),'Run prepare first'
    if args.detach:
        controller=BASE/f'{args.stage}.controller.json'
        if controller.exists():
            try:os.kill(json.loads(controller.read_text())['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('Controller is already running')
        snapshot=BASE/args.stage/'source_snapshot';snapshot.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(__file__).resolve(),snapshot/Path(__file__).name)
        save(BASE/f'{args.stage}.implementation.json',dict(script_sha256=sha(Path(__file__).resolve())))
        with (BASE/f'{args.stage}.pipeline.log').open('a') as log:
            proc=subprocess.Popen([str(CPU),'-u',str(Path(__file__).resolve()),args.stage],cwd=ROOT,
                env=environment([]),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(controller,dict(pid=proc.pid,started=time.time()));print(json.dumps(dict(pid=proc.pid,stage=args.stage)),flush=True);return
    lock=(BASE/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    implementation=BASE/f'{args.stage}.implementation.json'
    if implementation.exists():assert json.loads(implementation.read_text())['script_sha256']==sha(Path(__file__).resolve())
    try:
        if args.stage=='smoke':smoke_suite()
        else:full_suite()
    except Exception as error:
        save(BASE/f'{args.stage}.phase.json',dict(phase='failed',error=str(error),time=time.time()));raise


if __name__=='__main__':main()
