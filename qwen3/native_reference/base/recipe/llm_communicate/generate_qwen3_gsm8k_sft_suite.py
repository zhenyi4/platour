"""Fresh GSM8K training trajectories for native Qwen3 1.7B/4B/8B/235B-FP8."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal,InvalidOperation
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from generate_qwen35_train_suite import ROOT,SUFFIX,save,sha,read,jsonl,stop_job
from eval_qwen3_native_gsm8k_math500 import SAMPLING

BASE=ROOT/'outputs/llm_communicate/qwen3_17b_4b_8b_235b_gsm8k_train_sft_cap2048'
SCRIPT=Path(__file__).resolve()
DATA_REVISION='740312add88f781978c0658806c59bc2815b9866'
FOUR_B_REVISION='1cfa9a7208912126459214e8b04321603b3df60c'
ALLOCATIONS={'Qwen3-1.7B':[[0]],'Qwen3-4B':[[1]],'Qwen3-8B':[[2],[3]],'Qwen3-235B-A22B-FP8':[[0,1,2,3],[4,5,6,7]]}


def prepare(root):
    import torch
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download,snapshot_download
    from transformers import AutoTokenizer
    root.mkdir(parents=True,exist_ok=True)
    save(root/'phase.json',dict(phase='prepare',time=time.time()))
    snapshot_download('Qwen/Qwen3-4B',revision=FOUR_B_REVISION,local_dir=str(ROOT/'data/models/Qwen3-4B'),
        allow_patterns=['*.json','*.safetensors','*.jinja','*.txt'],max_workers=8)
    source=Path(hf_hub_download('openai/gsm8k',filename='main/train-00000-of-00001.parquet',
        repo_type='dataset',revision=DATA_REVISION,local_dir=str(ROOT/'data/gsm8k')))
    raw=pq.read_table(source).to_pylist();assert len(raw)==7473
    questions=[]
    for i,row in enumerate(raw):
        assert '####' in row['answer']
        question=row['question'].strip();gold=row['answer'].rsplit('####',1)[1].strip().replace(',','')
        assert question and gold
        questions.append(dict(index=i,id=f'gsm8k_train_{i:06d}',task='gsm8k',question=question,
            input_text=question+SUFFIX,gold=gold,gold_answer_text=row['answer'],sample_index=0,sampling_seed=1234+i))
    save(root/'questions.json',questions)
    source_info=dict(repo='openai/gsm8k',config='main',split='train',revision=DATA_REVISION,path=str(source),sha256=sha(source))
    save(ROOT/'data/gsm8k/source.json',source_info)
    order=torch.randperm(7473,generator=torch.Generator().manual_seed(42)).tolist()
    save(root/'split.json',dict(train=order[:7398],validation=order[7398:]))
    canonical_ids=None
    for name,groups in ALLOCATIONS.items():
        model=ROOT/'data/models'/name;tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
        config=json.loads((model/'config.json').read_text());directory=root/name
        old_protocol=directory/'protocol.json'
        if name=='Qwen3-235B-A22B-FP8' and old_protocol.exists():
            old=json.loads(old_protocol.read_text())
            if old['world_size']==1 and len(groups)==2 and (directory/'generation').exists():
                archive=directory/'generation_before_8gpu';assert not archive.exists()
                shutil.move(str(directory/'generation'),str(archive))
                previous=read(archive/'rank-00.jsonl')
                assert [r['index'] for r in previous]==list(range(len(previous)))
                for rank in range(2):jsonl(directory/'generation'/f'rank-{rank:02d}.jsonl',previous[rank::2])
                save(root/'allocation_migration.json',dict(old_protocol=old,new_gpu_groups=groups,
                    preserved_responses=len(previous),archive=str(archive),sampling_unchanged=True,
                    reason='All eight GPUs for 235B; TP8 unsupported by FP8 block dimensions'))
        prompts=[]
        for row in questions:
            ids=tokenizer.apply_chat_template([dict(role='user',content=row['input_text'])],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False)
            assert len(ids)<=2048 and tokenizer.decode(ids).endswith('<think>\n\n</think>\n\n')
            prompts.append(dict(row,prompt_token_ids=ids))
        actual_ids=[r['prompt_token_ids'] for r in prompts]
        if canonical_ids is None:canonical_ids=actual_ids
        else:assert actual_ids==canonical_ids,'Native prompt token IDs differ between models'
        save(directory/'prompts.json',prompts)
        index=model/'model.safetensors.index.json'
        shards=sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
        assert all((model/s).is_file() and (model/s).stat().st_size>0 for s in shards)
        p=dict(model='Qwen/'+name,model_path=str(model),task='gsm8k',split='train',source=source_info,
            questions=7473,responses=7473,samples_per_question=1,max_new_tokens=2048,max_model_len=6144,
            maximum_prompt_tokens=max(map(len,actual_ids)),enable_thinking=False,native_chat_template=True,
            prompt_instruction=SUFFIX,sampling=SAMPLING,seed='1234 + train index',request_n=1,
            gpu_groups=groups,world_size=len(groups),tensor_parallel_size=len(groups[0]),dtype='bfloat16',
            quantization=config.get('quantization_config',{}).get('quant_method'),
            checkpoint='Native model; no SFT, LoRA, or bridge',attention_backend='FLASH_ATTN',flash_attn_version=2,
            sample_filter='None; retain incorrect and clipped responses',prompts_sha256=sha(directory/'prompts.json'),
            model_config_sha256=sha(model/'config.json'),tokenizer_config_sha256=sha(model/'tokenizer_config.json'),
            shard_sizes={s:(model/s).stat().st_size for s in shards},shared_split_sha256=sha(root/'split.json'))
        save(directory/'protocol.json',p)
    files=[SCRIPT,ROOT/'recipe/llm_communicate/generate_qwen35_train_suite.py',ROOT/'recipe/llm_communicate/eval_qwen3_native_gsm8k_math500.py']
    hashes={str(f.relative_to(ROOT)):sha(f) for f in files};save(root/'source_sha256.json',hashes)
    for f in files:
        target=root/'source_snapshot'/f.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    save(root/'plan.json',dict(models=ALLOCATIONS,dataset=source_info,questions=7473,total_responses=29892,
        sampling=SAMPLING,max_new_tokens=2048,enable_thinking=False,prompt_ids_identical_between_models=True,
        automatically_start_sft=False,wandb_enabled=False))


def environment(gpus):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),
        PYTHONPATH=str(ROOT/'scripts')+':'+str(ROOT/'recipe/llm_communicate')+':'+str(ROOT/'third_party/verl'),
        OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',WANDB_DISABLED='true',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',HF_HUB_OFFLINE='1',
        TRITON_CACHE_DIR=str(ROOT/'.runtime/qwen35-generation-triton'),
        TORCHINDUCTOR_CACHE_DIR=str(ROOT/'.runtime/qwen35-generation-inductor'),
        XDG_CACHE_HOME=str(ROOT/'.runtime/qwen35-generation-cache'))


def worker(root,name,rank):
    import torch
    from vllm import LLM,SamplingParams
    torch.set_num_threads(4)
    directory=root/name;p=json.loads((directory/'protocol.json').read_text())
    assert sha(directory/'prompts.json')==p['prompts_sha256']
    for f,digest in json.loads((root/'source_sha256.json').read_text()).items():assert sha(ROOT/f)==digest
    llm=LLM(model=p['model_path'],dtype='bfloat16',tensor_parallel_size=p['tensor_parallel_size'],
        max_model_len=6144,max_num_seqs=64,gpu_memory_utilization=.85 if p['quantization'] else .65,
        moe_backend='triton' if p['quantization'] else 'auto',
        enable_prefix_caching=False,seed=1234,disable_log_stats=True,generation_config='vllm',
        attention_config={'backend':'FLASH_ATTN','flash_attn_version':2})
    config=llm.llm_engine.vllm_config
    assert config.model_config.quantization==p['quantization']
    path=directory/'generation'/f'rank-{rank:02d}.jsonl'
    rows=json.loads((directory/'prompts.json').read_text())[rank::p['world_size']];saved=read(path)
    for actual,expected in zip(saved,rows,strict=False):assert all(actual[k]==v for k,v in expected.items())
    probe=SamplingParams(n=1,max_tokens=32,seed=1234,**SAMPLING)
    assert all(getattr(probe,k)==v for k,v in SAMPLING.items())
    llm.generate([dict(prompt_token_ids=rows[0]['prompt_token_ids'])],probe,use_tqdm=False)
    runtime=dict(vllm=importlib.metadata.version('vllm'),transformers=importlib.metadata.version('transformers'),
        torch=torch.__version__,sampling=SAMPLING,quantization=config.model_config.quantization,
        tensor_parallel_size=p['tensor_parallel_size'],gpu_ids=os.environ['CUDA_VISIBLE_DEVICES'],
        moe_backend='triton' if p['quantization'] else 'auto',
        cudagraph_mode=str(config.compilation_config.cudagraph_mode))
    save(path.with_suffix('.runtime.json'),runtime);started=time.monotonic()
    for offset in range(len(saved),len(rows),64):
        group=rows[offset:offset+64]
        params=[SamplingParams(n=1,max_tokens=2048,seed=r['sampling_seed'],**SAMPLING) for r in group]
        outputs=llm.generate([dict(prompt_token_ids=r['prompt_token_ids']) for r in group],params,use_tqdm=False)
        for row,result in zip(group,outputs,strict=True):
            assert result.prompt_token_ids==row['prompt_token_ids'] and len(result.outputs)==1
            o=result.outputs[0];tokens=list(o.token_ids);assert 0<len(tokens)<=2048
            assert o.finish_reason!='length' or len(tokens)==2048
            saved.append(dict(row,response=o.text,response_token_ids=tokens,tokens=len(tokens),
                finish_reason=o.finish_reason,stop_reason=o.stop_reason,max_new_tokens=2048))
        jsonl(path,saved)
        live=dict(saved=len(saved),total=len(rows),seconds=time.monotonic()-started,
            generated_tokens=sum(r['tokens'] for r in saved),clipped=sum(r['finish_reason']=='length' for r in saved))
        save(path.with_suffix('.live.json'),live);print(json.dumps(live),flush=True)
    save(path.with_suffix('.complete.json'),dict(complete=True,responses=len(saved),output_sha256=sha(path),runtime=runtime))


def score(row):
    # GSM8K golds are numeric. Require the last balanced box; never use a prose fallback.
    matches=list(re.finditer(r'\\boxed\s*\{',row['response']));answer=None
    if matches:
        start=matches[-1].end();depth=1
        for end in range(start,len(row['response'])):
            depth+=(row['response'][end]=='{')-(row['response'][end]=='}')
            if depth==0:answer=row['response'][start:end].strip();break
    correct=0;timed_out=False
    if answer is not None:
        normalized=answer.replace(',','').replace('\\,','').replace('\\!','').replace('\\$','').strip().strip('$').strip()
        try:correct=int(Decimal(normalized)==Decimal(row['gold']))
        except InvalidOperation:
            def timeout(sig,frame):raise TimeoutError('math scoring timed out')
            previous=signal.signal(signal.SIGALRM,timeout);signal.alarm(5)
            try:
                from math_verify import parse,verify
                from math_verify.parser import LatexExtractionConfig
                correct=int(verify(parse('\\boxed{'+row['gold']+'}',extraction_config=[LatexExtractionConfig()]),
                    parse('\\boxed{'+answer+'}',extraction_config=[LatexExtractionConfig()])))
            except TimeoutError:timed_out=True
            except Exception:correct=0
            finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)
    return dict(row,acc=correct,score=correct,pred=answer,gold_missing=False,scoring_timed_out=timed_out)


def merge_export(root):
    exports={};summary={}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for name,groups in ALLOCATIONS.items():
            directory=root/name;rows=[]
            for rank in range(len(groups)):
                path=directory/'generation'/f'rank-{rank:02d}.jsonl'
                done=json.loads(path.with_suffix('.complete.json').read_text())
                assert done['complete'] and done['output_sha256']==sha(path);rows.extend(read(path))
            rows.sort(key=lambda r:r['index']);assert [r['index'] for r in rows]==list(range(7473))
            prompts=json.loads((directory/'prompts.json').read_text())
            assert all(all(r[k]==v for k,v in expected.items()) for r,expected in zip(rows,prompts,strict=True))
            jsonl(directory/'responses.jsonl',rows)
            scored=list(pool.map(score,rows,chunksize=16));jsonl(directory/'scored.jsonl',scored)
            records=[dict(id=r['id'],index=r['index'],task='gsm8k',split='train',model=name,enable_thinking=False,
                messages=[dict(role='user',content=r['input_text']),dict(role='assistant',content=r['response'])],
                prompt_token_ids=r['prompt_token_ids'],response_token_ids=r['response_token_ids'],
                gold=r['gold'],acc=r['acc'],gold_missing=False,tokens=r['tokens'],finish_reason=r['finish_reason'],
                clipped=r['finish_reason']=='length',sampling_seed=r['sampling_seed']) for r in scored]
            target=directory/'sft.jsonl';jsonl(target,records)
            info=dict(complete=True,model=name,responses=7473,accuracy=sum(r['acc'] for r in scored)/7473,
                correct=sum(r['acc'] for r in scored),mean_tokens=sum(r['tokens'] for r in scored)/7473,
                clipped=sum(r['finish_reason']=='length' for r in scored),scoring_timeouts=sum(r['scoring_timed_out'] for r in scored),
                responses_sha256=sha(directory/'responses.jsonl'),scored_sha256=sha(directory/'scored.jsonl'),sft_sha256=sha(target))
            save(directory/'summary.json',info);summary[name]=info
            exports[name]=dict(path=str(target),responses=7473,sha256=sha(target),filter='None; every training question retained')
    save(root/'summary.json',summary)
    save(root/'sft_exports.json',dict(complete=True,exports=dict(gsm8k=exports),responses=29892,
        split_path=str(root/'split.json'),split_sha256=sha(root/'split.json'),clipped_responses_add_synthetic_eos=False))


def run(root):
    children=[]
    def stop(sig,frame):raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        prepare(root)
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        assert all(int(x)<512 for x in used.splitlines()),'Eight GPUs must be idle'
        save(root/'phase.json',dict(phase='generating',time=time.time()))
        phases=[('finish_dense',[n for n in ALLOCATIONS if n!='Qwen3-235B-A22B-FP8']),
                ('generate_235b_8gpu',['Qwen3-235B-A22B-FP8'])]
        for phase,names in phases:
            save(root/'phase.json',dict(phase=phase,time=time.time()))
            pending=set()
            for name in names:
                for rank,gpus in enumerate(ALLOCATIONS[name]):
                    done=root/name/'generation'/f'rank-{rank:02d}.complete.json'
                    if done.exists() and json.loads(done.read_text())['complete']:continue
                    log=root/'logs'/f'{name}.rank-{rank:02d}.log';log.parent.mkdir(exist_ok=True)
                    with log.open('a') as f:
                        proc=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'worker','--root',str(root),'--model',name,'--rank',str(rank)],
                            cwd=ROOT,env=environment(gpus),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    children.append(proc);pending.add(proc)
            save(root/'workers.json',[dict(pid=p.pid) for p in children])
            while pending:
                for child in list(pending):
                    code=child.poll()
                    if code is None:continue
                    if code:raise RuntimeError(f'Generation worker {child.pid} failed ({code}); inspect logs')
                    pending.remove(child)
                if pending:time.sleep(2)
        save(root/'phase.json',dict(phase='scoring_and_export',time=time.time()))
        merge_export(root)
        save(root/'phase.json',dict(phase='complete',time=time.time()))
    except BaseException as error:
        save(root/'phase.json',dict(phase='failed',error=repr(error),time=time.time()));raise
    finally:
        for child in children:
            if child.poll() is None:stop_job(child)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['all','worker','export'])
    parser.add_argument('--root',type=Path,default=BASE);parser.add_argument('--model',choices=ALLOCATIONS);parser.add_argument('--rank',type=int,default=0)
    args=parser.parse_args()
    if args.stage=='worker':worker(args.root.absolute(),args.model,args.rank)
    elif args.stage=='export':merge_export(args.root.absolute())
    else:run(args.root.absolute())
