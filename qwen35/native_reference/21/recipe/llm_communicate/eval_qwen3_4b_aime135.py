"""Native Qwen3-4B AIME135 avg@4, optional thinking; same vLLM/VERL protocol as MATH evals."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
PYTHON=ROOT/'.runtime/nemotron-smoke/bin/python'
DATA=ROOT/'data/aime_2022_2026_135'
MODEL=ROOT/'data/models/Qwen3-4B'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def select_pilot(prompts,responses):
    if responses is None:return prompts
    groups=defaultdict(list)
    for row in prompts:groups[row['year']].append(row)
    rng=random.Random(42)
    for year in sorted(groups):rng.shuffle(groups[year])
    ordered=[groups[year][offset] for offset in range(max(map(len,groups.values())))
             for year in sorted(groups) if offset<len(groups[year])]
    return sorted(ordered[:responses//4],key=lambda row:row['index'])


def clipping_counts(rows,cap):
    count=len(rows)
    clipped=sum(row['finish_reason']=='length' for row in rows)
    hits=sum(row['tokens']==cap for row in rows)
    return dict(clipped=clipped,clipping_rate=clipped/count if count else None,
        cap_hits=hits,cap_hit_rate=hits/count if count else None,
        mean_tokens=sum(row['tokens'] for row in rows)/count if count else None,
        maximum_tokens=max((row['tokens'] for row in rows),default=None))


def thinking_fields(tokens,tokenizer):
    closing=tokenizer.convert_tokens_to_ids('</think>')
    ends=[i for i,token in enumerate(tokens) if token==closing]
    boundary=ends[-1]+1 if ends else len(tokens)
    return dict(thinking_completed=bool(ends),thinking_tokens=boundary,final_tokens=len(tokens)-boundary,
        reasoning_content=tokenizer.decode(tokens[:boundary],skip_special_tokens=True),
        final_response=tokenizer.decode(tokens[boundary:],skip_special_tokens=True))


def prepare(args):
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    manifest=json.loads((DATA/'complete.json').read_text())
    assert manifest['complete'] and manifest['questions']==135
    for name,digest in manifest['sha256'].items():assert sha(DATA/name)==digest
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    data=pq.read_table(DATA/'test.parquet').to_pylist()
    assert len(data)==135 and [row['extra_info']['index'] for row in data]==list(range(135))
    prompts=[]
    for row in data:
        ids=tokenizer.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True,
                                         enable_thinking=args.thinking,return_dict=False)
        assert 1<len(ids)<=2048
        rendered=tokenizer.decode(ids)
        if args.thinking:
            assert rendered.endswith('<|im_start|>assistant\n')
        else:assert rendered.endswith('<think>\n\n</think>\n\n')
        prompts.append(dict(**row['extra_info'],gold=row['reward_model']['ground_truth'],prompt_token_ids=ids))
    prompts=select_pilot(prompts,args.pilot_responses)
    config=json.loads((MODEL/'config.json').read_text())
    maximum=args.max_new_tokens+2048
    assert maximum<=config['max_position_embeddings'],'Requested context exceeds local model configuration'
    model_files=['config.json','generation_config.json','tokenizer_config.json','tokenizer.json',
                 'model.safetensors.index.json']
    model_hashes={name:sha(MODEL/name) for name in model_files if (MODEL/name).exists()}
    protocol=dict(model='Qwen/Qwen3-4B',model_path=str(MODEL),model_file_sha256=model_hashes,
        dataset=str(DATA/'test.parquet'),dataset_sha256=sha(DATA/'test.parquet'),
        coverage=manifest['coverage'],questions=len(prompts),responses=4*len(prompts),samples_per_question=4,
        enable_thinking=args.thinking,max_new_tokens=args.max_new_tokens,max_model_len=maximum,
        maximum_prompt_tokens=max(len(row['prompt_token_ids']) for row in prompts),
        temperature=.6 if args.thinking else .7,top_p=.95 if args.thinking else .8,
        top_k=20,min_p=0.,repetition_penalty=1.,presence_penalty=0.,frequency_penalty=0.,
        seed='1234 + 4*index + sample_index',sampling='four independent requests, vLLM n=1',
        world_size=args.world,dtype='bfloat16',max_num_seqs=16,request_batch_size=args.request_batch_size,
        backend='vLLM',vllm_version=importlib.metadata.version('vllm'),attention_backend='FLASH_ATTN',
        scorer='verl.utils.reward_score.math_verify.compute_score_dict',boxed_only=True,
        math_verify_version=importlib.metadata.version('math-verify'),score_timeout_seconds=10,
        metric=f'avg@4 = correct responses / {4*len(prompts)}; pass@4 reported separately',
        response_budget_source=('Default: 16384 if no non-thinking-specific recommendation; CLI override supported'
                                if args.max_new_tokens==16384 else 'Explicit CLI output budget'),
        official_guidance=dict(url='https://huggingface.co/Qwen/Qwen3-4B#best-practices',
            nonthinking_specific_length=None,general_output_tokens=32768,competition_output_tokens=38912,
            interpretation='General output-length advice; no separate non-thinking length given'))
    if args.thinking:
        protocol['response_budget_source']=('Official competition-problem output budget: 38912'
            if args.max_new_tokens==38912 else 'Explicit CLI output budget for thinking mode')
        protocol['output_budget_scope']='All generated tokens: thinking plus final response'
        protocol['scoring_scope']='Final response after last </think>; unfinished thinking has no final response'
    if args.pilot_responses is not None:
        protocol['pilot']=dict(responses=args.pilot_responses,selection_seed=42,
            selection='Shuffle within each year, then round-robin years; keep original question indices and sampling seeds',
            question_indices=[row['index'] for row in prompts],
            selected_coverage={str(year):sum(row['year']==year for row in prompts)
                               for year in sorted({row['year'] for row in prompts})})
    if (args.root/'protocol.json').exists():assert json.loads((args.root/'protocol.json').read_text())==protocol
    else:write(args.root/'protocol.json',protocol)
    if (args.root/'prompts.json').exists():assert json.loads((args.root/'prompts.json').read_text())==prompts
    else:write(args.root/'prompts.json',prompts)
    return protocol


def worker(args):
    import torch
    from vllm import LLM,SamplingParams
    torch.set_num_threads(4)
    protocol=json.loads((args.root/'protocol.json').read_text())
    prompts=json.loads((args.root/'prompts.json').read_text())
    records=[]
    for row in prompts[args.rank::args.world]:
        for sample in range(4):records.append(dict(**row,sample_index=sample,
            sampling_seed=1234+4*row['index']+sample))
    directory=args.root/'generation';directory.mkdir(exist_ok=True)
    path=directory/f'rank-{args.rank:02d}.jsonl'
    marker=directory/f'rank-{args.rank:02d}.complete.json'
    identity=dict(protocol_sha256=sha(args.root/'protocol.json'),prompts_sha256=sha(args.root/'prompts.json'),
                  rank=args.rank,world=args.world)
    if marker.exists():
        complete=json.loads(marker.read_text())
        assert complete['identity']==identity and complete['output_sha256']==sha(path)
        return
    existing=read(path)
    assert [(row['index'],row['sample_index']) for row in existing]==[
        (row['index'],row['sample_index']) for row in records[:len(existing)]]
    for saved,expected in zip(existing,records,strict=False):
        assert all(saved[key]==expected[key] for key in expected)
        assert saved['max_new_tokens']==args.max_new_tokens
    llm=None;started=time.monotonic()
    if len(existing)<len(records):
        llm=LLM(model=str(MODEL),dtype='bfloat16',tensor_parallel_size=1,
            max_model_len=protocol['max_model_len'],max_num_seqs=16,gpu_memory_utilization=.65,
            enforce_eager=True,enable_prefix_caching=False,enable_chunked_prefill=False,
            disable_hybrid_kv_cache_manager=True,disable_log_stats=True,seed=1234)
    for offset in range(len(existing),len(records),protocol['request_batch_size']):
        batch=records[offset:offset+protocol['request_batch_size']]
        params=[SamplingParams(n=1,temperature=protocol['temperature'],top_p=protocol['top_p'],
            top_k=protocol['top_k'],min_p=protocol['min_p'],
            max_tokens=args.max_new_tokens,seed=row['sampling_seed'],repetition_penalty=1.,
            presence_penalty=0.,frequency_penalty=0.) for row in batch]
        outputs=llm.generate([dict(prompt_token_ids=row['prompt_token_ids']) for row in batch],params,use_tqdm=False)
        for row,result in zip(batch,outputs,strict=True):
            assert len(result.outputs)==1
            response=result.outputs[0];tokens=list(response.token_ids)
            assert 0<len(tokens)<=args.max_new_tokens
            fields=thinking_fields(tokens,llm.get_tokenizer()) if protocol['enable_thinking'] else {}
            existing.append(dict(**row,response=response.text,response_token_ids=tokens,tokens=len(tokens),
                max_new_tokens=args.max_new_tokens,finish_reason=response.finish_reason,stop_reason=response.stop_reason,
                **fields))
        temp=path.with_suffix('.tmp')
        temp.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in existing));temp.replace(path)
        live=dict(rank=args.rank,responses=len(existing),total=len(records),seconds=time.monotonic()-started,
                  **clipping_counts(existing,args.max_new_tokens))
        write(directory/f'rank-{args.rank:02d}.live.json',live)
        print(json.dumps(live),flush=True)
    write(marker,dict(complete=True,identity=identity,output_sha256=sha(path),responses=len(records),seconds=time.monotonic()-started))


def score(row):
    # Exact built-in scorer used by our prior MATH and GSM bridge evaluations.
    from verl.utils.reward_score.math_verify import compute_score_dict
    response=row.get('final_response',row['response'])
    result=compute_score_dict(response,row['gold'],timeout_score=-1)
    if result['score']<0:result=compute_score_dict(response,row['gold'],timeout_score=-1)
    if result['score']<0:raise RuntimeError(f"Unresolved scoring timeout: {row['index']}/{row['sample_index']}")
    return dict(row,**result)


def self_test():
    cases=[(r'\boxed{33}','33',1),(r'\boxed{033}','33',1),
           (r'\boxed{34}','33',0),('Final answer: 33','33',0)]
    for response,gold,expected in cases:
        assert score(dict(index=0,sample_index=0,response=response,gold=gold))['acc']==expected
    rows=[dict(finish_reason='length',tokens=16),dict(finish_reason='stop',tokens=16),
          dict(finish_reason='stop',tokens=4)]
    counts=clipping_counts(rows,16)
    assert counts['clipped']==1 and counts['cap_hits']==2
    assert counts['clipping_rate']==1/3 and counts['mean_tokens']==12
    assert clipping_counts([],16)['clipping_rate'] is None
    fixture=[dict(index=year*30+i,year=2022+year) for year in range(5) for i in range(30)]
    pilot=select_pilot(fixture,20)
    assert len(pilot)==5 and len({row['year'] for row in pilot})==5
    assert pilot==select_pilot(fixture,20) and select_pilot(fixture,None) is fixture
    class TokenizerFixture:
        def convert_tokens_to_ids(self,token):assert token=='</think>';return 99
        def decode(self,tokens,skip_special_tokens):return '|'.join(map(str,tokens))
    parsed=thinking_fields([1,2,99,3,4],TokenizerFixture())
    assert parsed['thinking_completed'] and parsed['thinking_tokens']==3 and parsed['final_response']=='3|4'
    unfinished=thinking_fields([1,2],TokenizerFixture())
    assert not unfinished['thinking_completed'] and unfinished['final_response']==''
    assert score(dict(index=0,sample_index=0,response=r'\boxed{33}',final_response='',gold='33'))['acc']==0
    assert score(dict(index=0,sample_index=0,response=r'\boxed{34}',final_response=r'\boxed{33}',gold='33'))['acc']==1
    print('VERL scoring checks passed: boxed correctness, leading zeros, incorrect and unboxed answers',flush=True)


def merge(args):
    protocol=json.loads((args.root/'protocol.json').read_text())
    prompts=json.loads((args.root/'prompts.json').read_text())
    by_index={row['index']:row for row in prompts}
    rows=[]
    for rank in range(args.world):
        marker=json.loads((args.root/f'generation/rank-{rank:02d}.complete.json').read_text())
        path=args.root/f'generation/rank-{rank:02d}.jsonl'
        assert marker['complete'] and marker['output_sha256']==sha(path)
        assert marker['identity']==dict(protocol_sha256=sha(args.root/'protocol.json'),
            prompts_sha256=sha(args.root/'prompts.json'),rank=rank,world=args.world)
        rows.extend(read(path))
    rows.sort(key=lambda row:(row['index'],row['sample_index']))
    assert [(row['index'],row['sample_index']) for row in rows]==[(i,j) for i in sorted(by_index) for j in range(4)]
    for row in rows:
        assert row['sampling_seed']==1234+4*row['index']+row['sample_index']
        assert row['gold']==by_index[row['index']]['gold'] and row['prompt_token_ids']==by_index[row['index']]['prompt_token_ids']
        assert row['max_new_tokens']==args.max_new_tokens and row['tokens']==len(row['response_token_ids'])
    with ProcessPoolExecutor(max_workers=min(8,args.world)) as pool:rows=list(pool.map(score,rows,chunksize=8))
    def summarize(records):
        groups=defaultdict(list)
        for row in records:groups[row['index']].append(row['acc'])
        assert all(len(values)==4 for values in groups.values())
        correct=sum(row['acc'] for row in records);count=len(records);questions=len(groups)
        lengths=sorted(row['tokens'] for row in records)
        def percentile(fraction):
            position=(len(lengths)-1)*fraction;lo=int(position);hi=min(lo+1,len(lengths)-1)
            return lengths[lo]+(lengths[hi]-lengths[lo])*(position-lo)
        return dict(questions=questions,responses=count,correct=correct,avg_at_4=correct/count,
            pass_at_4=sum(any(values) for values in groups.values())/questions,
            clipped=sum(row['finish_reason']=='length' for row in records),
            clipping_rate=sum(row['finish_reason']=='length' for row in records)/count,
            cap_hit_rate=sum(row['tokens']==args.max_new_tokens for row in records)/count,
            mean_tokens=sum(lengths)/count,median_tokens=percentile(.5),p90_tokens=percentile(.9),
            minimum_tokens=min(lengths),maximum_tokens=max(lengths),scoring_timeouts=0)
    output=args.root/'scored.jsonl'
    output.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    per_question=[dict(index=i,id=by_index[i]['id'],year=by_index[i]['year'],exam=by_index[i]['exam'],
        correct=sum(row['acc'] for row in rows if row['index']==i),samples=4,
        accuracy=sum(row['acc'] for row in rows if row['index']==i)/4) for i in sorted(by_index)]
    write(args.root/'per_question.json',per_question)
    summary=dict(**summarize(rows),per_year={str(year):summarize([row for row in rows if row['year']==year])
        for year in sorted({row['year'] for row in prompts})},max_new_tokens=args.max_new_tokens,protocol=protocol,
        scored_sha256=sha(output))
    if protocol['enable_thinking']:
        summary['thinking_completed']=sum(row['thinking_completed'] for row in rows)
        summary['mean_thinking_tokens']=sum(row['thinking_tokens'] for row in rows)/len(rows)
        summary['mean_final_tokens']=sum(row['final_tokens'] for row in rows)/len(rows)
    write(args.root/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def command(args,*extra):
    return [str(PYTHON),'-u',str(Path(__file__).resolve()),'--root',str(args.root),
        '--world',str(args.world),'--max-new-tokens',str(args.max_new_tokens),
        '--request-batch-size',str(args.request_batch_size),
        *(['--thinking'] if args.thinking else []),
        *(['--pilot-responses',str(args.pilot_responses)] if args.pilot_responses is not None else []),*extra]


def run(args):
    lock=(args.root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=prepare(args)
    if args.wait_for:
        write(args.root/'phase.json',dict(phase='waiting_for_prior_suite',waiting_for=str(args.wait_for),started=time.time()))
        while not (args.wait_for/'complete.json').exists():
            phase=args.wait_for/'phase.json'
            if phase.exists() and json.loads(phase.read_text())['phase']=='failed':
                raise RuntimeError('Preceding suite failed; AIME evaluation not started')
            time.sleep(10)
        assert json.loads((args.wait_for/'complete.json').read_text())['complete']
    implementation=args.root/'implementation_sha256.json'
    if implementation.exists():
        for name,digest in json.loads(implementation.read_text()).items():
            assert sha(ROOT/name)==digest,'Evaluation implementation changed while queued'
    write(args.root/'phase.json',dict(phase='generating',started=time.time()))
    logs=args.root/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONPATH=f'{ROOT}/recipe/llm_communicate:{ROOT}/scripts:{ROOT}/third_party/verl',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
        VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',
        VLLM_ATTENTION_BACKEND='FLASH_ATTN',VLLM_USE_FLASHINFER_SAMPLER='0',
        MATH_VERIFY_BOXED_ONLY='1',MATH_VERIFY_TIMEOUT_S='10')
    jobs=[]
    try:
        for rank in range(args.world):
            log=(logs/f'rank-{rank:02d}.log').open('a')
            job=subprocess.Popen(command(args,'--worker','--rank',str(rank)),cwd=ROOT,
                env=dict(env,CUDA_VISIBLE_DEVICES=str(rank)),stdout=log,stderr=subprocess.STDOUT)
            jobs.append((job,log))
        while any(job.poll() is None for job,_ in jobs):
            if any(job.poll() not in (None,0) for job,_ in jobs):raise RuntimeError('AIME generation worker failed')
            time.sleep(2)
        assert all(job.returncode==0 for job,_ in jobs)
    finally:
        for job,log in jobs:
            if job.poll() is None:job.terminate()
            job.wait();log.close()
    write(args.root/'phase.json',dict(phase='scoring',started=time.time()))
    subprocess.run(command(args,'--merge'),cwd=ROOT,env=env,check=True)
    summary=json.loads((args.root/'summary.json').read_text())
    assert summary['questions']==protocol['questions'] and summary['responses']==protocol['responses'] and summary['scoring_timeouts']==0
    write(args.root/'complete.json',dict(complete=True,questions=protocol['questions'],responses=protocol['responses'],
        avg_at_4=summary['avg_at_4'],finished=time.time()))
    write(args.root/'phase.json',dict(phase='complete',time=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--max-new-tokens',type=int,help='Default: non-thinking 16384, thinking competition budget 38912')
    parser.add_argument('--thinking',action='store_true')
    parser.add_argument('--world',type=int,default=8)
    parser.add_argument('--request-batch-size',type=int,default=16)
    parser.add_argument('--pilot-responses',type=int,help='Pilot only: multiple of 4, stratified across years (seed 42)')
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--merge',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--wait-for',type=Path)
    args=parser.parse_args()
    if args.max_new_tokens is None:args.max_new_tokens=38912 if args.thinking else 16384
    if not 1<=args.world<=8 or min(args.max_new_tokens,args.request_batch_size)<=0:
        parser.error('Positive output cap/batch size and world in [1,8] required')
    if args.pilot_responses is not None and (not 4<=args.pilot_responses<=540 or args.pilot_responses%4):
        parser.error('Pilot response count must be a multiple of 4 in [4,540]')
    mode='thinking' if args.thinking else 'nonthinking'
    name=(f'qwen3_4b_aime135_pilot{args.pilot_responses}_{mode}_cap{args.max_new_tokens}'
          if args.pilot_responses is not None else f'qwen3_4b_aime135_{mode}_avg4_cap{args.max_new_tokens}')
    args.root=(args.root or ROOT/'outputs/llm_communicate'/name).absolute()
    args.root.mkdir(parents=True,exist_ok=True)
    if args.wait_for:args.wait_for=args.wait_for.absolute()
    sys.path.insert(0,str(ROOT/'third_party/verl'))
    os.environ['MATH_VERIFY_BOXED_ONLY']='1';os.environ['MATH_VERIFY_TIMEOUT_S']='10'
    if args.status:
        phase=args.root/'phase.json';summary=args.root/'summary.json'
        saved=[]
        for path in (args.root/'generation').glob('rank-*.jsonl'):
            with path.open() as handle:
                for line in handle:
                    if not line.strip():continue
                    row=json.loads(line)
                    saved.append({key:row[key] for key in ['tokens','finish_reason','year','index']})
        print(json.dumps(dict(phase=json.loads(phase.read_text()) if phase.exists() else None,
            responses_saved=len(saved),total_responses=(json.loads((args.root/'protocol.json').read_text())['responses']
                if (args.root/'protocol.json').exists() else args.pilot_responses or 540),max_new_tokens=args.max_new_tokens,
            clipping_so_far=clipping_counts(saved,args.max_new_tokens),
            clipping_by_year={str(year):clipping_counts([row for row in saved if row['year']==year],args.max_new_tokens)
                              for year in sorted({row['year'] for row in saved})},
            summary=json.loads(summary.read_text()) if summary.exists() else None),indent=2));return
    if args.self_test:self_test();return
    if args.prepare:print(json.dumps(prepare(args),indent=2));return
    if args.worker:worker(args);return
    if args.merge:merge(args);return
    if args.detach:
        prepare(args)
        pid_path=args.root/'controller.json'
        if pid_path.exists():
            previous=json.loads(pid_path.read_text())
            try:os.kill(previous['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('AIME controller already running')
        snapshot=args.root/'source_snapshot';snapshot.mkdir(exist_ok=True)
        sources=[Path(__file__).resolve(),ROOT/'recipe/llm_communicate/prepare_aime_2022_2026_135.py',
                 ROOT/'third_party/verl/verl/utils/reward_score/math_verify.py',
                 ROOT/'third_party/verl/verl/utils/reward_score/math_reward.py']
        for path in sources:shutil.copyfile(path,snapshot/path.name)
        write(args.root/'implementation_sha256.json',{str(path.relative_to(ROOT)):sha(path) for path in sources})
        cmd=command(args)
        if args.wait_for:cmd+=['--wait-for',str(args.wait_for)]
        with (args.root/'pipeline.log').open('a') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(pid=proc.pid,root=str(args.root),started=time.time())
        write(pid_path,record);print(json.dumps(record),flush=True);return
    try:run(args)
    except Exception as error:
        write(args.root/'phase.json',dict(phase='failed',error=str(error),time=time.time()));raise


if __name__=='__main__':main()
