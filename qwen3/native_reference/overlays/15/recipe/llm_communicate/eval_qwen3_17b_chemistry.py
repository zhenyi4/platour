"""SDPO Chemistry test: answer-only greedy, nonthinking CoT@8, thinking CoT@8."""
import argparse
from collections import Counter
from functools import lru_cache
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from arc_bridge_sft_common import ROOT,HERE,save,sha,GREEDY
from generate_qwen35_train_suite import read,jsonl,stop_job
from run_qwen3_4b_17b_gsm8k_sft import environment
import eval_qwen3_arc_challenge as native

BASE=ROOT/'outputs/llm_communicate/qwen3_17b_chemistry_three_modes_avg8'
MODEL='Qwen3-1.7B'
SCRIPT=Path(__file__).resolve()
MODES={'answer_only':(False,False,1,64,GREEDY),
    'nonthinking_cot':(False,True,8,2048,dict(native.SAMPLING)),
    'thinking_cot':(True,True,8,8192,dict(native.SAMPLING,temperature=.6,top_p=.95))}

@lru_cache(None)
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ROOT/'data/models'/MODEL,local_files_only=True)

def score(row):
    text=row['response'].strip();pred=None;status='missing_answer'
    if row['mode']=='answer_only':
        if text in row['labels']:pred=text;status='valid'
        else:status='invalid_direct_answer'
    else:
        if row['mode']=='thinking_cot':
            ids=row['response_token_ids']
            if 151668 not in ids:
                return dict(row,extracted_answer=None,extraction_status='unclosed_thinking',acc=0)
            last=len(ids)-1-ids[::-1].index(151668)
            text=tokenizer().decode(ids[last+1:],skip_special_tokens=True)
        # Match SDPO's last <answer> extraction, including its incomplete-tag
        # behavior; never recover a letter from arbitrary reasoning prose.
        value=text.split('<answer>')[-1].split('</answer>')[0].strip()
        if value in row['labels']:pred=value;status='valid'
        elif '<answer>' in text:status='invalid_answer'
    return dict(row,extracted_answer=pred,extraction_status=status,acc=int(pred==row['gold']))

def prepare():
    BASE.mkdir(parents=True,exist_ok=True)
    assert not (BASE/'manifest.json').exists(),'Already prepared'
    source=Path('/tmp/sdpo-inspect');revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    destination=BASE/'data';destination.mkdir(exist_ok=True)
    for split in ['train','test']:shutil.copyfile(source/f'datasets/sciknoweval/chemistry/{split}.json',destination/f'{split}.json')
    rows=read(destination/'test.json');training=read(destination/'train.json')
    assert len(rows)==210 and len(training)==1890
    assert not {r['idx'] for r in rows}&{r['idx'] for r in training}
    assert not {r['prompt'] for r in rows}&{r['prompt'] for r in training}
    save(destination/'source.json',dict(repo='https://github.com/lasgroup/SDPO',revision=revision,
        train=1890,test=210,test_sha256=sha(destination/'test.json'),train_sha256=sha(destination/'train.json'),
        test_path='datasets/sciknoweval/chemistry/test.json',overlap=0))
    t=tokenizer();assert t.convert_tokens_to_ids('</think>')==151668
    for mode,(thinking,cot,n,cap,sampling) in MODES.items():
        root=BASE/mode;directory=root/MODEL;directory.mkdir(parents=True,exist_ok=True);prompts=[]
        for index,row in enumerate(rows):
            assert row['prompt'].endswith('\nPlease reason step by step.')
            question=row['prompt'].removesuffix('\nPlease reason step by step.')
            labels=re.findall(r'^([A-D]): ',question,re.MULTILINE);assert len(labels) in (2,4) and row['answer'] in labels
            messages=([dict(role='system',content=row['system']),dict(role='user',content=row['prompt'])] if cot else
                [dict(role='user',content=question+'\n\nReply with only the letter of the correct option. Do not include any explanation or any other text.')])
            ids=t.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=thinking,return_dict=False)
            rendered=t.decode(ids)
            if thinking:assert rendered.endswith(('<|im_start|>assistant\n','<think>\n'))
            else:assert rendered.endswith('<think>\n\n</think>\n\n')
            prompts.append(dict(index=index,id=str(row['idx']),mode=mode,question=question,labels=labels,
                gold=row['answer'],messages=messages,prompt_token_ids=ids))
        save(directory/'prompts.json',prompts);maximum=max(len(r['prompt_token_ids']) for r in prompts)
        p=dict(model='Qwen/'+MODEL,model_path=str(ROOT/'data/models'/MODEL),task='SciKnowEval Chemistry L3',
            mode=mode,split='SDPO test',questions=210,samples_per_question=n,max_new_tokens=cap,
            max_model_len=((maximum+cap+127)//128)*128,maximum_prompt_tokens=maximum,
            enable_thinking=thinking,cot=cot,sampling=sampling,world_size=8,gpu_ids=list(range(8)),
            seed='1234 + samples_per_question * question_index + sample_index',dtype='bfloat16',
            prompts_sha256=sha(directory/'prompts.json'),source_test_sha256=sha(destination/'test.json'),
            official_sampling_source='https://huggingface.co/Qwen/Qwen3-1.7B#best-practices',
            scorer='strict bare choice for direct; SDPO XML choice for CoT, final answer after thinking closes',
            clipping_policy='Retain all clipped generations; no forced closing think or answer continuation',
            training_checkpoint_loaded=False,wandb_enabled=False)
        assert p['max_model_len']<=32768;save(directory/'protocol.json',p)
        files=[SCRIPT,Path(native.__file__).resolve(),Path(native.suite.__file__).resolve()]
        hashes={str(f.relative_to(ROOT)):sha(f) for f in files};save(root/'implementation.json',hashes)
        for f in files:
            target=root/'source_snapshot'/f.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    save(BASE/'manifest.json',dict(modes=list(MODES),model=MODEL,questions=210,source_revision=revision,
        greedy_note='One deterministic generation per question; repeated greedy avg@8 is identical',
        avg_at_8_definition='Mean correctness over 210 questions times 8 independent samples, not pass@8'))

def merge(mode):
    root=BASE/mode;directory=root/MODEL;p=json.loads((directory/'protocol.json').read_text())
    prompts=json.loads((directory/'prompts.json').read_text());n=p['samples_per_question'];rows=[]
    for rank in range(8):
        path=directory/'generation'/f'rank-{rank:02d}.jsonl';done=json.loads(path.with_suffix('.complete.json').read_text())
        assert done['complete'] and done['output_sha256']==sha(path) and done['protocol_sha256']==sha(directory/'protocol.json')
        shard=read(path);expected=native.requests(prompts[rank::8],n);assert len(shard)==len(expected)
        for actual,exp in zip(shard,expected,strict=True):
            assert all(actual[k]==v for k,v in exp.items()) and score(actual)==actual
        rows+=shard
    rows.sort(key=lambda r:(r['index'],r['sample_index']))
    assert [(r['index'],r['sample_index']) for r in rows]==[(i,s) for i in range(210) for s in range(n)]
    jsonl(directory/'scored.jsonl',rows)
    summary=dict(complete=True,mode=mode,questions=210,responses=len(rows),samples_per_question=n,
        correct=sum(r['acc'] for r in rows),accuracy=sum(r['acc'] for r in rows)/len(rows),
        avg_at_8=sum(r['acc'] for r in rows)/len(rows),deterministic_greedy_equivalence=n==1,
        sample_accuracies=[sum(r['acc'] for r in rows if r['sample_index']==s)/210 for s in range(n)],
        clipped=sum(r['finish_reason']=='length' for r in rows),clipping_rate=sum(r['finish_reason']=='length' for r in rows)/len(rows),
        mean_tokens=sum(r['tokens'] for r in rows)/len(rows),extraction_status_counts=dict(Counter(r['extraction_status'] for r in rows)),
        protocol=p,scored_sha256=sha(directory/'scored.jsonl'))
    save(directory/'summary.json',summary);return summary

def queue():
    lock=(BASE/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);children=[];results={}
    def stopped(sig,frame):raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    try:
        usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        assert all(int(x)<512 for x in usage.splitlines()),'Eight GPUs must be idle'
        for mode in MODES:
            directory=BASE/mode/MODEL
            if not (directory/'summary.json').exists():
                save(BASE/'phase.json',dict(phase=mode,time=time.time(),pid=os.getpid()))
                for rank in range(8):
                    path=directory/'generation'/f'rank-{rank:02d}.complete.json'
                    if path.exists():continue
                    env=environment();env.update(CUDA_VISIBLE_DEVICES=str(rank),ARC_SFT_GPU_IDS=str(rank))
                    with (directory/f'rank-{rank:02d}.log').open('a') as log:
                        children.append(subprocess.Popen([sys.executable,'-u',str(SCRIPT),'worker','--mode',mode,'--rank',str(rank)],
                            cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
                while any(c.poll() is None for c in children):
                    assert all(c.poll() in (None,0) for c in children),'Worker failed; inspect rank logs'
                    time.sleep(2)
                assert all(c.returncode==0 for c in children),'Worker failed';children.clear()
                merge(mode)
            results[mode]=json.loads((directory/'summary.json').read_text());save(BASE/'summary.json',results)
        save(BASE/'phase.json',dict(phase='complete',time=time.time()))
    except BaseException as error:
        save(BASE/'failure.json',dict(error=repr(error),time=time.time()));raise
    finally:
        for child in children:
            if child.poll() is None:stop_job(child)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['start','queue','worker'])
    parser.add_argument('--mode',choices=MODES);parser.add_argument('--rank',type=int,default=0);args=parser.parse_args()
    if args.stage=='worker':native.score=score;native.worker(BASE/args.mode,MODEL,args.rank)
    elif args.stage=='queue':queue()
    else:
        if not (BASE/'manifest.json').exists():prepare()
        with (BASE/'controller.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'queue'],cwd=ROOT,env=environment(),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(BASE/'controller.json',dict(pid=child.pid,time=time.time()));print(child.pid)
