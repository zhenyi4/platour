"""Eight-GPU zero/uniform Hidden Bridge SFT on fresh 4B GSM8K trajectories."""
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from arc_bridge_sft_common import ROOT,HERE,save,sha
from generate_qwen35_train_suite import read,jsonl,SUFFIX,stop_job

BASE=ROOT/'outputs/llm_communicate/qwen3_4b_to_17b_gsm8k_hidden_zero_sft_ep3_bs64_8gpu'
GENERATION=ROOT/'outputs/llm_communicate/qwen3_17b_4b_8b_235b_gsm8k_train_sft_cap2048'
REVISION='740312add88f781978c0658806c59bc2815b9866'


def prepare(base):
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from hidden_bridge import bridge_layer_map
    from nemotron_chat_data import INDEX_DTYPE
    from qwen3_17b_hidden_sft import sources,SAMPLING
    base.mkdir(parents=True,exist_ok=True)
    assert not (base/'protocol.json').exists(),'Use a fresh run directory'
    source=GENERATION/'Qwen3-4B/scored.jsonl';rows=read(source);assert len(rows)==7473
    generated=json.loads((GENERATION/'Qwen3-4B/summary.json').read_text())
    assert sha(source)==generated['scored_sha256']
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'data/models/Qwen3-1.7B',local_files_only=True)
    data=base/'data';data.mkdir(exist_ok=True)
    split=json.loads((GENERATION/'split.json').read_text());assert len(split['train'])==7398 and len(split['validation'])==75
    save(data/'split.json',split)
    assert sha(data/'split.json')==sha(GENERATION/'split.json')
    records=[];offset=0;index=[]
    with (data/'tokens.bin').open('wb') as f:
        for i,row in enumerate(rows):
            assert row['index']==i
            ids=tokenizer.apply_chat_template([dict(role='user',content=row['input_text'])],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False)
            assert ids==row['prompt_token_ids']
            response=row['response_token_ids'];assert len(response)==row['tokens']<=2048 and response
            assert len(ids)<=1024 and len(ids)+len(response)<=3072
            tokens=ids+response;np.asarray(tokens,dtype=np.uint32).tofile(f)
            index.append((offset,len(ids),len(tokens),0,int(i in split['validation']),i));offset+=len(tokens)
            records.append(row)
    np.asarray(index,dtype=INDEX_DTYPE).tofile(data/'index.bin');jsonl(data/'train.jsonl',records)
    save(data/'complete.json',dict(complete=True,records=7473,train_records=7398,validation_records=75,
        source=str(source),source_sha256=sha(source),filter='None; all incorrect and clipped responses retained',
        original_generated_token_ids=True,synthetic_eos_added=False,clipped=generated['clipped'],
        tokens_sha256=sha(data/'tokens.bin'),index_sha256=sha(data/'index.bin')))
    test_path=Path(hf_hub_download('openai/gsm8k',filename='main/test-00000-of-00001.parquet',
        repo_type='dataset',revision=REVISION,local_dir=str(ROOT/'data/gsm8k')))
    test=pq.read_table(test_path).to_pylist();assert len(test)==1319
    assert not {r['question'].strip() for r in rows}&{r['question'].strip() for r in test}
    prompts=[]
    for i,r in enumerate(test):
        question=r['question'].strip();text=question+SUFFIX
        ids=tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        prompts.append(dict(index=i,id=f'gsm8k_test_{i:06d}',task='gsm8k',question=question,input_text=text,
            gold=r['answer'].rsplit('####',1)[1].strip().replace(',',''),prompt_token_ids=ids))
    save(base/'eval_prompts.json',prompts)
    teacher=json.loads((ROOT/'data/models/Qwen3-4B/config.json').read_text())
    student=json.loads((ROOT/'data/models/Qwen3-1.7B/config.json').read_text())
    mapping=bridge_layer_map(student['num_hidden_layers'],teacher['num_hidden_layers'],'uniform_endpoints')
    p=dict(method='hidden',task='gsm8k',dataset='GSM8K',sender='Qwen3-4B',receiver='Qwen3-1.7B',
        teacher_checkpoint=str(ROOT/'data/models/Qwen3-4B'),student_checkpoint=str(ROOT/'data/models/Qwen3-1.7B'),
        layer_map=mapping,layer_mapping='uniform_endpoints',initialization='zero',
        trainable_parameters=len(mapping)*teacher['hidden_size']*student['hidden_size'],
        examples=7473,train_examples=7398,heldout_examples=75,data_root=str(data),
        source=str(source),source_sha256=sha(source),split_sha256=sha(data/'split.json'),
        epochs=3,batch_size=64,micro_batch=8,gradient_accumulation_steps=1,world_size=8,gpu_ids=list(range(8)),
        steps_per_epoch=116,total_steps=348,learning_rate=1e-4,optimizer='AdamW',betas=[.9,.999],weight_decay=0.,grad_clip=1.,
        scheduler='linear',warmup_ratio=.1,warmup_steps=35,seed=42,max_prompt_length=1024,max_response_length=2048,max_total_length=3072,
        frozen_backbones=True,teacher_response_forward=False,final_prompt_token_native=True,
        objective='global valid-response-token mean CE',filter='None; retain wrong and clipped training trajectories',
        eval_sampling=SAMPLING,eval_samples=2,eval_questions=1319,eval_response_cap=2048,eval_epochs=[3],
        enable_thinking=False,eval_prompts_sha256=sha(base/'eval_prompts.json'),
        test_source=dict(path=str(test_path),sha256=sha(test_path),revision=REVISION),
        prompt_instruction=SUFFIX,wandb_enabled=False,tracking='disabled',reference_generation=str(GENERATION))
    save(base/'protocol.json',p)
    save(base/'data_audit.json',dict(passed=True,records=7473,original_response_token_ids=True,
        exact_receiver_prompt_ids=True,shared_split_sha256=sha(data/'split.json'),test_question_overlap=0,
        incorrect_retained=sum(not r['acc'] for r in rows),clipped_retained=generated['clipped']))
    hashes=sources()
    for file in [Path(__file__).resolve(),HERE/'generate_qwen3_gsm8k_sft_suite.py']:
        hashes[str(file.relative_to(ROOT))]=sha(file)
    save(base/'source_sha256.json',hashes)
    for name in hashes:
        target=base/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)


def environment():
    return dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        PYTHONPATH=str(ROOT/'scripts')+':'+str(HERE)+':'+str(ROOT/'third_party/verl'),
        ARC_SFT_PYTHON=sys.executable,ARC_SFT_SCORING_PYTHON=sys.executable,PLATOUR_GENERATION_PYTHON=sys.executable,
        WANDB_MODE='disabled',WANDB_DISABLED='true',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1')


def run():
    active=None
    def stopped(sig,frame):raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    try:
        if not (BASE/'protocol.json').exists():prepare(BASE)
        usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        assert all(int(x)<512 for x in usage.splitlines()),'Eight GPUs must be idle'
        hashes=json.loads((BASE/'source_sha256.json').read_text())
        def execute(stage,command):
            nonlocal active
            assert all(sha(ROOT/name)==digest for name,digest in hashes.items()),'Source changed during run'
            save(BASE/'phase.json',dict(phase=stage,time=time.time()))
            with (BASE/f'{stage}.log').open('w') as f:
                active=subprocess.Popen(command,cwd=ROOT,env=environment(),stdin=subprocess.DEVNULL,
                    stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                save(BASE/'active.json',dict(phase=stage,pid=active.pid,command=command))
                code=active.wait()
                if code:raise RuntimeError(f'{stage} failed ({code}); see {BASE}/{stage}.log')
            active=None
        trainer=str(HERE/'qwen3_17b_hidden_sft.py')
        if not (BASE/'worst_case.json').exists():execute('profile',[sys.executable,'-u',trainer,'profile','--root',str(BASE)])
        for smoke in [True,False]:
            stage='smoke' if smoke else 'train'
            if not (BASE/stage/'complete.json').exists():
                execute(stage,[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                    trainer,'train','--root',str(BASE),*(['--smoke'] if smoke else [])])
            done=json.loads((BASE/stage/'complete.json').read_text())
            output=BASE/('smoke_eval' if smoke else 'eval')
            if not (output/'summary.json').exists():
                execute(stage+'_eval',[sys.executable,'-u',str(HERE/'eval_qwen3_17b_hidden_sft.py'),'queue',
                    '--root',str(BASE),'--output',str(output),'--checkpoint',done['selected_checkpoint'],*(['--smoke'] if smoke else [])])
        save(BASE/'phase.json',dict(phase='complete',time=time.time()))
    except BaseException as error:
        save(BASE/'phase.json',dict(phase='failed',error=repr(error),time=time.time()));raise
    finally:
        if active is not None and active.poll() is None:stop_job(active)


if __name__=='__main__':run()
