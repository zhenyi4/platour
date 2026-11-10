"""Native persistent vLLM decode with exact, in-place full-text-model weight reload."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def tensor_digest(model):
    import torch
    value=hashlib.sha256()
    for name,p in model.named_parameters():
        value.update(name.encode())
        value.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def reload_model(model,path,file_sha):
    import safetensors
    import torch
    target=model.language_model
    with Path(path).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==file_sha
    with safetensors.safe_open(path,framework='pt',device='cpu') as source:
        names=source.keys()
        assert names and all(name.startswith(('model.','lm_head.')) for name in names)
        loaded=target.load_weights((name,source.get_tensor(name)) for name in names)
    aliases=dict(target.named_parameters(remove_duplicate=False))
    assert loaded<=set(aliases),f'vLLM loaded unknown weight names: {sorted(loaded-set(aliases))}'
    expected={id(value) for _,value in target.named_parameters()}
    covered={id(aliases[name]) for name in loaded}
    assert expected==covered,f'vLLM full-weight coverage mismatch: {len(expected-covered)} distinct parameters missing'
    torch.cuda.synchronize()
    return dict(loaded_parameters=len(loaded),actual_parameter_sha256=tensor_digest(target),file_sha256=file_sha,
                all_text_weights_reloaded=True)


class FullWeightExtension:
    def load_full_model(self,path,file_sha):
        return reload_model(self.model_runner.model,path,file_sha)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--max-model-len',type=int,default=18432)
    parser.add_argument('--memory-fraction',type=float,default=.25)
    args=parser.parse_args()
    from vllm import LLM,SamplingParams
    import torch
    from qwen35_bridge_opd_core import save
    args.directory.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    llm=LLM(model=str(args.model),dtype='bfloat16',tensor_parallel_size=1,max_model_len=args.max_model_len,
        max_num_seqs=16,max_num_batched_tokens=4096,gpu_memory_utilization=args.memory_fraction,
        enable_prefix_caching=False,seed=42,disable_log_stats=True,language_model_only=True,
        limit_mm_per_prompt={'image':0,'video':0},attention_config={'backend':'FLASH_ATTN'},
        worker_extension_cls='full4b_generation.FullWeightExtension')
    tokenizer=llm.get_tokenizer()
    identity=None
    save(args.directory/'ready.json',dict(ready=True,pid=os.getpid(),load_seconds=time.monotonic()-started,
                                        native_model=True,bridge=False,max_model_len=args.max_model_len))
    try:
        for line in __import__('sys').stdin:
            request=json.loads(line)
            if request.get('command')=='stop':break
            result=Path(request['result'])
            try:
                if request['command']=='sync':
                    reports=llm.collective_rpc('load_full_model',kwargs=dict(path=request['path'],file_sha=request['file_sha256']))
                    assert len(reports)==1 and reports[0]['all_text_weights_reloaded']
                    identity=request['state_sha256']
                    save(result,dict(complete=True,identity=identity,reports=reports))
                else:
                    assert identity==request['state_sha256'],'Decode must use current full-model weights'
                    rows=request['rows']
                    config=request['sampling']
                    cap=request['cap']
                    parameters=[SamplingParams(n=1,max_tokens=cap,seed=row['sampling_seed'],
                        logprobs=1 if request.get('logprobs') else None,**config) for row in rows]
                    tick=time.monotonic()
                    outputs=llm.generate([dict(prompt_token_ids=row['prompt_token_ids']) for row in rows],parameters,use_tqdm=False)
                    decoded=[]
                    for row,output in zip(rows,outputs,strict=True):
                        a=output.outputs[0]
                        ids=list(a.token_ids)
                        assert 0<len(ids)<=cap and (a.finish_reason!='length' or len(ids)==cap)
                        entry=dict(row,response=tokenizer.decode(ids,skip_special_tokens=True),response_token_ids=ids,
                            tokens=len(ids),finish_reason=a.finish_reason,stop_reason=a.stop_reason,
                            full_model_state_sha256=identity)
                        if request.get('logprobs'):
                            entry['rollout_log_probs']=[float(lp[token].logprob) for token,lp in zip(ids,a.logprobs,strict=True)]
                            assert len(entry['rollout_log_probs'])==len(ids)
                        decoded.append(entry)
                    save(result,dict(complete=True,rows=decoded,state_sha256=identity,seconds=time.monotonic()-tick))
            except BaseException as error:
                import traceback
                save(result,dict(complete=False,error=repr(error),traceback=traceback.format_exc()))
                raise
    finally:
        llm.llm_engine.engine_core.shutdown()
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()
