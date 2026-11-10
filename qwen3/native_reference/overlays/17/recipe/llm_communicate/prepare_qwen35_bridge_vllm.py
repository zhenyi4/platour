"""Export bridge prompt memory and benchmark the original HF+FLA decoder.

Run in .runtime/qwen35-bridge. The companion vLLM worker runs in the generation
environment. No model/bridge weights or attention formulas are changed.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
DEFAULT_TRAIN = ROOT/'outputs/llm_communicate/qwen35_35b_to_4b_math_direct_zero_last32_linear_8gpu_fresh'
PROMPTS = ROOT/'outputs/llm_communicate/qwen35_math500_official_sample4_cap4096/Qwen3.5-4B/prompts.json'
FORMAT = 'qwen35-hf-bridge-hybrid-memory-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory_key(ids, checkpoint_sha, mode):
    return hashlib.sha256(json.dumps([ids,checkpoint_sha,mode],separators=(',',':')).encode()).hexdigest()


def save_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def cache_to_cpu(cache):
    import torch
    tensors = {}
    for i,layer in enumerate(cache.layers):
        if hasattr(layer,'recurrent_states'):
            # HF/FLA is [head,key,value]; vLLM recurrent kernels are [head,value,key].
            tensors[f'layer.{i}.recurrent'] = layer.recurrent_states[0].transpose(-1,-2).contiguous().cpu()
            # HF keeps four inputs; vLLM needs the preceding three for kernel=4.
            tensors[f'layer.{i}.conv'] = layer.conv_states[0][:,:,-3:].transpose(-1,-2).contiguous().cpu()
        else:
            tensors[f'layer.{i}.kv'] = torch.stack((layer.keys,layer.values),1).transpose(2,3).contiguous().cpu()
    return tensors


def export_memory(batch_memory,row,checkpoint,mode,directory,batch_index=0):
    from safetensors.torch import save_file
    tokens = row['prompt_token_ids']
    key = memory_key(tokens,checkpoint,mode)
    tensors = {name:(value[batch_index,:, -len(tokens):] if name.endswith('.kv') else value[batch_index]).contiguous()
        for name,value in batch_memory.items()}
    path = directory/(key+'.safetensors')
    temp = path.with_suffix('.tmp')
    save_file(tensors,temp,metadata=dict(format=FORMAT,checkpoint_sha256=checkpoint,
        bridge_mode=mode,prefix_tokens=str(len(tokens)),memory_key=key))
    temp.replace(path)
    return dict(memory_key=key,prefix_tokens=len(tokens),qwen35_bridge_memory=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_TRAIN/'train/epoch-3/bridge.pt')
    p.add_argument('--mode',choices=['full_hidden','local_memory'],default='full_hidden')
    p.add_argument('--questions',type=int,default=16)
    p.add_argument('--indices',help='Comma-separated MATH500 indices; overrides --questions')
    p.add_argument('--tokens',type=int,default=256)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--export-only',action='store_true')
    p.add_argument('--sampling',action='store_true',help='Sample the handoff token with the existing official HF sampler')
    args = p.parse_args()
    import torch
    from transformers import AutoTokenizer
    from eval_qwen35_35b_4b_math_bridge_8gpu import prefill, decode_step, prompt_tensors, sample_tokens
    if args.mode == 'local_memory':
        from qwen35_cache_only_math_bridge import install_loader, prefill
        install_loader()
    from qwen35_35b_4b_math_bridge import load_bridge
    args.output.mkdir(parents=True,exist_ok=True)
    directory = args.output/'memory'
    directory.mkdir(exist_ok=True)
    started = time.monotonic()
    bridge,runtime = load_bridge(ROOT)
    state = torch.load(args.checkpoint,map_location='cuda',weights_only=False)
    bridge.maps.load_state_dict(state['P'])
    bridge.eval().requires_grad_(False)
    checkpoint = digest(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'data/models/Qwen3.5-4B',local_files_only=True)
    pad = tokenizer.pad_token_id
    all_rows = json.loads(PROMPTS.read_text())
    indices = list(map(int,args.indices.split(','))) if args.indices else list(range(args.questions))
    rows = [all_rows[i] for i in indices]
    manifest = dict(format=FORMAT,checkpoint=str(args.checkpoint),checkpoint_sha256=checkpoint,
        bridge_mode=args.mode,model_path=str(ROOT/'data/models/Qwen3.5-4B'),
        prompts_path=str(PROMPTS),prompts_sha256=digest(PROMPTS),rows=[],tokens=args.tokens,
        batch=args.batch,runtime=runtime,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    manifest.update(handoff='Full original prompt memory + first HF response token',
        first_token_decoding='official_sampling' if args.sampling else 'greedy')
    with torch.inference_mode():
        # Warm up teacher, student prefill and recurrent decode before timing.
        ids,mask = prompt_tensors(rows[:1],pad,'cuda')
        out = prefill(bridge,ids,mask)
        out,_ = decode_step(bridge.student,out.logits[:,-1].argmax(-1,keepdim=True),mask,out.past_key_values)
        torch.cuda.synchronize()
        export_started = time.monotonic()
        manifest['hf_load_and_warm_seconds']=export_started-started
        for offset in range(0,len(rows),args.batch):
            batch = rows[offset:offset+args.batch]
            ids,mask = prompt_tensors(batch,pad,'cuda')
            output = prefill(bridge,ids,mask)
            cache = output.past_key_values
            first_tokens = output.logits[:,-1].argmax(-1)
            if args.sampling:
                uniforms=torch.stack([torch.rand(args.tokens,device='cuda',
                    generator=torch.Generator(device='cuda').manual_seed(1234+4*r['index']))[0] for r in batch])
                first_tokens=sample_tokens(output.logits[:,-1],torch.zeros_like(output.logits[:,-1],dtype=torch.bool),uniforms)
            first_tokens=first_tokens.tolist()
            batch_memory = cache_to_cpu(cache)
            for i,row in enumerate(batch):
                params = export_memory(batch_memory,row,checkpoint,args.mode,directory,i)
                manifest['rows'].append(dict(**row,transfer=params,hf_first_token=first_tokens[i]))
            del cache,batch_memory
        torch.cuda.synchronize()
        manifest['memory_export_seconds'] = time.monotonic()-export_started
        save_json(args.output/'manifest.json',manifest)
        print(json.dumps(dict(phase='memory_exported',rows=len(rows),seconds=manifest['memory_export_seconds'])),flush=True)
        if args.export_only:
            return
        responses=[]
        measured=[]
        first_logits=[]
        forced_steps = [n for n in (0,1,8,32,127) if n < args.tokens]
        forced_logits = []
        for offset in range(0,len(rows),args.batch):
            batch=rows[offset:offset+args.batch]
            ids,mask=prompt_tensors(batch,pad,'cuda')
            torch.cuda.synchronize()
            t0=time.monotonic()
            out=prefill(bridge,ids,mask)
            torch.cuda.synchronize()
            t1=time.monotonic()
            # Save complete first-token logits for backend parity checks.
            initial_logits = out.logits[:,-1].detach().clone()
            probes = [initial_logits]
            nxt=out.logits[:,-1].argmax(-1,keepdim=True)
            generated=[nxt]
            # Fixed-work benchmark deliberately ignores EOS for both backends.
            for step in range(1,args.tokens):
                out,mask=decode_step(bridge.student,nxt,mask,out.past_key_values)
                if step in forced_steps:
                    probes.append(out.logits[:,-1].detach().clone())
                nxt=out.logits[:,-1].argmax(-1,keepdim=True)
                generated.append(nxt)
            torch.cuda.synchronize()
            t2=time.monotonic()
            first_logits.extend(initial_logits.float().cpu().unbind(0))
            forced_logits.extend(torch.stack(probes,1).cpu().unbind(0))
            tokens=torch.cat(generated,1).tolist()
            for row,answer in zip(batch,tokens):
                responses.append(dict(index=row['index'],response_token_ids=answer,response=tokenizer.decode(answer)))
            measured.append(dict(batch_size=len(batch),tokens=len(batch)*args.tokens,
                prefill_seconds=t1-t0,decode_seconds=t2-t1,total_seconds=t2-t0))
            save_json(args.output/'naive.live.json',dict(completed=len(responses),total=len(rows),batches=measured))
            print(json.dumps(dict(phase='naive_benchmark',completed=len(responses),batch=measured[-1])),flush=True)
        torch.save(torch.stack(first_logits),args.output/'naive_first_logits.pt')
        torch.save(dict(steps=forced_steps,logits=torch.stack(forced_logits)),args.output/'naive_forced_logits.pt')
        save_json(args.output/'naive_responses.json',responses)
        summary=dict(backend='HF+FLA+SDPA',fixed_tokens=True,ignore_eos=True,
            max_new_tokens=args.tokens,questions=len(rows),batch=args.batch,batches=measured,
            prefill_seconds=sum(x['prefill_seconds'] for x in measured),
            decode_seconds=sum(x['decode_seconds'] for x in measured),
            total_seconds=sum(x['total_seconds'] for x in measured),
            generated_tokens=len(rows)*args.tokens,load_warm_export_and_run_seconds=time.monotonic()-started,
            checkpoint_sha256=checkpoint,runtime=runtime)
        save_json(args.output/'naive_summary.json',summary)
        print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
