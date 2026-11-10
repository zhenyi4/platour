"""Native standalone Qwen3.5 rollout worker with explicit actor weight updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from qwen35_bridge_opd_core import save


def patch_native_lora_gdn_mapping():
    # vLLM 0.17.1's fused GDN layer has four physical slices, but its
    # advertised LoRA mapping has two logical inputs. Expand only the LoRA
    # mapping; native checkpoint loading and model computation are unchanged.
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase,Qwen3_5ForConditionalGeneration
    for cls in (Qwen3_5ForCausalLMBase,Qwen3_5ForConditionalGeneration):
        cls.packed_modules_mapping=dict(cls.packed_modules_mapping,
            in_proj_qkvz=['in_proj_q','in_proj_k','in_proj_v','in_proj_z'])


if os.environ.get('PLATOUR_NATIVE_LORA_GDN_WORKER')=='1':
    patch_native_lora_gdn_mapping()


def tensor_digest(state):
    import torch
    h=hashlib.sha256()
    for name,value in state.items():
        h.update(name.encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_actor(model,path,expected_digest):
    """File data plane; native vLLM loader handles fused GDN/QKV/MLP tensors."""
    import torch
    state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if 'actor' in state:
        assert state['mode']=='full'
        state=state['actor']
    assert tensor_digest(state)==expected_digest,'Actor weight identity mismatch'
    # HF text-only model -> original official multimodal checkpoint names.
    weights=((('model.language_model.'+n[len('model.'):]) if n.startswith('model.') else n,t)
             for n,t in state.items())
    loaded=model.load_weights(weights)
    assert loaded,'Native vLLM loader consumed no weights'
    language=getattr(model,'language_model',model)
    # Audit every unfused embedding, norm and weight as well as each fused
    # projection shard, rather than accepting only an advertised SHA.
    actual=dict(language.named_parameters())
    fused={'q_proj':('qkv_proj',0,3),'k_proj':('qkv_proj',1,3),'v_proj':('qkv_proj',2,3),
           'gate_proj':('gate_up_proj',0,2),'up_proj':('gate_up_proj',1,2),
           'in_proj_qkv':('in_proj_qkvz',0,2),'in_proj_z':('in_proj_qkvz',1,2),
           'in_proj_b':('in_proj_ba',0,2),'in_proj_a':('in_proj_ba',1,2)}
    verified=0
    for name,value in state.items():
        if name=='lm_head.weight' and language.config.tie_word_embeddings:
            target=language.model.embed_tokens.weight
        elif name in actual:
            target=actual[name]
        else:
            stem,leaf=name.rsplit('.',1)
            module=stem.rsplit('.',1)[-1]
            assert module in fused,(name,list(actual)[:10])
            packed,slot,n=fused[module]
            parent=stem.rsplit('.',1)[0]
            target=actual[parent+'.'+packed+'.'+leaf]
            sizes=[]
            parts=({'qkv_proj':['q_proj','k_proj','v_proj'],
                    'gate_up_proj':['gate_proj','up_proj'],
                    'in_proj_qkvz':['in_proj_qkv','in_proj_z'],
                    'in_proj_ba':['in_proj_b','in_proj_a']})[packed]
            for part in parts:
                sizes.append(state[parent+'.'+part+'.'+leaf].shape[0])
            target=target.narrow(0,sum(sizes[:slot]),sizes[slot])
        assert target.shape==value.shape,(name,target.shape,value.shape)
        assert torch.equal(target.detach().cpu(),value),(name,'actor/vLLM weight mismatch')
        verified+=1
    return dict(actor_state_sha256=expected_digest,verified_tensors=verified,
                loaded_native_tensors=len(loaded),all_actor_weights_equal=True)


class NativeActorWorkerExtension:
    def update_actor(self,path,expected_digest):
        return load_actor(self.model_runner.model,path,expected_digest)

    def update_actor_lora(self,path,adapter_id,expected_digest,previous_id):
        import torch
        from vllm.lora.request import LoRARequest
        state=torch.load(str(Path(path)/'actor_source.pt'),map_location='cpu',weights_only=True,mmap=True)
        assert tensor_digest(state)==expected_digest
        if previous_id:self.remove_lora(previous_id)
        self.add_lora(LoRARequest('actor',adapter_id,path))
        manager=self.model_runner.lora_manager._adapter_manager
        adapter=manager.get_adapter(adapter_id)
        assert adapter is not None
        mapping={'q_proj':('qkv_proj',0),'k_proj':('qkv_proj',1),'v_proj':('qkv_proj',2),
                 'gate_proj':('gate_up_proj',0),'up_proj':('gate_up_proj',1),
                 'in_proj_qkv':('in_proj_qkvz',0),'in_proj_z':('in_proj_qkvz',3),
                 'in_proj_b':('in_proj_ba',0),'in_proj_a':('in_proj_ba',1)}
        verified=0
        for name,value in state.items():
            stem,factor=name.rsplit('.',1)
            if getattr(self.model_runner.model.config,'model_type',None) not in ('llama','qwen3'):
                stem='language_model.'+stem
            if stem.endswith('.in_proj_qkv'):
                layer=adapter.loras[stem[:-len('in_proj_qkv')]+'in_proj_qkvz']
                sizes=[2048,2048,4096]
                values=[value]*3 if factor=='lora_A' else list(value.split(sizes,dim=0))
                loaded=layer.lora_a if factor=='lora_A' else layer.lora_b
                for tensor,expected in zip(loaded[:3],values,strict=True):
                    assert tensor is not None and tensor.shape==expected.shape,(name,'GDN LoRA shape mismatch')
                    assert torch.equal(tensor.cpu(),expected.to(tensor.dtype)),(name,'GDN LoRA values mismatch')
                verified+=1
                continue
            if stem in adapter.loras:
                layer=adapter.loras[stem]
                tensor=layer.lora_a if factor=='lora_A' else layer.lora_b
            else:
                parent,leaf=stem.rsplit('.',1)
                packed,slot=mapping[leaf]
                layer=adapter.loras[parent+'.'+packed]
                tensor=(layer.lora_a if factor=='lora_A' else layer.lora_b)[slot]
            assert tensor is not None and tensor.shape==value.shape,(name,'LoRA shape mismatch')
            assert torch.equal(tensor.cpu(),value.to(tensor.dtype)),(name,'LoRA values mismatch')
            verified+=1
        return dict(actor_state_sha256=expected_digest,verified_tensors=verified,
            all_actor_weights_equal=True,native_lora=True,adapter_id=adapter_id,
            adapter_forward_dtype='bfloat16',factor_master_dtype='float32')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--model-path',type=Path,required=True)
    parser.add_argument('--memory-fraction',type=float,default=.12)
    parser.add_argument('--max-model-len',type=int,default=6144)
    parser.add_argument('--max-num-seqs',type=int,default=32)
    parser.add_argument('--decode-batch-size',type=int,default=32)
    parser.add_argument('--generation-config',default='vllm')
    args=parser.parse_args()
    protocol=json.loads((args.directory.parents[3]/'protocol.json').read_text())
    use_lora=protocol['actor_training_mode']=='lora'
    is_llama=protocol.get('model_family')=='llama31'
    is_dense=is_llama or protocol.get('model_family')=='qwen3'
    if use_lora and not is_dense:
        os.environ['PLATOUR_NATIVE_LORA_GDN_WORKER']='1'
        patch_native_lora_gdn_mapping()
    from vllm import LLM,SamplingParams
    started=time.monotonic()
    llm=LLM(model=str(args.model_path),dtype='bfloat16',tensor_parallel_size=1,
        max_model_len=args.max_model_len,max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=4096,gpu_memory_utilization=args.memory_fraction,
        generation_config='vllm',seed=42,disable_log_stats=True,
        language_model_only=True,limit_mm_per_prompt={'image':0,'video':0},
        attention_config={'backend':'FLASH_ATTN', **({'flash_attn_version':2} if is_dense else {})},enable_prefix_caching=False,
        mamba_cache_mode='none',mamba_ssm_cache_dtype='float32',
        worker_extension_cls='qwen35_native_actor_vllm_worker.NativeActorWorkerExtension',
        enable_lora=use_lora,**(dict(max_lora_rank=128,max_loras=1,max_cpu_loras=1,lora_dtype='bfloat16') if use_lora else {}))
    stop_options={}
    if is_llama:
        eos=json.loads((args.model_path/'generation_config.json').read_text())['eos_token_id']
        stop_options=dict(stop_token_ids=eos if isinstance(eos,list) else [eos])
    tokenizer=llm.get_tokenizer()
    save(args.directory/'ready.json',dict(pid=os.getpid(),load_seconds=time.monotonic()-started,
        backend='native vLLM; native prefill and all response tokens',actor_sync='native load_weights + exact tensor audit'))
    identity=None
    lora_request=None
    adapter_id=0
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('command')=='stop':break
        target=Path(request['result'])
        try:
            manifest=json.loads(Path(request['manifest']).read_text())
            sync_started=time.monotonic()
            if identity!=manifest['bridge_state_sha256']:
                if use_lora:
                    from vllm.lora.request import LoRARequest
                    previous=adapter_id;adapter_id+=1
                    reports=llm.collective_rpc('update_actor_lora',args=(manifest['actor_weights'],adapter_id,
                        manifest['actor_state_sha256'],previous))
                    lora_request=LoRARequest('actor',adapter_id,manifest['actor_weights'])
                else:
                    reports=llm.collective_rpc('update_actor',args=(manifest['actor_weights'],manifest['actor_state_sha256']))
                identity=manifest['bridge_state_sha256']
            else:reports=[]
            sync_seconds=time.monotonic()-sync_started
            rows=manifest['rows']; responses=[]; batches=[]
            batch=args.decode_batch_size or max(len(rows),1)
            for offset in range(0,len(rows),batch):
                current=rows[offset:offset+batch]; start=time.monotonic()
                params=[SamplingParams(max_tokens=manifest['tokens'],seed=r['sampling_seed'],
                    logprobs=1 if manifest['return_logprobs'] else None,**stop_options,**manifest['sampling']) for r in current]
                outputs=llm.generate([dict(prompt_token_ids=r['prompt_token_ids']) for r in current],params,
                    use_tqdm=False,lora_request=lora_request)
                for row,out in zip(current,outputs,strict=True):
                    answer=out.outputs[0]; ids=list(answer.token_ids)
                    extra={}
                    if manifest['return_logprobs']:
                        extra['rollout_log_probs']=[float(lp[t].logprob) for t,lp in zip(ids,answer.logprobs,strict=True)]
                    responses.append(dict(**row,**extra,response_token_ids=ids,
                        response=tokenizer.decode(ids,skip_special_tokens=True),tokens=len(ids),
                        finish_reason=answer.finish_reason,stop_reason=answer.stop_reason,
                        bridge_state_sha256=identity))
                batches.append(dict(responses=len(current),seconds=time.monotonic()-start))
                save(target.with_suffix('.live.json'),dict(completed=len(responses),total=len(rows)))
            save(target,dict(complete=True,rows=responses,batches=batches,
                seconds=sum(x['seconds'] for x in batches),actor_sync_seconds=sync_seconds,
                actor_weight_audits=reports,bridge_state_sha256=identity))
        except BaseException as error:
            traceback.print_exc()
            save(target,dict(complete=False,error=str(error),traceback=traceback.format_exc()))
            raise


if __name__=='__main__':main()
