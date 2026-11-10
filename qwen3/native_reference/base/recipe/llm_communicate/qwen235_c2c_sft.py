"""Original C2CProjector with HF5 KV-cache transport and a frozen FP8 teacher."""
import json
import os
from pathlib import Path
import sys

from arc_bridge_sft_common import ROOT,save


def load_bridge(base,checkpoint=None,devices=None,student_device=None):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
    from transformers import DynamicCache,AutoTokenizer,AutoModelForCausalLM
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from qwen3_235b_trivia_hidden_sft import load_bridge as load_backbones
    sys.path.insert(0,str(ROOT/'third_party/c2c'))
    from rosetta.model.projector import C2CProjector
    p=json.loads((Path(base)/'protocol.json').read_text())
    if p.get('teacher_family')=='qwen3_dense':
        if devices is None:
            primary=2*int(os.environ.get('LOCAL_RANK',0)) if p.get('distributed_sft') else 0
            devices=[primary,primary+1] if torch.cuda.device_count()>1 else [0,0]
        if student_device is None:student_device=devices[0]
        options=dict(dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
        student=AutoModelForCausalLM.from_pretrained(p['student_checkpoint'],device_map={'':student_device},**options)
        teacher=AutoModelForCausalLM.from_pretrained(p['teacher_checkpoint'],device_map={'':devices[1]},**options)
    else:
        student,teacher=load_backbones(base,devices=devices,student_device=student_device,backbones_only=True)
    student.requires_grad_(False).eval();teacher.requires_grad_(False).eval()
    device=student.device;teacher_device=teacher.get_input_embeddings().weight.device
    tokenizer=AutoTokenizer.from_pretrained(p['student_checkpoint'],local_files_only=True)

    class Bridge(nn.Module):
        def __init__(self):
            super().__init__();self.student=student;self.teacher=teacher;self.pad_token_id=tokenizer.pad_token_id
            self.layer_map=p['layer_map']
            self.maps=nn.ModuleList([C2CProjector(source_dim=teacher.config.head_dim,target_dim=student.config.head_dim,
                source_num_heads=teacher.config.num_key_value_heads,target_num_heads=student.config.num_key_value_heads,
                **p['projector'],dtype=torch.bfloat16).to(device) for _ in self.layer_map])

        def bridge_parameters(self):return list(self.maps.parameters())

        def projected_cache(self,ids,mask):
            positions=(mask.cumsum(-1)-1).clamp(min=0)
            with torch.no_grad():
                source=self.teacher.model(input_ids=ids.to(teacher_device),attention_mask=mask.to(teacher_device),
                    position_ids=positions.to(teacher_device),use_cache=True).past_key_values
                target=self.student.model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=True).past_key_values
            cache=DynamicCache(config=self.student.config)
            for i,(projector,layer) in enumerate(zip(self.maps,self.layer_map,strict=True)):
                args=(source.layers[layer].keys.to(device),source.layers[layer].values.to(device),
                      target.layers[i].keys,target.layers[i].values)
                def project(sk,sv,tk,tv,module=projector):return module((sk,sv),(tk,tv))
                k,v=(activation_checkpoint(project,*args,use_reentrant=False) if torch.is_grad_enabled() else project(*args))
                cache.update(k.to(self.student.dtype),v.to(self.student.dtype),i)
            return cache

        def chat_loss(self,batch,evaluate_lm=False):
            # FP8 teacher prefill runs without padding. Combine exact response-token
            # sums before the shared trainer applies global token normalization.
            values=[];tokens=[]
            for i in range(len(batch['input_ids'])):
                n=int(batch['attention_mask'][i].sum());q=int(batch['prefix_mask'][i].sum())
                ids=batch['input_ids'][i:i+1,:n]
                cache=self.projected_cache(ids[:,:q-1],torch.ones_like(ids[:,:q-1]))
                hidden=self.student.model(input_ids=ids[:,q-1:],attention_mask=torch.ones_like(ids),
                    position_ids=torch.arange(q-1,n,device=device)[None],past_key_values=cache,use_cache=True).last_hidden_state
                targets=ids[:,q:];predictors=hidden[:,:-1]
                def ce(h,y):return F.cross_entropy(self.student.lm_head(h).float().reshape(-1,self.student.config.vocab_size),y.reshape(-1),reduction='sum')
                value=activation_checkpoint(ce,predictors,targets,use_reentrant=False) if torch.is_grad_enabled() else ce(predictors,targets)
                values.append(value);tokens.append(targets.numel())
            loss=torch.stack(values).sum()/sum(tokens)
            return loss.new_zeros(()),loss,dict(lm_nll=loss)

        def padded_chat_loss_unused(self,batch,evaluate_lm=False):
            prefixes=batch['prefix_mask'].sum(-1).tolist();lengths=batch['attention_mask'].sum(-1).tolist()
            width=max(prefixes)-1;tail_width=max(n-q+1 for n,q in zip(lengths,prefixes))
            b=len(prefixes)
            ids=torch.full((b,width),self.pad_token_id,device=device,dtype=torch.long);mask=torch.zeros_like(ids)
            tail=torch.full((b,tail_width),self.pad_token_id,device=device,dtype=torch.long);tail_mask=torch.zeros_like(tail)
            labels=torch.full_like(tail,-100)
            for i,(n,q) in enumerate(zip(lengths,prefixes)):
                ids[i,width-q+1:]=batch['input_ids'][i,:q-1];mask[i,width-q+1:]=1
                size=n-q+1;tail[i,:size]=batch['input_ids'][i,q-1:n];tail_mask[i,:size]=1
                labels[i,1:size]=batch['input_ids'][i,q:n]
            cache=self.projected_cache(ids,mask)
            positions=torch.tensor(prefixes,device=device)[:,None]-1+torch.arange(tail_width,device=device)[None]
            hidden=self.student.model(input_ids=tail,attention_mask=torch.cat((mask,tail_mask),-1),
                position_ids=positions,past_key_values=cache,use_cache=True).last_hidden_state
            active=labels[:,1:]!=-100;predictors=hidden[:,:-1][active];targets=labels[:,1:][active]
            pieces=[]
            for offset in range(0,len(targets),128):
                h=predictors[offset:offset+128];y=targets[offset:offset+128]
                def ce(h,y):return F.cross_entropy(self.student.lm_head(h).float(),y,reduction='sum')
                pieces.append(activation_checkpoint(ce,h,y,use_reentrant=False) if torch.is_grad_enabled() else ce(h,y))
            loss=torch.stack(pieces).sum()/len(targets)
            return loss.new_zeros(()),loss,dict(lm_nll=loss)

        @torch.no_grad()
        def prefill_rows(self,rows,tokenizer):
            # Preserve the original unpadded KV token order expected by the vLLM connector.
            results=[]
            for row in rows:
                ids=torch.tensor([row['prompt_token_ids']],device=device)
                cache=self.projected_cache(ids[:,:-1],torch.ones_like(ids[:,:-1]))
                results.append(self.student(input_ids=ids[:,-1:],attention_mask=torch.ones_like(ids),
                    position_ids=torch.tensor([[ids.shape[1]-1]],device=device),past_key_values=cache,use_cache=True,logits_to_keep=1))
            width=max(len(r['prompt_token_ids']) for r in rows);cache=DynamicCache(config=self.student.config)
            for layer in range(self.student.config.num_hidden_layers):
                cache.update(torch.cat([F.pad(o.past_key_values.layers[layer].keys,(0,0,0,width-len(r['prompt_token_ids']))) for o,r in zip(results,rows)]),
                    torch.cat([F.pad(o.past_key_values.layers[layer].values,(0,0,0,width-len(r['prompt_token_ids']))) for o,r in zip(results,rows)]),layer)
            return CausalLMOutputWithPast(logits=torch.cat([o.logits for o in results]),past_key_values=cache)

        def save_projectors(self,directory):
            directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
            torch.save(dict(maps=self.maps.state_dict(),layer_map=self.layer_map,projector=p['projector']),directory/'bridge.pt')

    bridge=Bridge().eval()
    if checkpoint:
        state=torch.load(checkpoint,map_location=device,weights_only=True)
        assert state['layer_map']==bridge.layer_map and state['projector']==p['projector']
        bridge.maps.load_state_dict(state['maps'])
    # Separate audit files for workers with the same logical cuda:0 after
    # CUDA_VISIBLE_DEVICES remapping; sharing a .json.tmp races at os.replace.
    audit_rank=os.environ.get('RANK',str(os.getpid()))
    save(Path(base)/('c2c_initialization.'+str(device).replace(':','-')+'.rank-'+audit_rank+'.json'),dict(
        original_projector_source=str(ROOT/'third_party/c2c/rosetta/model/projector.py'),
        parameters=sum(x.numel() for x in bridge.maps.parameters()),dtype='bfloat16',layer_map=bridge.layer_map,
        source_kv_heads=teacher.config.num_key_value_heads,target_kv_heads=student.config.num_key_value_heads,
        frozen_backbones=True,random_initialization=True,gumbel_training=True,hard_gate_eval=True))
    return bridge
