"""Batched right-padded KV prefill adapter for the existing 235B-version C2C."""
import types

from qwen235_c2c_sft import load_bridge as serial_load_bridge


def load_bridge(base,checkpoint=None,devices=None,student_device=None):
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
    from transformers import DynamicCache
    from transformers.modeling_outputs import CausalLMOutputWithPast
    bridge=serial_load_bridge(base,checkpoint,devices,student_device)
    bridge.serial_chat_loss=bridge.chat_loss
    bridge.serial_prefill_rows=bridge.prefill_rows

    def pack(self,batch):
        prefixes=batch['prefix_mask'].sum(-1).tolist();lengths=batch['attention_mask'].sum(-1).tolist()
        width=max(prefixes)-1;tail_width=max(n-q+1 for n,q in zip(lengths,prefixes))
        device=self.student.device;b=len(prefixes)
        ids=torch.full((b,width),self.pad_token_id,device=device,dtype=torch.long);mask=torch.zeros_like(ids)
        tail=torch.full((b,tail_width),self.pad_token_id,device=device,dtype=torch.long);tail_mask=torch.zeros_like(tail)
        labels=torch.full_like(tail,-100)
        for i,(n,q) in enumerate(zip(lengths,prefixes)):
            ids[i,:q-1]=batch['input_ids'][i,:q-1];mask[i,:q-1]=1
            size=n-q+1;tail[i,:size]=batch['input_ids'][i,q-1:n];tail_mask[i,:size]=1
            labels[i,1:size]=batch['input_ids'][i,q:n]
        # Physical KV padding is masked; RoPE positions follow each real prompt.
        positions=torch.tensor(prefixes,device=device)[:,None]-1+torch.arange(tail_width,device=device)[None]
        return ids,mask,tail,tail_mask,labels,positions

    def chat_loss(self,batch,evaluate_lm=False):
        ids,mask,tail,tail_mask,labels,positions=pack(self,batch)
        cache=self.projected_cache(ids,mask)
        hidden=self.student.model(input_ids=tail,attention_mask=torch.cat((mask,tail_mask),-1),
            position_ids=positions,past_key_values=cache,use_cache=True).last_hidden_state
        active=labels[:,1:]!=-100;predictors=hidden[:,:-1][active];targets=labels[:,1:][active]
        pieces=[]
        def ce(h,y):return F.cross_entropy(self.student.lm_head(h).float(),y,reduction='sum')
        for offset in range(0,len(targets),128):
            h=predictors[offset:offset+128];y=targets[offset:offset+128]
            pieces.append(activation_checkpoint(ce,h,y,use_reentrant=False) if torch.is_grad_enabled() else ce(h,y))
        loss=torch.stack(pieces).sum()/len(targets)
        return loss.new_zeros(()),loss,dict(lm_nll=loss)

    @torch.no_grad()
    def prefill_rows(self,rows,tokenizer):
        lengths=[len(r['prompt_token_ids']) for r in rows];width=max(lengths)-1;device=self.student.device
        ids=torch.full((len(rows),width),self.pad_token_id,device=device,dtype=torch.long);mask=torch.zeros_like(ids)
        last=[]
        for i,row in enumerate(rows):
            n=lengths[i]-1;ids[i,:n]=torch.tensor(row['prompt_token_ids'][:-1],device=device);mask[i,:n]=1
            last.append(row['prompt_token_ids'][-1])
        cache=self.projected_cache(ids,mask)
        output=self.student(input_ids=torch.tensor(last,device=device)[:,None],
            attention_mask=torch.cat((mask,torch.ones_like(mask[:,:1])),-1),
            position_ids=torch.tensor(lengths,device=device)[:,None]-1,past_key_values=cache,use_cache=True,logits_to_keep=1)
        # Remove the padded gap before exporting KV to the vLLM connector.
        compact=DynamicCache(config=self.student.config)
        for layer_index,layer in enumerate(output.past_key_values.layers):
            pairs=[]
            for value in [layer.keys,layer.values]:
                pieces=[F.pad(torch.cat((value[i:i+1,:,:n-1],value[i:i+1,:,width:width+1]),2),
                    (0,0,0,width+1-n)) for i,n in enumerate(lengths)]
                pairs.append(torch.cat(pieces,0))
            compact.update(*pairs,layer_index)
        return CausalLMOutputWithPast(logits=output.logits,past_key_values=compact)

    bridge.chat_loss=types.MethodType(chat_loss,bridge)
    bridge.prefill_rows=types.MethodType(prefill_rows,bridge)
    return bridge
