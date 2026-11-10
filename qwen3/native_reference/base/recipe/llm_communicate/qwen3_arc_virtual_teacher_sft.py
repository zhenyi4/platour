"""Qwen3 template Gaussian virtual teacher: original initialization, ARC SFT adapter."""
import json
from pathlib import Path

from arc_bridge_sft_common import save,sha


def load_bridge(base,checkpoint=None):
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from qwen35_template_virtual_teacher_bridge import TemplateVirtualTeacherBridge
    p=json.loads((Path(base)/'protocol.json').read_text())
    path=Path(p['virtual_hidden_statistics_path']);assert sha(path)==p['virtual_hidden_statistics_sha256']
    stats=torch.load(path,map_location='cpu',weights_only=True)
    student=AutoModelForCausalLM.from_pretrained(p['student_checkpoint'],dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True,device_map={'':0}).eval().requires_grad_(False)

    class Bridge(TemplateVirtualTeacherBridge):
        # ARC token batches use right padding. Class identities follow each
        # sample's prompt boundary, retaining prefix3 + suffix24 exactly.
        def features(self,ids,mask=None):
            mask=torch.ones_like(ids) if mask is None else mask
            length=mask.sum(-1);positions=torch.arange(ids.shape[1],device=ids.device)[None]
            assert bool((length>28).all())
            classes=torch.full_like(ids,28)
            classes=torch.where(positions<3,positions,classes)
            suffix=(positions>=length[:,None]-25)&(positions<length[:,None])
            classes=torch.where(suffix,3+positions-(length[:,None]-25),classes)
            active=classes.lt(27)&mask.bool()
            assert torch.equal(active.sum(-1),torch.full_like(length,27))
            assert torch.equal(ids[active],self.template_token_ids[classes[active]])
            return tuple(classes for _ in self.layer_map)

        def chat_loss(self,batch,evaluate_lm=False):
            features=self.features(batch['prefix_ids'],batch['prefix_mask'])
            injection=batch['prefix_mask']&features[0].lt(27)
            with self.inject(features,injection):
                hidden=self.student.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).last_hidden_state
            labels=batch['labels'][:,1:];active=labels!=-100
            predictors=hidden[:,:-1][active];targets=labels[active];pieces=[]
            def ce(h,y):return F.cross_entropy(self.student.lm_head(h).float(),y,reduction='sum')
            for offset in range(0,len(targets),128):
                h=predictors[offset:offset+128];y=targets[offset:offset+128]
                pieces.append(activation_checkpoint(ce,h,y,use_reentrant=False) if torch.is_grad_enabled() else ce(h,y))
            loss=torch.stack(pieces).sum()/len(targets)
            return loss.new_zeros(()),loss,dict(lm_nll=loss)

        def prefill_rows(self,rows,tokenizer):
            width=max(len(r['prompt_token_ids']) for r in rows)
            ids=torch.full((len(rows),width),tokenizer.pad_token_id,device=self.student.device,dtype=torch.long)
            mask=torch.zeros_like(ids)
            for i,row in enumerate(rows):
                n=len(row['prompt_token_ids']);ids[i,:n]=torch.tensor(row['prompt_token_ids'],device=ids.device);mask[i,:n]=1
            features=self.features(ids,mask)
            with self.inject(features,mask.bool()&features[0].lt(27)):
                return self.student(input_ids=ids,attention_mask=mask,position_ids=torch.arange(width,device=ids.device)[None].expand(len(rows),-1),use_cache=True,logits_to_keep=1)

        def save_projectors(self,directory):
            directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
            torch.save(self.bridge_checkpoint_state(),directory/'bridge.pt')

    bridge=Bridge(student,stats,p['template_prefix_ids'],p['template_suffix_ids']).eval()
    bridge.pad_token_id=AutoTokenizer.from_pretrained(p['student_checkpoint'],local_files_only=True).pad_token_id
    assert sum(x.numel() for x in bridge.bridge_parameters())==p['trainable_parameters']
    if checkpoint:bridge.load_bridge_checkpoint(checkpoint)
    return bridge


def statistics(base):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    p=json.loads((base/'protocol.json').read_text());data=Path(p['data_root'])
    rows=[json.loads(x) for x in (data/'train.jsonl').read_text().splitlines()]
    split=json.loads((data/'split.json').read_text())
    order=torch.randperm(len(split['train']),generator=torch.Generator().manual_seed(42)).tolist()[:500]
    selected=[split['train'][i] for i in order];assert not set(selected)&set(split['validation'])
    save(base/'statistics_samples.json',dict(indices=selected,samples=500,training_only=True))
    tokenizer=AutoTokenizer.from_pretrained(p['student_checkpoint'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(p['student_checkpoint'],dtype=torch.bfloat16,
        attn_implementation='sdpa',device_map={'':0},local_files_only=True).eval().requires_grad_(False)
    sums=torch.zeros((27,28),device='cuda',dtype=torch.float64);squares=torch.zeros_like(sums)
    with torch.inference_mode():
        for offset in range(0,len(selected),16):
            group=[rows[i] for i in selected[offset:offset+16]];width=max(len(r['prompt_token_ids']) for r in group)
            ids=torch.full((len(group),width),tokenizer.pad_token_id,device='cuda',dtype=torch.long);mask=torch.zeros_like(ids)
            for i,r in enumerate(group):
                n=len(r['prompt_token_ids']);ids[i,:n]=torch.tensor(r['prompt_token_ids'],device='cuda');mask[i,:n]=1
            hidden=model.model(input_ids=ids,attention_mask=mask,use_cache=False,output_hidden_states=True).hidden_states
            for layer in range(28):
                values=torch.stack([hidden[layer][i,list(range(3))+list(range(len(r['prompt_token_ids'])-25,len(r['prompt_token_ids'])-1))] for i,r in enumerate(group)]).double()
                assert torch.isfinite(values).all()
                sums[:,layer]+=values.sum((0,2));squares[:,layer]+=values.square().sum((0,2))
            save(base/'statistics_live.json',dict(completed=min(offset+16,len(selected)),total=len(selected)))
    mean=sums/(500*2048);variance=(squares/(500*2048)-mean.square()).clamp_min(0)
    stats=dict(mean=mean.float().cpu().unsqueeze(-1),std=variance.sqrt().float().cpu().unsqueeze(-1),
        broadcast_shape=[27,28,2048],layer_map=list(range(28)),
        template_token_ids=p['template_prefix_ids']+p['template_suffix_ids'][:-1],
        samples=500,pooling='training questions and hidden channels; population variance',hidden_space='decoder block input before RMSNorm')
    target=Path(p['virtual_hidden_statistics_path']);torch.save(stats,target)
    p['virtual_hidden_statistics_sha256']=sha(target);save(base/'protocol.json',p)
    save(base/'statistics_complete.json',dict(complete=True,samples=500,training_only=True,statistics_sha256=sha(target),shape=[27,28,1],broadcast_shape=[27,28,2048]))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    statistics(parser.parse_args().root.absolute())
