"""Mode-aware Qwen3 chat boundaries and compact ragged token storage."""
import hashlib
import json
from pathlib import Path

import numpy as np

INDEX_DTYPE = np.dtype([('offset','<u8'),('prefix','<u2'),('length','<u2'),
    ('thinking','u1'),('validation','u1'),('group','<u8')])


def render_record(record, tokenizer, seed=42):
    messages = record['messages']
    positions = [i for i,m in enumerate(messages) if m['role']=='assistant']
    if not positions:
        raise ValueError('Missing assistant reply')
    identity = record['uuid']
    turn = positions[int.from_bytes(hashlib.sha256(f'{seed}:{identity}'.encode()).digest()[:8],'little') % len(positions)]
    mode = record['reasoning']
    if mode not in ('on','off'):
        raise ValueError(f'Unknown reasoning mode: {mode}')
    thinking = mode=='on'
    # Earlier assistant thoughts are omitted, matching the Qwen3 chat template.
    history = [dict(role=m['role'],content=m['content']) for m in messages[:turn]]
    prefix = tokenizer.apply_chat_template(history,tokenize=False,
        add_generation_prompt=True,enable_thinking=thinking)
    answer = messages[turn]
    content = answer['content'].lstrip('\n')
    if thinking:
        reason = answer.get('reasoning_content','').strip('\n')
        reply = '<think>\n'+reason+'\n</think>\n\n'+content+'<|im_end|>\n'
    else:
        reply = content+'<|im_end|>\n'
    # All versions of the same system/user conversation share a holdout group,
    # even when UUIDs, assistant answers, selected turns, or thinking modes differ.
    users = [(m['role'],m['content']) for m in messages if m['role'] in ('system','user')]
    group = int.from_bytes(hashlib.sha256(json.dumps(users,ensure_ascii=False,
        separators=(',',':')).encode()).digest()[:8],'little')
    return prefix,reply,dict(thinking=thinking,group=group,
        validation=group%2000==0,assistant_turn=turn,uuid=identity)


def clip_tokens(prefix,reply,max_length=4096,max_reply=2048):
    # Keep every valid record. Preserve the prompt's final generation marker;
    # very long contexts are left-truncated, responses right-truncated like the
    # paper. A clipped response is not given a fabricated end-of-turn token.
    original_prefix,original_reply=len(prefix),len(reply)
    reply=reply[:max_reply]
    prefix=prefix[-(max_length-len(reply)):]
    if not prefix or not reply:
        raise ValueError('Empty prefix/reply')
    return prefix+reply,len(prefix),dict(prefix_clipped=original_prefix>len(prefix),
        reply_clipped=original_reply>len(reply))


class ChatTokens:
    def __init__(self,root):
        root=Path(root)
        self.metadata=json.loads((root/'complete.json').read_text())
        self.index=np.memmap(root/'index.bin',dtype=INDEX_DTYPE,mode='r')
        self.tokens=np.memmap(root/'tokens.bin',dtype=np.uint32,mode='r')
        assert len(self.index)==self.metadata['records']

    def __getitem__(self,index):
        row=self.index[index]
        start=int(row['offset']); length=int(row['length'])
        return np.asarray(self.tokens[start:start+length],dtype=np.int64),int(row['prefix'])


def collate(dataset,indices,pad_token_id,device):
    import torch
    items=[dataset[int(i)] for i in indices]
    width=max(len(ids) for ids,n in items);prefix_width=max(n for ids,n in items)
    ids=torch.full((len(items),width),pad_token_id,dtype=torch.long,device=device)
    attention=torch.zeros_like(ids)
    labels=torch.full_like(ids,-100)
    prefix_ids=torch.full((len(items),prefix_width),pad_token_id,dtype=torch.long,device=device)
    prefix_mask=torch.zeros_like(prefix_ids,dtype=torch.bool)
    for i,(row,n) in enumerate(items):
        length=len(row)
        ids[i,:length]=torch.as_tensor(row,device=device)
        attention[i,:length]=1
        labels[i,n:length]=ids[i,n:length]
        prefix_ids[i,:n]=ids[i,:n]
        prefix_mask[i,:n]=True
    return dict(input_ids=ids,attention_mask=attention,labels=labels,
        prefix_ids=prefix_ids,prefix_mask=prefix_mask)


def training_order(dataset,seed=42,rng=None):
    rows=np.flatnonzero(dataset.index['validation']==0)
    rng=np.random.default_rng(seed) if rng is None else rng
    return rows[rng.permutation(len(rows))]


def batch_order(dataset,global_batch=128,seed=42,smoke=False,limit=None,order='bucket'):
    if global_batch<1 or (limit is not None and limit<1) or order not in ('bucket','shuffled'):
        raise ValueError('Invalid batch size, conversation limit or order')
    rng=np.random.default_rng(seed)
    rows=training_order(dataset,seed,rng=rng)
    if limit is not None:
        if limit>len(rows):raise ValueError('Conversation limit exceeds training corpus')
        rows=rows[:limit]
    if smoke:
        # First batch mixed modes, second longest prefixes, third a genuine tail.
        off=rows[dataset.index['thinking'][rows]==0][:global_batch//2]
        on=rows[dataset.index['thinking'][rows]==1][:global_batch-len(off)]
        first=np.concatenate([off,on]);rng.shuffle(first)
        longest=rows[np.argsort(dataset.index['prefix'][rows],kind='stable')[-global_batch:]]
        tail=rows[-min(13,len(rows)%global_batch or 13):]
        return [first,longest,tail]
    if order=='shuffled':
        return [rows[start:start+global_batch] for start in range(0,len(rows),global_batch)]
    # Bound padding in both teacher-prefix and student-full forwards. Every row
    # is visited exactly once; no balance/oversampling between thinking modes.
    buckets={}
    for i in rows:
        r=dataset.index[i]
        key=(int(r['prefix'])//256,int(r['length'])//256)
        buckets.setdefault(key,[]).append(int(i))
    batches=[]; leftovers=[]
    for key in sorted(buckets):
        values=buckets[key];end=len(values)//global_batch*global_batch
        batches.extend(np.asarray(values[j:j+global_batch],dtype=np.int64) for j in range(0,end,global_batch))
        leftovers.extend(values[end:])
    batches.extend(np.asarray(leftovers[j:j+global_batch],dtype=np.int64) for j in range(0,len(leftovers),global_batch))
    rng.shuffle(batches)
    assert sum(map(len,batches))==len(rows)
    return batches
