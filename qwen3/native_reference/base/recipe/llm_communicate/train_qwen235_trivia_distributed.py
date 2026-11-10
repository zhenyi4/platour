"""Four-replica, two-GPU-per-replica TriviaQA hidden-bridge SFT."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from transformers import get_scheduler

from qwen3_235b_trivia_hidden_sft import load_bridge,save,sha
from qwen3_17b_hidden_sft import digest
from nemotron_chat_data import ChatTokens,collate


def sync_gradients(params,clip=1.):
    """Sum globally token-normalized gradients, including empty tail ranks."""
    for p in params:
        if p.grad is None:p.grad=torch.zeros_like(p)
    flat=torch.cat([p.grad.reshape(-1) for p in params])
    dist.all_reduce(flat,op=dist.ReduceOp.SUM)
    assert torch.isfinite(flat).all(),'Nonfinite global bridge gradient'
    norm=torch.linalg.vector_norm(flat)
    flat.mul_((clip/(norm+1e-6)).clamp(max=1.))
    offset=0
    for p in params:
        p.grad.copy_(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()
    return norm


def test_distributed(base):
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.set_num_threads(2);dist.init_process_group('gloo',timeout=timedelta(minutes=2))
    cases=[]
    for count in (64,44,1):
        torch.manual_seed(42)
        parameter=torch.nn.Parameter(torch.randn(3,dtype=torch.float64))
        initial=parameter.detach().clone()
        groups=[torch.arange(1,(i%5)+2,dtype=torch.float64)[:,None].repeat(1,3)*(i+1)/100 for i in range(count)]
        token_count=sum(len(x) for x in groups)
        for i in range(rank,count,world):
            x=groups[i];((x@parameter).square().sum()/token_count).backward()
        sync_gradients([parameter],clip=1.)
        reference=initial.clone().requires_grad_(True)
        (sum((x@reference).square().sum() for x in groups)/token_count).backward()
        expected=reference.grad*(1/(reference.grad.norm()+1e-6)).clamp(max=1.)
        torch.testing.assert_close(parameter.grad,expected,rtol=1e-12,atol=1e-12)
        cases.append(dict(examples=count,passed=True))
    if rank==0:save(base/'distributed_gradient_test.json',dict(passed=True,world_size=world,cases=cases,global_token_mean=True))
    dist.destroy_process_group()


def train(base,smoke=False):
    rank,world,local=[int(os.environ[n]) for n in ('RANK','WORLD_SIZE','LOCAL_RANK')]
    primary=local*2;torch.cuda.set_device(primary);torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    protocol=json.loads((base/'protocol.json').read_text())
    assert world==protocol['world_size']==4 and protocol['batch_size']==64
    assert world*protocol['micro_batch']*protocol['gradient_accumulation_steps']==64
    dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=torch.device('cuda',primary))
    output=base/('smoke' if smoke else 'train');output.mkdir(exist_ok=True)
    assert not (output/'metrics.jsonl').exists(),'Use a fresh run directory'
    c2c=protocol.get('bridge_family')=='c2c_235b'
    loader=load_bridge
    if c2c:
        from qwen235_c2c_sft import load_bridge as loader
    bridge=loader(base,devices=[primary,primary+1],student_device=primary).eval()
    params=bridge.bridge_parameters()
    if not c2c:assert all(p.count_nonzero()==0 for p in bridge.maps.parameters())
    initial=digest(bridge);hashes=[None]*world;dist.all_gather_object(hashes,initial);assert len(set(hashes))==1
    dataset=ChatTokens(Path(protocol['data_root']))
    split=json.loads((Path(protocol['data_root'])/'split.json').read_text());ids=split['train'];heldout=split['validation']
    assert len(ids)==protocol['train_examples'] and len(heldout)==protocol['heldout_examples']
    micro=protocol['micro_batch'];pad=bridge.pad_token_id
    def count(i):return int(dataset.index[i]['length']-dataset.index[i]['prefix'])
    def chunks(group):
        ordered=sorted(group,key=lambda i:int(dataset.index[i]['length']),reverse=True)
        return [ordered[i:i+micro] for i in range(0,len(ordered),micro)]
    def loss(group):return bridge.chat_loss(collate(dataset,group,pad,torch.device('cuda',primary)))[1]
    @torch.no_grad()
    def validate():
        if c2c:bridge.maps.eval()
        value=torch.zeros(2,device=primary,dtype=torch.float64)
        for chunk in chunks(heldout[rank::world]):
            n=sum(map(count,chunk));value[0]+=loss(chunk).double()*n;value[1]+=n
        dist.all_reduce(value)
        if c2c:bridge.maps.train()
        return float(value[0]/value[1])
    batch=collate(dataset,[ids[0]],pad,torch.device('cuda',primary))
    with torch.no_grad():
        zero=bridge.chat_loss(batch)[1]
        native=bridge.student(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],labels=batch['labels'],use_cache=False).loss
        torch.testing.assert_close(zero,native,atol=2e-3 if c2c else 2e-5,rtol=2e-3 if c2c else 2e-5)
    if rank==0:save(output/'sanity.json',dict(passed=True,zero_bridge_ce=float(zero),native_ce=float(native),replica_hashes=hashes))
    del batch,zero,native
    groups=[]
    for epoch in range(3):
        order=torch.randperm(len(ids),generator=torch.Generator().manual_seed(42+epoch)).tolist()
        shuffled=[ids[i] for i in order];groups.extend(shuffled[i:i+64] for i in range(0,len(shuffled),64))
    if smoke:
        longest=max(ids,key=lambda i:int(dataset.index[i]['length']))
        groups=[ids[:64],ids[64:128],[longest]]
    optimizer=torch.optim.AdamW(params,lr=protocol['learning_rate'],betas=(.9,.999),weight_decay=protocol['weight_decay'])
    matched_smoke=smoke and protocol.get('smoke_use_training_schedule',False)
    scheduler=get_scheduler('linear',optimizer,
        num_warmup_steps=protocol['warmup_steps'] if matched_smoke or not smoke else 1,
        num_training_steps=protocol['total_steps'] if matched_smoke else len(groups))
    val=validate()
    if rank==0:save(output/'initial_validation.json',dict(nll=val))
    start=time.monotonic();epochs=[];visited=[];epoch_loss=0.;epoch_tokens=0
    for step,group in enumerate(groups,1):
        tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
        tokens=sum(map(count,group));total=torch.zeros((),device=primary,dtype=torch.float64)
        for chunk in chunks(group[rank::world]):
            n=sum(map(count,chunk));value=loss(chunk)
            assert torch.isfinite(value)
            total+=value.detach().double()*n;(value*(n/tokens)).backward()
        norm=sync_gradients(params)
        assert all(p.grad is None and not p.requires_grad for m in (bridge.teacher,bridge.student) for p in m.parameters())
        lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step();dist.all_reduce(total)
        if c2c:
            for projector in bridge.maps:projector.update_temperature(step)
        visited.extend(group);epoch_loss+=float(total);epoch_tokens+=tokens
        timing=torch.tensor(time.monotonic()-tick,device=primary);dist.all_reduce(timing,op=dist.ReduceOp.MAX)
        row=dict(step=step,total_steps=len(groups),train_nll=float(total)/tokens,learning_rate=lr,grad_norm=float(norm),
            examples=len(group),response_tokens=tokens,world_size=world,gpu_count=8,global_batch_size=64,
            seconds=time.monotonic()-start,step_seconds=float(timing),peak_allocated_gib=torch.cuda.max_memory_allocated(primary)/2**30)
        if step%protocol['steps_per_epoch']==0 or (smoke and step==len(groups)):
            epoch=1 if smoke else step//protocol['steps_per_epoch'];val=validate();row['heldout_nll']=val
            state_hash=digest(bridge);hashes=[None]*world;dist.all_gather_object(hashes,state_hash);assert len(set(hashes))==1
            epochs.append(dict(epoch=epoch,step=step,train_nll=epoch_loss/epoch_tokens,heldout_nll=val))
            directory=output/f'epoch-{epoch}'
            save_checkpoint = not protocol.get('save_final_checkpoint_only', False) or step == len(groups)
            if rank==0:
                if save_checkpoint:
                    bridge.save_projectors(directory)
                    if not protocol.get('save_final_checkpoint_only', False):
                        torch.save(dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step),directory/'optimizer.pt')
                save(output/'epochs.json',epochs)
            dist.barrier()
            if save_checkpoint and not protocol.get('save_final_checkpoint_only', False):
                torch.save(dict(cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(primary)),directory/f'rng.rank-{rank:02d}.pt')
            epoch_loss=0.;epoch_tokens=0
        if rank==0:
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            save(output/'live.json',row);print(json.dumps(row),flush=True)
    if not smoke:
        assert len(groups)==protocol['total_steps'] and len(visited)==3*len(ids)
        assert all(sorted(visited[e*len(ids):(e+1)*len(ids)])==sorted(ids) for e in range(3))
    final=digest(bridge);hashes=[None]*world;dist.all_gather_object(hashes,final);assert len(set(hashes))==1 and final!=initial
    if rank==0:
        checkpoint=output/f'epoch-{1 if smoke else 3}/bridge.pt'
        save(output/'complete.json',dict(complete=True,smoke=smoke,steps=len(groups),epochs=1 if smoke else 3,
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),final_sha256=final,initial_sha256=initial,
            seconds=time.monotonic()-start,backbones_frozen_verified=True,data_split_sha256=sha(Path(protocol['data_root'])/'split.json'),
            global_response_token_mean=True,global_batch_size=64,world_size=world,gpu_count=8,replica_exact_weights=True,
            each_train_example_once_per_epoch=not smoke,singleton_tail_verified=smoke))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--smoke',action='store_true');parser.add_argument('--test-distributed',action='store_true')
    args=parser.parse_args()
    if args.test_distributed:test_distributed(args.root.absolute())
    else:train(args.root.absolute(),args.smoke)
