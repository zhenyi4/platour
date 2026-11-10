"""Full text-model GRPO with FP32 optimizer shards; no teacher, bridge or adapters."""
from contextlib import contextmanager
import hashlib
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint


def owners_for(parameters, world):
    sizes=[0]*world
    owners=[None]*len(parameters)
    for index in sorted(range(len(parameters)),key=lambda i:(-parameters[i].numel(),i)):
        owner=min(range(world),key=lambda rank:(sizes[rank],rank))
        owners[index]=owner
        sizes[owner]+=parameters[index].numel()
    return owners


def model_digest(model):
    result=hashlib.sha256()
    for name,value in model.state_dict().items():
        result.update(name.encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


@contextmanager
def checkpoint_blocks(model):
    originals=[]
    try:
        for layer in model.model.layers:
            old=layer.forward
            def forward(*args,_call=old,**kwargs):
                return checkpoint(_call,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
            originals.append((layer,old,'forward' in layer.__dict__))
            layer.forward=forward
        yield
    finally:
        for layer,forward,owned in originals:
            if owned:layer.forward=forward
            else:del layer.forward


def log_probs(model,prompt,response,checkpointing=True,chunk_tokens=32):
    if len(prompt)<2 or not response:raise ValueError('Need a prompt and response including EOS')
    ids=torch.tensor([prompt+response[:-1]],device=model.device)
    with checkpoint_blocks(model) if checkpointing and torch.is_grad_enabled() else __import__('contextlib').nullcontext():
        hidden=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(ids.shape[1],device=ids.device)[None],use_cache=False).last_hidden_state[0,len(prompt)-1:]
    labels=torch.tensor(response,device=ids.device)
    assert len(hidden)==len(labels)
    def chunk(h,y):
        logits=model.lm_head(h).float()
        return logits.gather(1,y[:,None]).squeeze(1)-logits.logsumexp(-1)
    pieces=[]
    for offset in range(0,len(response),chunk_tokens):
        h=hidden[offset:offset+chunk_tokens];y=labels[offset:offset+chunk_tokens]
        pieces.append(checkpoint(chunk,h,y,use_reentrant=False,preserve_rng_state=False)
                      if torch.is_grad_enabled() else chunk(h,y))
    return torch.cat(pieces)


class ShardedMasterAdam:
    def __init__(self,model,rank=0,world=1,lr=1e-5):
        self.named=list(model.named_parameters())
        self.params=[p for _,p in self.named]
        assert all(p.requires_grad for p in self.params)
        self.rank,self.world=rank,world
        self.owners=owners_for(self.params,world)
        self.local=[i for i,owner in enumerate(self.owners) if owner==rank]
        self.masters=[torch.nn.Parameter(self.params[i].detach().float().clone()) for i in self.local]
        self.optimizer=torch.optim.AdamW(self.masters,lr=lr,betas=(.9,.999),weight_decay=0.,foreach=False)
        self.updates=0

    def zero_grad(self):
        for p in self.params:p.grad=None
        self.optimizer.zero_grad(set_to_none=True)

    def step(self,max_norm=1.):
        device=self.params[0].device
        finite=torch.ones((),device=device,dtype=torch.int32)
        gradient_square=torch.zeros((),device=device,dtype=torch.float64)
        missing=0
        for p in self.params:
            if p.grad is None:
                missing+=1
                p.grad=torch.zeros_like(p)
            if self.world>1:dist.all_reduce(p.grad)
            finite*=torch.isfinite(p.grad).all().to(torch.int32)
            gradient_square+=p.grad.float().square().sum().double()
        if self.world>1:dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item():raise RuntimeError('Non-finite full-model gradients; no optimizer update')
        norm=float(gradient_square.sqrt())
        factor=min(1.,max_norm/(norm+1e-6))
        if norm>0:
            for master,index in zip(self.masters,self.local,strict=True):
                master.grad=self.params[index].grad.detach().float().mul_(factor)
            for p in self.params:p.grad=None
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            for master,index in zip(self.masters,self.local,strict=True):
                self.params[index].data.copy_(master.data.to(self.params[index].dtype))
            for p,owner in zip(self.params,self.owners,strict=True):
                if self.world>1:dist.broadcast(p.data,src=owner)
            self.updates+=1
        else:
            self.zero_grad()
        return dict(grad_norm=norm,optimizer_updated=norm>0,local_missing_gradients=missing)

    def state_dict(self):
        return dict(rank=self.rank,world=self.world,owners=self.owners,local=self.local,
            masters=[p.detach().cpu() for p in self.masters],optimizer=self.optimizer.state_dict(),updates=self.updates,
            names=[name for name,_ in self.named],precision='BF16 model and grads; sharded FP32 master/moments')

    def load_state_dict(self,state):
        assert state['rank']==self.rank and state['world']==self.world and state['owners']==self.owners
        assert state['names']==[name for name,_ in self.named] and state['local']==self.local
        for p,saved in zip(self.masters,state['masters'],strict=True):p.data.copy_(saved)
        self.optimizer.load_state_dict(state['optimizer'])
        self.updates=state['updates']
        for index,master in zip(self.local,self.masters,strict=True):
            torch.testing.assert_close(master.detach().to(self.params[index].dtype),self.params[index].detach(),rtol=0,atol=0)
