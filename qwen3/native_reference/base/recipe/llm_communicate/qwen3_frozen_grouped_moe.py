"""Pack frozen Qwen3 MoE experts for native Torch grouped GEMMs."""
import torch
from torch import nn
from torch.nn import functional as F


class FrozenGroupedMoe(nn.Module):
    def __init__(self,original):
        super().__init__()
        self.gate=original.gate
        self.top_k=original.top_k;self.num_experts=original.num_experts
        self.norm_topk_prob=original.norm_topk_prob
        self.gate_up=nn.Parameter(torch.stack([torch.cat([e.gate_proj.weight,e.up_proj.weight],dim=0)
            for e in original.experts]),requires_grad=False)
        self.down=nn.Parameter(torch.stack([e.down_proj.weight for e in original.experts]),requires_grad=False)
        # Preserve every expert's order and exact BF16 values.
        for i,e in enumerate(original.experts):
            a,b=self.gate_up[i].chunk(2,dim=0)
            assert torch.equal(a,e.gate_proj.weight) and torch.equal(b,e.up_proj.weight)
            assert torch.equal(self.down[i],e.down_proj.weight)
        self.requires_grad_(False)

    def forward(self,hidden_states):
        shape=hidden_states.shape;x=hidden_states.reshape(-1,shape[-1])
        router=self.gate(x)
        weights,experts=torch.topk(F.softmax(router,dim=-1,dtype=torch.float32),self.top_k,dim=-1)
        if self.norm_topk_prob:weights=weights/weights.sum(-1,keepdim=True)
        weights=weights.to(x.dtype)
        # Sort token/expert pairs; grouped_mm operates on independent expert weights.
        flat=experts.reshape(-1);order=torch.argsort(flat,stable=True)
        offsets=torch.bincount(flat,minlength=self.num_experts).cumsum(0).to(torch.int32)
        source=(order//self.top_k)
        projected=F.grouped_mm(x[source],self.gate_up.transpose(1,2),offs=offsets)
        gate,up=projected.chunk(2,dim=-1)
        value=F.grouped_mm(F.silu(gate)*up,self.down.transpose(1,2),offs=offsets)
        value=value*weights.reshape(-1)[order,None]
        restored=torch.empty_like(value);restored[order]=value
        output=restored.reshape(len(x),self.top_k,-1).sum(dim=1)
        return output.reshape(shape),router


@torch.no_grad()
def accelerate_frozen_teacher(teacher):
    assert teacher.config.model_type=='qwen3_moe'
    assert not any(p.requires_grad for p in teacher.parameters())
    checks=[];depth=len(teacher.model.layers)
    for i,layer in enumerate(teacher.model.layers):
        original=layer.mlp;packed=FrozenGroupedMoe(original)
        if i in [0,depth//2,depth-1]:
            generator=torch.Generator(device=teacher.device).manual_seed(42+i)
            x=torch.randn(1,128,teacher.config.hidden_size,device=teacher.device,dtype=torch.bfloat16,generator=generator)
            expected=original(x)[0];actual=packed(x)[0]
            relative=float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12))
            check=dict(layer=i,relative_l2=relative,max_abs_error=float((actual.float()-expected.float()).abs().max()),passed=relative<.02)
            checks.append(check);assert check['passed'],check
        layer.mlp=packed
        del original
    torch.cuda.empty_cache()
    return dict(implementation='torch.nn.functional.grouped_mm',teacher_layers=depth,
        exact_weight_copy_all_experts=True,reference='native Transformers 4.57.6 eager MoE',
        checked_layers=checks,passed=all(c['passed'] for c in checks),base_dtype='bfloat16',quantization=None)
