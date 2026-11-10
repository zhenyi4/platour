"""Low-rank hidden projectors for the unchanged dense PLatour GRPO policy."""
import math
import types
import torch
from torch import nn
from low_rank_bridge import LowRankLinear,zero_output_initialization

def attach(bridge,p):
    rank=p['low_rank_rank'];alpha=p['low_rank_alpha']
    bridge.maps=nn.ModuleList([LowRankLinear(m.in_features,m.out_features,rank,alpha,device=bridge.student.device) for m in bridge.maps])
    def initialize(self,seed=42):
        generator=torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for m in self.maps:
                values=torch.empty(m.A.weight.shape,dtype=torch.float32).uniform_(-1/math.sqrt(m.in_features),1/math.sqrt(m.in_features),generator=generator)
                m.A.weight.copy_(values.to(m.A.weight.device));m.B.weight.zero_()
        assert zero_output_initialization(self.maps)
        self.initialization='zero'
        return dict(kind='low-rank; A Kaiming uniform, B zero',seed=seed,rank=rank,alpha=alpha,
                    scaling=alpha/rank,constant_gate=1.,sft_checkpoint_loaded=False,effective_projection_zero=True)
    original=bridge.metadata
    def metadata(self):
        return dict(original(),projection_type='low_rank',rank=rank,alpha=alpha,scaling=alpha/rank,
                    initialization='A Kaiming uniform; B zero; effective projection zero',dropout=0.)
    bridge.initialize_for_grpo=types.MethodType(initialize,bridge)
    bridge.metadata=types.MethodType(metadata,bridge)
    assert sum(x.numel() for x in bridge.bridge_parameters())==p['trainable_parameters']
    return bridge
