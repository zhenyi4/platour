"""Bias-free FP32 low-rank projector with a zero effective initial map."""
import math

import torch
from torch import nn


class LowRankLinear(nn.Module):
    def __init__(self,in_features,out_features,rank=128,alpha=128,device=None):
        super().__init__()
        if not 0<rank<=min(in_features,out_features) or not math.isfinite(alpha) or alpha<=0:
            raise ValueError('Use a valid positive rank and alpha')
        self.in_features=in_features;self.out_features=out_features
        self.rank=rank;self.alpha=float(alpha);self.scaling=self.alpha/rank
        self.A=nn.Linear(in_features,rank,bias=False,device=device,dtype=torch.float32)
        self.B=nn.Linear(rank,out_features,bias=False,device=device,dtype=torch.float32)
        nn.init.zeros_(self.B.weight)

    def forward(self,hidden):
        return self.B(self.A(hidden))*self.scaling

    def effective_weight(self):
        return (self.B.weight@self.A.weight)*self.scaling


def zero_output_initialization(maps):
    return all(isinstance(m,LowRankLinear) and not m.B.weight.count_nonzero() and
               m.A.weight.count_nonzero()>0 for m in maps)
